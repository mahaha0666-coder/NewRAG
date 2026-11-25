"""Versioned drawing features and evidence-returning, development-only matching.

This store is not a route model or a source of automatically confirmed labels.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from collections import Counter
from contextlib import closing
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .document_schema import COLLECTIONS, DrawingDocument
from .part_dossiers import build_dossiers


VERSION = "drawing_features_v1"
UNKNOWN_VALUES = {"", "unknown", "unspecified", "null", "n/a", "not_visible", "\u672a\u77e5", "\u65e0\u6cd5\u786e\u5b9a"}
# Only explicit categorical fields are aliased; free text remains verbatim.
ALIASES = {
    "hole": "hole", "\u5b54": "hole", "through hole": "through_hole",
    "through_hole": "through_hole", "\u901a\u5b54": "through_hole",
    "blind hole": "blind_hole", "blind_hole": "blind_hole", "\u76f2\u5b54": "blind_hole",
    "slot": "slot", "\u69fd": "slot", "chamfer": "chamfer", "\u5012\u89d2": "chamfer",
    "diameter": "diameter", "\u76f4\u5f84": "diameter",
    "radius": "radius", "\u534a\u5f84": "radius",
}
MATCH_FIELDS = {
    "features": ("feature_type",),
    "dimensions": ("dimension_type", "value", "unit", "upper_deviation", "lower_deviation"),
    "threads": ("designation", "pitch", "depth", "handedness"),
    "tolerances": ("tolerance_type", "upper_deviation", "lower_deviation", "standard"),
    "gdt": ("symbol", "frame", "value", "unit"),
    "surface_finish": ("finish_type", "value", "unit"),
    "welding_symbols": ("symbol", "size", "side"),
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def checksum(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def canonical_attributes(kind: str, attributes: dict) -> dict:
    result = dict(attributes)
    for key in ("feature_type", "dimension_type"):
        if isinstance(result.get(key), str):
            value = " ".join(unicodedata.normalize("NFKC", result[key]).casefold().split())
            result[key] = ALIASES.get(value, value)
    unit = result.get("unit")
    unit = unicodedata.normalize("NFKC", unit).strip().casefold() if isinstance(unit, str) else unit
    # No guessed drawing-wide units, and no conversion of thread pitch or GD&T frames.
    factors = {"mm": "1", "\u6beb\u7c73": "1", "cm": "10", "m": "1000", "in": "25.4", "inch": "25.4"}
    if kind == "dimensions" and isinstance(unit, str) and unit in factors:
        converted = {}
        for key in ("value", "upper_deviation", "lower_deviation"):
            value = result.get(key)
            if value is None:
                continue
            try:
                number = Decimal(str(value)) * Decimal(factors[unit])
                if not number.is_finite():
                    raise InvalidOperation
                converted[key] = format(number.normalize(), "f")
            except InvalidOperation:
                return result  # Preserve the whole original unit/value bundle on ambiguity.
        result.update(converted)
        result["unit"] = "mm"
    return result


def observed(value: object) -> bool:
    return value is not None and not (isinstance(value, str) and value.strip().casefold() in UNKNOWN_VALUES)


def build_feature_snapshot(document: DrawingDocument) -> dict:
    document = DrawingDocument.model_validate(document.model_dump(mode="json"))
    payload = document.model_dump(mode="json")
    records = []
    for entity in sorted(document.entities(), key=lambda row: row.id):
        attributes = canonical_attributes(entity.kind, entity.attributes)
        fields = {key: attributes[key] for key in MATCH_FIELDS.get(entity.kind, ())
                  if observed(attributes.get(key))}
        # A unit alone is not a description of a feature.
        if not set(fields) - {"unit"}:
            fields = {}
        records.append({"entity_id": entity.id, "kind": entity.kind,
                        "canonical_attributes": attributes, "match_fields": fields,
                        "source_entity": entity.model_dump(mode="json"),
                        "review_status": "unreviewed"})
    dossiers = build_dossiers(document)
    audit = {
        "semantic_scope": document.configuration.get("semantic_scope", "not_requested"),
        "semantic_page_statuses": {str(page.page_number): page.semantic_status for page in document.pages},
        "extraction_status": document.extraction_status,
        "empty_collections_mean": "unknown_or_out_of_scope_never_negative_training_labels",
        "collection_counts": {key: len(getattr(document, key)) for key in COLLECTIONS},
        "source_entities": len(document.entities()), "feature_records": len(records),
        "relationships": len(document.relationships),
        "matchable_records": sum(bool(row["match_fields"]) for row in records),
        "without_locations": [e.id for e in document.entities() if not e.locations],
        "without_evidence_refs": [e.id for e in document.entities() if not e.evidence_ids and not e.raw_result_refs],
        "unassigned_features": dossiers["unassigned_feature_ids"],
        "unassigned_bom": dossiers["unassigned_bom_ids"],
        "ambiguous_feature_owners": sorted({f["feature_id"] for d in dossiers["dossiers"]
                                             for f in d["features"] if f["ownership_ambiguous"]}),
        "confirmed_records": 0, "feature_accuracy": None, "route_accuracy": None,
    }
    result = {"schema_version": VERSION, "document_sha256": checksum(payload),
              "source_sha256": document.source_sha256, "document": payload,
              "records": records, "dossiers": dossiers, "audit": audit,
              "status": "development_candidates_not_training_gold"}
    return {"snapshot_id": checksum(result), **result}


def validate_snapshot(snapshot: dict) -> None:
    expected = build_feature_snapshot(DrawingDocument.model_validate(snapshot["document"]))
    if snapshot != expected:
        raise ValueError("Feature snapshot is stale, modified, or uses an unsupported version")


def save_document(document: DrawingDocument, output: Path, *, memory_path: Path | None = None,
                  expected_snapshot: str | None = None) -> dict:
    """All parsing entry points publish the same lossless feature sidecar."""
    if expected_snapshot is not None and memory_path is None:
        raise ValueError("expected_snapshot requires memory_path")
    snapshot = build_feature_snapshot(document)
    for name in ("drawing.json", "training_features.json"):
        if (output / name).exists():
            raise FileExistsError(output / name)
    for name, payload in (("drawing.json", snapshot["document"]), ("training_features.json", snapshot)):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2, allow_nan=False)
    if memory_path is not None:
        memory = FeatureMemory(memory_path)
        if expected_snapshot is None:
            memory.ingest(snapshot)
        else:
            memory.replace_current(snapshot, expected=expected_snapshot)
    return snapshot


class FeatureMemory:
    """Transactional local cache; retain old snapshots and never inherit their reviews."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection, connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id TEXT PRIMARY KEY, source TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS current_documents (
                    source TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews (
                    snapshot_id TEXT NOT NULL, entity_id TEXT NOT NULL,
                    verdict TEXT NOT NULL, reviewer TEXT NOT NULL, reference TEXT NOT NULL,
                    PRIMARY KEY (snapshot_id, entity_id));
                CREATE TABLE IF NOT EXISTS review_events (
                    id INTEGER PRIMARY KEY, snapshot_id TEXT NOT NULL, entity_id TEXT NOT NULL,
                    verdict TEXT NOT NULL, reviewer TEXT NOT NULL, reference TEXT NOT NULL,
                    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            """)

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def ingest(self, snapshot: dict) -> str:
        validate_snapshot(snapshot)
        identifier, source = snapshot["snapshot_id"], snapshot["source_sha256"]
        with closing(self.connect()) as connection, connection:
            connection.execute("INSERT OR IGNORE INTO snapshots VALUES (?, ?, ?)",
                               (identifier, source, canonical_json(snapshot)))
            # Re-importing an older artifact must not silently roll back the active version.
            current = connection.execute("SELECT snapshot_id FROM current_documents WHERE source=?", (source,)).fetchone()
            if current and current[0] != identifier:
                raise ValueError("Existing source revision: use replace_current with its expected snapshot ID")
            connection.execute("INSERT OR IGNORE INTO current_documents VALUES (?, ?)", (source, identifier))
        return identifier

    def replace_current(self, snapshot: dict, *, expected: str) -> str:
        validate_snapshot(snapshot)
        identifier, source = snapshot["snapshot_id"], snapshot["source_sha256"]
        with closing(self.connect()) as connection, connection:
            changed = connection.execute(
                "UPDATE current_documents SET snapshot_id=? WHERE source=? AND snapshot_id=?",
                (identifier, source, expected)).rowcount
            if changed != 1:
                raise ValueError("Stale expected snapshot; current features changed")
            connection.execute("INSERT OR IGNORE INTO snapshots VALUES (?, ?, ?)",
                               (identifier, source, canonical_json(snapshot)))
        return identifier

    def load(self, identifier: str, *, current_only: bool = True) -> dict:
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT payload FROM snapshots WHERE id=?", (identifier,)).fetchone()
            current = connection.execute("SELECT 1 FROM current_documents WHERE snapshot_id=?", (identifier,)).fetchone()
        if row is None or (current_only and current is None):
            raise ValueError("Missing or superseded feature snapshot")
        result = json.loads(row[0])
        validate_snapshot(result)
        return result

    def review(self, identifier: str, entity_id: str, *, verdict: str, reviewer: str, reference: str) -> None:
        """Import an explicit external review; model confidence is never a review signal."""
        snapshot = self.load(identifier)
        if verdict not in {"correct", "incorrect", "uncertain"} or not reviewer.strip() or not reference.strip():
            raise ValueError("Explicit verdict, reviewer and external review reference are required")
        if entity_id not in {row["entity_id"] for row in snapshot["records"]}:
            raise ValueError("Review references an unknown entity")
        with closing(self.connect()) as connection, connection:
            connection.execute("INSERT INTO review_events (snapshot_id, entity_id, verdict, reviewer, reference) VALUES (?, ?, ?, ?, ?)",
                               (identifier, entity_id, verdict, reviewer, reference))
            connection.execute("INSERT OR REPLACE INTO reviews VALUES (?, ?, ?, ?, ?)",
                               (identifier, entity_id, verdict, reviewer, reference))

    def match(self, query: dict, history: list[dict], *, query_group: str,
              query_entity_ids: set[str], limit: int = 10, diagnostic_unreviewed: bool = False) -> dict:
        """Compare explicit entity scopes against a caller-supplied training-fold allowlist.

        Group/split provenance is supplied by a frozen manifest, never inferred from names.
        Scores are descriptor overlap, not accuracy or evidence of physical ownership.
        """
        validate_snapshot(query)
        if not query_group.strip() or limit < 1:
            raise ValueError("Known physical group and positive limit are required")
        by_id = {row["entity_id"]: row for row in query["records"]}
        if not query_entity_ids <= by_id.keys():
            raise ValueError("Unknown query entity scope")
        query_rows = [by_id[key] for key in sorted(query_entity_ids)]
        hits, excluded = [], Counter()
        with closing(self.connect()) as connection:
            for entry in history:
                if entry.get("split") != "train" or not entry.get("group_id") or entry["group_id"] == query_group:
                    excluded["split_or_group"] += 1
                    continue
                stored = self.load(entry["snapshot_id"])
                if stored["source_sha256"] == query["source_sha256"]:
                    excluded["same_source"] += 1
                    continue
                verdicts = {entity: (verdict, reviewer, reference) for entity, verdict, reviewer, reference in
                            connection.execute("SELECT entity_id, verdict, reviewer, reference FROM reviews WHERE snapshot_id=?",
                                               (entry["snapshot_id"],))}
                candidates = {row["entity_id"]: row for row in stored["records"]}
                scope = set(entry["entity_ids"])
                if not scope <= candidates.keys():
                    raise ValueError("Unknown historical entity scope")
                for entity_id in sorted(scope):
                    candidate = candidates[entity_id]
                    review = verdicts.get(entity_id)
                    if review and review[0] != "correct":
                        continue
                    if not review and not diagnostic_unreviewed:
                        excluded["unreviewed_entity"] += 1
                        continue
                    for row in query_rows:
                        left, right = row["match_fields"], candidate["match_fields"]
                        if row["kind"] != candidate["kind"] or not left or not right:
                            continue
                        matched = sorted(key for key in left.keys() & right.keys() if left[key] == right[key])
                        if not set(matched) - {"unit"}:
                            continue
                        conflicts = sorted(key for key in left.keys() & right.keys() if left[key] != right[key])
                        hits.append({"query_entity_id": row["entity_id"], "snapshot_id": entry["snapshot_id"],
                                     "history_entity_id": entity_id, "matched_fields": matched,
                                     "conflicting_fields": conflicts, "missing_fields": sorted(left.keys() ^ right.keys()),
                                     "descriptor_overlap": len(matched) / len(left.keys() | right.keys()),
                                     "history_review": {"verdict": review[0], "reviewer": review[1], "reference": review[2]} if review else None,
                                     "query_evidence": row["source_entity"], "history_evidence": candidate["source_entity"],
                                     "needs_review": True})
        hits.sort(key=lambda hit: (-hit["descriptor_overlap"], hit["snapshot_id"], hit["history_entity_id"], hit["query_entity_id"]))
        return {"status": "development_diagnostic" if diagnostic_unreviewed else "reviewed_history_candidates",
                "matches": hits[:limit], "excluded": dict(excluded), "feature_accuracy": None,
                "route_accuracy": None, "automatic_adoption": False}
