"""Compose parts, dimensions, symbols and literal requirements into auditable dossiers."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any

from .geometry import TOLERANCE, GeometryIndex, bbox, topology_groups
from .symbols import CONTEXT_OBJECTS, build_symbols, literal_text

SCHEMA = "native_dwg_understanding.v1"


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _dimension(row: dict, display: dict | None = None) -> dict:
    raw = row.get("raw_text", "")
    literal = literal_text(raw).strip()
    if display:
        if display["dimension_source_ref"] != row["source_ref"] or any(
            r["source_ref"][key] != row["source_ref"][key]
            for r in display["texts"]
            for key in ("source_sha256", "response_sha256")
        ):
            raise ValueError("Dimension display refers to different evidence")
        text = " / ".join(r["text"] for r in display["texts"])
        display_source = "native_block_text"
    elif literal and "<>" not in literal:
        text, display_source = literal, "literal_override"
    else:
        text, display_source = "尺寸格式待核验", "unavailable"
    category = "dimension"
    if row["native_kind"] == "DIMENSION_RADIUS":
        category = "radius"
    elif row["native_kind"] == "DIMENSION_DIAMETER" or "Ø" in text + literal:
        category = "diameter"
    elif "ANG" in row["native_kind"]:
        category = "angle"
    elif re.search(r"±|[Hh][1-9]|\\S", text + raw):
        category = "tolerance"
    return {
        **row,
        "display_text": text,
        "display_text_source": display_source,
        "display_text_evidence": display,
        "display_text_verified": display_source != "unavailable",
        "category": category,
        "anchor": row.get("cad_coordinates", {}).get("text_midpt"),
        "topology_part_candidates": [],
        "group_ids": [],
    }


def _requirements(data: dict, symbols: list[dict], parts: list[dict]) -> list[dict]:
    inputs = [
        {
            "id": r["id"],
            "raw_text": r["plain_text"],
            "source_ref": r["source_ref"],
            "anchor": r.get("native_origin"),
            "semantics": r.get("semantics", {}),
        }
        for r in data.get("requirements", [])
    ]
    inputs += [
        {
            "id": r["id"],
            "raw_text": "\n".join(t["text"] for t in r["texts"]),
            "source_ref": r["source_ref"],
            "anchor": next((t["point"] for t in r["texts"] if t["point"]), None),
            "semantics": {},
        }
        for r in symbols
        if r["category"] == "note"
    ]
    facts = []
    for block in inputs:
        for ordinal, line in enumerate(block["raw_text"].splitlines()):
            if not line.strip() or line.strip() == "技术要求":
                continue
            content = line.strip()
            normalized = content.replace("，", ",").replace("；", ";").rstrip(";；。")
            semantics = [
                r
                for r in block["semantics"].get("clauses", [])
                if normalized in r["text"] or r["text"] in normalized
            ]
            part_mentions = []
            for part in parts:
                fields = part["fields"] or part.get("field_candidates", {})
                # A short generic name such as “板/套” cannot resolve an object mention.
                name = fields.get("name", "")
                if len(name) >= 3 and name in content:
                    part_mentions.append(
                        {
                            "part_id": part["id"],
                            "evidence_text": name,
                            "basis": "literal_name_mention_only",
                            "physical_scope_confirmed": False,
                        }
                    )
            order = []
            match = re.search(r"焊后\s*(磨平|加工|抛光|检验)", content)
            if match:
                order.append(
                    {
                        "before": "焊接",
                        "after": match[1],
                        "evidence_text": match[0],
                        "basis": "explicit_after_welding_literal",
                        "scope_confirmed": False,
                    }
                )
            facts.append(
                {
                    "id": f"clause:{block['id']}:{ordinal}",
                    "block_id": block["id"],
                    "raw_text": line,
                    "text": content,
                    "source_ref": block["source_ref"],
                    "anchor": block["anchor"],
                    "localization": "block_origin_only_no_clause_bbox",
                    "actions": [
                        a for r in semantics for a in r.get("action_mentions", [])
                    ],
                    "standards": [s for r in semantics for s in r.get("standards", [])],
                    "literal_order": order,
                    "part_candidates": part_mentions,
                    "scope": "unresolved",
                    "confidence": None,
                    "relation_confidence": None,
                    "hard_constraint_eligible": False,
                }
            )
    return facts


def build_understanding(
    data: dict[str, Any],
    native_paths: dict | None = None,
    dimension_texts: dict | None = None,
) -> dict[str, Any]:
    """Derive a new revision without modifying any frozen input bytes or labels."""
    source_sha = data["source"]["sha256"]
    for key in ("parts", "callouts", "dimensions", "requirements", "native_symbols"):
        for row in data.get(key, []):
            if row["source_ref"]["source_sha256"] != source_sha:
                raise ValueError("Cross-source DWG evidence rejected")
    geometry = data.get("geometry", [])
    for row in geometry:
        refs = row.get("source_refs", [row.get("source_ref")])
        if any(r is None or r["source_sha256"] != source_sha for r in refs):
            raise ValueError("Cross-source geometry rejected")
    if len({r["id"] for r in geometry}) != len(geometry):
        raise ValueError("Duplicate geometry identity")
    parts = deepcopy(data.get("parts", []))
    drawing_fields: dict[str, list[dict]] = defaultdict(list)
    for row in data.get("native_symbols", []):
        if row["native_kind"] != "TDMTITLE":
            continue
        for key, candidates in (
            row.get("closed_cell_fields", {}).get("fields", {}).items()
        ):
            for candidate in candidates:
                if candidate.get("raw_value") is not None:
                    drawing_fields[key].append(
                        {
                            "value": candidate["raw_value"],
                            "status": candidate["status"],
                            "source_ref": row["source_ref"],
                            "native_cell_bbox": candidate.get("value_cell_cad_bbox"),
                            "binding_basis": candidate["binding_basis"],
                            "value_evidence": candidate["value_evidence"],
                            "physical_scope_confirmed": False,
                            "confidence": None,
                        }
                    )
    groups, group_of = topology_groups(geometry, parts)
    group_map = {r["id"]: r for r in groups}
    index = GeometryIndex(geometry)
    geometry_owners: dict[str, set[str]] = defaultdict(set)
    for part in parts:
        part["parent_drawing_id"] = "drawing:" + source_sha
        part["parent_relation_basis"] = (
            "same_source_BOM_membership_not_confirmed_assembly"
        )
        original = set(part["geometry_ids"])
        if not original.issubset(index.rows):
            raise ValueError("Part refers to missing geometry")
        group_ids = sorted({group_of[i] for i in original if i in group_of})
        expanded = set(original)
        for identifier in group_ids:
            group = group_map[identifier]
            # A multi-owner or branched graph is context only. Never flood-highlight it.
            if (
                group["part_candidates"] == [part["id"]]
                and not group["branch_endpoints"]
            ):
                expanded.update(group["geometry_ids"])
        part.update(
            display_geometry_ids=sorted(expanded),
            topology_group_ids=group_ids,
            dimension_links=[],
            symbol_ids=[],
            requirement_ids=[],
            display_bbox=bbox([p for i in expanded for p in index.rows[i]["points"]]),
        )
        for identifier in expanded:
            geometry_owners[identifier].add(part["id"])
    by_part = {r["id"]: r for r in parts}
    dimensions = [
        _dimension(r, (dimension_texts or {}).get(r["id"]))
        for r in data.get("dimensions", [])
    ]
    for dimension in dimensions:
        if not set(dimension["geometry_ids"]).issubset(index.rows):
            raise ValueError("Dimension refers to missing geometry")
        native_hits: dict[str, list[str]] = defaultdict(list)
        for identifier in dimension["geometry_ids"]:
            for part_id in geometry_owners[identifier]:
                native_hits[part_id].append(identifier)
        for part_id, ids in native_hits.items():
            dimension["topology_part_candidates"].append(
                {
                    "part_id": part_id,
                    "geometry_ids": sorted(ids),
                    "basis": "existing_dimension_geometry_chain",
                    "status": "candidate_not_physical_scope",
                }
            )
        coordinates = dimension.get("cad_coordinates", {})
        if (
            dimension["native_kind"] in {"DIMENSION_LINEAR", "DIMENSION_ALIGNED"}
            and all(k in coordinates for k in ("xline1_pt", "xline2_pt"))
            and math.dist(coordinates["xline1_pt"], coordinates["xline2_pt"])
            > TOLERANCE
        ):
            first, second = (
                index.contacts(coordinates[f"xline{i}_pt"]) for i in (1, 2)
            )
            common = {group_of[i] for i in first if i in group_of} & {
                group_of[i] for i in second if i in group_of
            }
            dimension["group_ids"] = sorted(common)
            for identifier in sorted(common):
                group = group_map[identifier]
                for part_id in group["part_candidates"]:
                    if not any(
                        c["part_id"] == part_id
                        for c in dimension["topology_part_candidates"]
                    ):
                        dimension["topology_part_candidates"].append(
                            {
                                "part_id": part_id,
                                "group_id": identifier,
                                "geometry_ids": sorted(
                                    set(first + second) & set(group["geometry_ids"])
                                ),
                                "measurement_origin_witnesses": [
                                    {
                                        "point": coordinates["xline1_pt"],
                                        "geometry_ids": first,
                                    },
                                    {
                                        "point": coordinates["xline2_pt"],
                                        "geometry_ids": second,
                                    },
                                ],
                                "basis": "both_native_measurement_origins_on_one_endpoint_group",
                                "status": "candidate_not_physical_scope",
                            }
                        )
        for candidate in dimension["topology_part_candidates"]:
            by_part[candidate["part_id"]]["dimension_links"].append(
                {"dimension_id": dimension["id"], **candidate}
            )
        dimension["display_geometry_ids"] = sorted(
            set(dimension["geometry_ids"])
            | {
                i
                for candidate in dimension["topology_part_candidates"]
                for i in candidate["geometry_ids"]
            }
        )
    symbols = build_symbols(data, index, native_paths)
    for symbol in symbols:
        owners: dict[str, list[str]] = defaultdict(list)
        for identifier in symbol["geometry_ids"]:
            for part_id in geometry_owners[identifier]:
                owners[part_id].append(identifier)
        symbol["part_candidates"] = [
            {
                "part_id": p,
                "geometry_ids": sorted(ids),
                "basis": "preview_contact_to_native_topology_candidate",
            }
            for p, ids in sorted(owners.items())
        ]
        for part_id in owners:
            by_part[part_id]["symbol_ids"].append(symbol["id"])
    datum_refs = defaultdict(list)
    for symbol in symbols:
        if symbol["category"] == "datum":
            for text in symbol["texts"]:
                if re.fullmatch("[A-Z]", text["text"].strip()):
                    datum_refs[text["text"].strip()].append(symbol["id"])
    for symbol in symbols:
        symbol["datum_candidates"] = [
            {
                "label": t["text"].strip(),
                "symbol_ids": datum_refs[t["text"].strip()],
                "basis": "same_source_literal_datum_label",
                "relation_confirmed": False,
            }
            for t in symbol["texts"]
            if symbol["category"] == "gdt" and t["text"].strip() in datum_refs
        ]
    requirements = _requirements(data, symbols, parts)
    for row in requirements:
        for candidate in row["part_candidates"]:
            by_part[candidate["part_id"]]["requirement_ids"].append(row["id"])
    views = deepcopy(data.get("view_candidates", []))
    for symbol in symbols:
        if symbol["category"] == "section":
            labels = sorted(
                {
                    t["text"].strip()
                    for t in symbol["texts"]
                    if re.fullmatch("[A-Z]", t["text"].strip())
                }
            )
            views.append(
                {
                    "id": "cut:" + symbol["id"],
                    "label": " / ".join(labels),
                    "source_ref": symbol["source_ref"],
                    "native_origin": symbol["bbox"][:2] if symbol["bbox"] else None,
                    "type": "cutting_marker",
                    "view_bbox": None,
                    "symbol_id": symbol["id"],
                    "physical_scope_confirmed": False,
                }
            )
    unresolved = []
    for part in parts:
        fields = part["fields"] or part.get("field_candidates", {})
        reasons = ["完整零件边界待核验"]
        if not part["display_geometry_ids"]:
            reasons.append("件号尚未关联可核对的几何")
        if not part["dimension_links"]:
            reasons.append("尺寸与本零件的证据链尚未闭合")
        if not part["fields"]:
            reasons.append("续表字段来源待核验")
        unresolved.append(
            {
                "kind": "part",
                "id": part["id"],
                "label": fields.get("name", "明细行"),
                "reasons": reasons,
            }
        )
    exposed = {s["source_ref"]["native_handle"] for s in symbols}
    for row in data.get("native_symbols", []):
        if (
            row["native_kind"] not in CONTEXT_OBJECTS
            and row["source_ref"]["native_handle"] not in exposed
        ):
            unresolved.append(
                {
                    "kind": "unavailable_symbol",
                    "id": "unavailable:" + row["source_ref"]["native_handle"],
                    "label": "原生标注待解码",
                    "source_ref": row["source_ref"],
                    "reasons": ["原对象保留，尚无足够文字或图形供展示"],
                }
            )
    counts = {
        "parts": len(parts),
        "geometry_groups": len(groups),
        "dimensions": len(dimensions),
        "dimensions_with_native_display": sum(
            r["display_text_source"] == "native_block_text" for r in dimensions
        ),
        "symbols": len(symbols),
        "requirement_clauses": len(requirements),
        "view_labels": len(views),
        "parts_with_geometry": sum(bool(p["display_geometry_ids"]) for p in parts),
        "parts_with_expanded_geometry": sum(
            len(p["display_geometry_ids"]) > len(p["geometry_ids"]) for p in parts
        ),
        "dimension_part_candidates": sum(
            bool(d["topology_part_candidates"]) for d in dimensions
        ),
        "symbols_with_geometry_contacts": sum(bool(s["geometry_ids"]) for s in symbols),
        "symbols_with_part_candidates": sum(
            bool(s["part_candidates"]) for s in symbols
        ),
        "unavailable_symbols": sum(
            r["kind"] == "unavailable_symbol" for r in unresolved
        ),
        "recovered_symbol_polygons": sum(
            p.get("filled", False) for s in symbols for p in s["paths"]
        ),
        "physical_owners_confirmed": 0,
        "hard_operation_constraints": 0,
    }
    result = {
        "schema": SCHEMA,
        "source": {k: v for k, v in data["source"].items() if k != "path"},
        "drawing_fields": dict(drawing_fields),
        "base_semantic_sha256": data.get("semantic_sha256"),
        "parts": parts,
        "groups": groups,
        "dimensions": dimensions,
        "symbols": symbols,
        "requirements": requirements,
        "views": views,
        "unresolved": unresolved,
        "counts": counts,
        "policy": {
            "route_labels_read": False,
            "route_model_changed": False,
            "ocr_used": False,
            "gpu_used": False,
            "physical_ownership_inferred": False,
            "unknown_scope_is_hard_constraint": False,
        },
    }
    result["semantic_sha256"] = fingerprint(result)
    return result
