"""Local, traceable evidence records used by both MechRAG pipelines.

The original notebooks indexed a whole image or a long PDF as one item.  This
module makes the retrieval unit a CAD feature.  Regions from different views
of the same CAD feature are kept in one :class:`EvidenceUnit`, so repeated
views cannot be mistaken for independent evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class EvidenceConflictError(ValueError):
    """Raised when two views disagree about a supposedly identical feature."""


@dataclass(frozen=True)
class BoundingBox:
    """Normalized image coordinates in ``[0, 1]``."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Bounding-box coordinates must be finite")
        if not (0 <= self.x_min < self.x_max <= 1):
            raise ValueError("Expected 0 <= x_min < x_max <= 1")
        if not (0 <= self.y_min < self.y_max <= 1):
            raise ValueError("Expected 0 <= y_min < y_max <= 1")

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def iou(self, other: "BoundingBox") -> float:
        width = max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        height = max(0.0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        intersection = width * height
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)


@dataclass(frozen=True)
class EvidenceSource:
    """Provenance for one extracted fact or feature."""

    uri: str
    page: int | None = None
    record_id: str | None = None
    extractor: str | None = None

    def label(self) -> str:
        suffix = []
        if self.page is not None:
            suffix.append(f"page={self.page}")
        if self.record_id:
            suffix.append(f"record={self.record_id}")
        return self.uri + ("#" + ",".join(suffix) if suffix else "")


@dataclass(frozen=True)
class ViewRegion:
    """One visible projection of a CAD feature.

    ``embedding`` is optional and is normally produced by a frozen CLIP
    encoder from the cropped region.  It is stored here so the trainable MLP
    never needs to update the visual encoder.
    """

    view_id: str
    image_uri: str
    bbox: BoundingBox
    cad_face_ids: tuple[str, ...] = ()
    embedding: tuple[float, ...] = ()

    def identity_key(self) -> tuple[Any, ...]:
        return (self.view_id, self.image_uri, *self.bbox.as_tuple())


@dataclass(frozen=True)
class EvidenceUnit:
    """Smallest retrievable and citable engineering evidence unit.

    This directly implements the record proposed by the user:

    ``X = {part_id, feature_id, CAD faces, multi-view local regions,
    geometric properties, source}``.
    """

    part_id: str
    feature_id: str
    feature_type: str
    cad_face_ids: tuple[str, ...] = ()
    view_regions: tuple[ViewRegion, ...] = ()
    geometric_properties: Mapping[str, Any] = field(default_factory=dict)
    sources: tuple[EvidenceSource, ...] = ()
    summary: str = ""
    family_id: str | None = None
    cost: float = 1.0

    def __post_init__(self) -> None:
        if not self.part_id.strip() or not self.feature_id.strip():
            raise ValueError("part_id and feature_id are required")
        if self.cost <= 0 or not math.isfinite(self.cost):
            raise ValueError("Evidence cost must be a positive finite number")

    @property
    def evidence_id(self) -> str:
        safe_part = self.part_id.replace(" ", "_")
        safe_feature = self.feature_id.replace(" ", "_")
        return f"{safe_part}:{safe_feature}"

    def prompt_block(self) -> str:
        properties = json.dumps(
            dict(self.geometric_properties), ensure_ascii=False, sort_keys=True
        )
        regions = [
            {
                "view": region.view_id,
                "bbox": [round(value, 5) for value in region.bbox.as_tuple()],
                "faces": list(region.cad_face_ids),
                "image": region.image_uri,
            }
            for region in self.view_regions
        ]
        sources = [source.label() for source in self.sources]
        return (
            f"[E:{self.evidence_id}] part={self.part_id}; feature={self.feature_id}; "
            f"type={self.feature_type}; CAD_faces={list(self.cad_face_ids)}; "
            f"regions={json.dumps(regions, ensure_ascii=False)}; "
            f"properties={properties}; sources={sources}; summary={self.summary}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "feature_id": self.feature_id,
            "feature_type": self.feature_type,
            "cad_face_ids": list(self.cad_face_ids),
            "view_regions": [
                {
                    "view_id": region.view_id,
                    "image_uri": region.image_uri,
                    "bbox": list(region.bbox.as_tuple()),
                    "cad_face_ids": list(region.cad_face_ids),
                    "embedding": list(region.embedding),
                }
                for region in self.view_regions
            ],
            "geometric_properties": dict(self.geometric_properties),
            "sources": [
                {
                    "uri": source.uri,
                    "page": source.page,
                    "record_id": source.record_id,
                    "extractor": source.extractor,
                }
                for source in self.sources
            ],
            "summary": self.summary,
            "family_id": self.family_id,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceUnit":
        regions = []
        for region in data.get("view_regions", []):
            bbox = region["bbox"]
            regions.append(
                ViewRegion(
                    view_id=str(region["view_id"]),
                    image_uri=str(region["image_uri"]),
                    bbox=BoundingBox(*map(float, bbox)),
                    cad_face_ids=tuple(map(str, region.get("cad_face_ids", []))),
                    embedding=tuple(map(float, region.get("embedding", []))),
                )
            )
        sources = tuple(
            EvidenceSource(
                uri=str(source["uri"]),
                page=source.get("page"),
                record_id=source.get("record_id"),
                extractor=source.get("extractor"),
            )
            for source in data.get("sources", [])
        )
        return cls(
            part_id=str(data["part_id"]),
            feature_id=str(data["feature_id"]),
            feature_type=str(data.get("feature_type", "unknown")),
            cad_face_ids=tuple(map(str, data.get("cad_face_ids", []))),
            view_regions=tuple(regions),
            geometric_properties=dict(data.get("geometric_properties", {})),
            sources=sources,
            summary=str(data.get("summary", "")),
            family_id=data.get("family_id"),
            cost=float(data.get("cost", 1.0)),
        )


def _values_equal(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    return left == right


def deduplicate_cross_view(
    units: Iterable[EvidenceUnit], *, numeric_tolerance: float = 1e-6
) -> list[EvidenceUnit]:
    """Merge repeated views of the same ``(part_id, feature_id)``.

    CAD face IDs and ``feature_id`` are the stable correspondence keys.  A
    conflicting property is rejected instead of silently counted twice.
    """

    grouped: dict[tuple[str, str], list[EvidenceUnit]] = {}
    for unit in units:
        grouped.setdefault((unit.part_id, unit.feature_id), []).append(unit)

    merged: list[EvidenceUnit] = []
    for key, group in grouped.items():
        first = group[0]
        feature_types = {unit.feature_type for unit in group}
        family_ids = {unit.family_id for unit in group if unit.family_id is not None}
        if len(feature_types) > 1 or len(family_ids) > 1:
            raise EvidenceConflictError(f"Inconsistent identity metadata for {key}")

        properties: dict[str, Any] = {}
        for unit in group:
            for name, value in unit.geometric_properties.items():
                if name in properties and not _values_equal(
                    properties[name], value, numeric_tolerance
                ):
                    raise EvidenceConflictError(
                        f"Conflicting property {name!r} for {key}: "
                        f"{properties[name]!r} != {value!r}"
                    )
                properties[name] = value

        face_ids = tuple(
            sorted(
                {
                    face
                    for unit in group
                    for face in (
                        *unit.cad_face_ids,
                        *(
                            face_id
                            for region in unit.view_regions
                            for face_id in region.cad_face_ids
                        ),
                    )
                }
            )
        )
        region_map = {
            region.identity_key(): region
            for unit in group
            for region in unit.view_regions
        }
        source_map = {
            (source.uri, source.page, source.record_id, source.extractor): source
            for unit in group
            for source in unit.sources
        }
        summaries = list(dict.fromkeys(unit.summary for unit in group if unit.summary))
        merged.append(
            replace(
                first,
                cad_face_ids=face_ids,
                view_regions=tuple(region_map.values()),
                geometric_properties=properties,
                sources=tuple(source_map.values()),
                summary=" | ".join(summaries),
                family_id=next(iter(family_ids), first.family_id),
                cost=max(unit.cost for unit in group),
            )
        )
    return sorted(merged, key=lambda item: (item.part_id, item.feature_id))


@dataclass(frozen=True)
class AggregateRow:
    part_id: str
    value: Any
    evidence_id: str
    sources: tuple[EvidenceSource, ...] = ()


@dataclass(frozen=True)
class AggregateResult:
    property_name: str
    operation: str
    rows: tuple[AggregateRow, ...]
    expected_part_ids: tuple[str, ...]
    missing_part_ids: tuple[str, ...]
    universe_complete: bool

    @property
    def is_complete(self) -> bool:
        return (
            self.universe_complete
            and not self.missing_part_ids
            and bool(self.expected_part_ids)
        )

    @property
    def winners(self) -> tuple[AggregateRow, ...]:
        if not self.rows:
            return ()
        if self.operation not in {"max", "min"}:
            return ()
        if self.operation == "min":
            target = min(row.value for row in self.rows)
        else:
            target = max(row.value for row in self.rows)
        return tuple(row for row in self.rows if math.isclose(row.value, target))

    def as_evidence_unit(self) -> EvidenceUnit:
        result_cost_items = 1
        if self.operation in {"max", "min"}:
            relevant_rows = self.winners
            result: Any = [
                {
                    "part_id": row.part_id,
                    "value": row.value,
                    "evidence_id": row.evidence_id,
                }
                for row in relevant_rows
            ]
            result_cost_items = len(relevant_rows)
        elif self.operation == "count_distinct":
            groups: dict[str, dict[str, Any]] = {}
            for row in self.rows:
                key = json.dumps(row.value, ensure_ascii=False, sort_keys=True)
                group = groups.setdefault(
                    key,
                    {"value": row.value, "part_ids": [], "evidence_ids": []},
                )
                group["part_ids"].append(row.part_id)
                group["evidence_ids"].append(row.evidence_id)
            result = {
                "distinct_count": len(groups),
                "groups": [
                    {
                        "value": group["value"],
                        "count": len(group["part_ids"]),
                        "example_part_ids": group["part_ids"][:5],
                        "example_evidence_ids": group["evidence_ids"][:5],
                    }
                    for group in groups.values()
                ],
            }
            result_cost_items = len(groups)
            relevant_rows = self.rows
        else:
            relevant_rows = self.rows
            result = [
                {
                    "part_id": row.part_id,
                    "value": row.value,
                    "evidence_id": row.evidence_id,
                }
                for row in relevant_rows
            ]
            result_cost_items = len(relevant_rows)
        properties = {
            "aggregate_property": self.property_name,
            "operation": self.operation,
            "result": result,
            "evaluated_part_count": len(self.rows),
            "expected_part_count": len(self.expected_part_ids),
            "missing_part_ids": list(self.missing_part_ids),
            "complete": self.is_complete,
        }
        return EvidenceUnit(
            part_id="GLOBAL",
            feature_id=f"aggregate_{self.operation}_{self.property_name}",
            feature_type="structured_aggregate",
            geometric_properties=properties,
            sources=(
                EvidenceSource(
                    uri="structured://evidence-store",
                    record_id=f"{self.operation}:{self.property_name}",
                    extractor="EvidenceStore.aggregate",
                ),
                *tuple(
                    dict.fromkeys(
                        source
                        for row in (
                            () if self.operation == "count_distinct" else relevant_rows
                        )
                        for source in row.sources
                    )
                ),
            ),
            summary=(
                f"Exact structured {self.operation} over {len(self.rows)} parts; "
                f"complete={self.is_complete}."
            ),
            cost=max(1.0, result_cost_items / 10),
        )


class EvidenceStore:
    """In-memory feature index with exact structured aggregation support."""

    def __init__(
        self,
        units: Iterable[EvidenceUnit] = (),
        *,
        expected_part_ids: Iterable[str] = (),
        universe_complete: bool | None = None,
    ) -> None:
        self._units: dict[str, EvidenceUnit] = {}
        self._expected_part_ids = set(expected_part_ids)
        self._universe_complete = (
            bool(self._expected_part_ids)
            if universe_complete is None
            else universe_complete
        )
        self.add_many(units)

    def add_many(self, units: Iterable[EvidenceUnit]) -> None:
        combined = list(self._units.values()) + list(units)
        self._units = {
            unit.evidence_id: unit for unit in deduplicate_cross_view(combined)
        }
        self._expected_part_ids.update(unit.part_id for unit in self._units.values())

    @property
    def units(self) -> tuple[EvidenceUnit, ...]:
        return tuple(self._units.values())

    @property
    def part_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._expected_part_ids))

    def for_parts(self, part_ids: Iterable[str]) -> list[EvidenceUnit]:
        wanted = set(part_ids)
        return [unit for unit in self._units.values() if unit.part_id in wanted]

    def for_property(self, property_name: str) -> list[EvidenceUnit]:
        return [
            unit
            for unit in self._units.values()
            if property_name in unit.geometric_properties
        ]

    def aggregate(self, property_name: str, operation: str) -> AggregateResult:
        """Compute an exact global result without semantic Top-K retrieval."""

        if operation not in {
            "max",
            "min",
            "sort_desc",
            "sort_asc",
            "count_distinct",
        }:
            raise ValueError(f"Unsupported aggregate operation: {operation}")
        per_part: dict[str, AggregateRow] = {}
        for unit in self.for_property(property_name):
            value = unit.geometric_properties[property_name]
            if operation == "count_distinct":
                if isinstance(value, (dict, list, tuple, set)):
                    value = json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
                row = AggregateRow(unit.part_id, value, unit.evidence_id, unit.sources)
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                row = AggregateRow(
                    unit.part_id, float(value), unit.evidence_id, unit.sources
                )
            previous = per_part.get(unit.part_id)
            if previous and not _values_equal(previous.value, row.value, 1e-9):
                raise EvidenceConflictError(
                    f"Part {unit.part_id} has conflicting {property_name} values"
                )
            per_part[unit.part_id] = row

        reverse = operation in {"max", "sort_desc"}
        sort_key = (
            (lambda row: (json.dumps(row.value, ensure_ascii=False, sort_keys=True), row.part_id))
            if operation == "count_distinct"
            else (lambda row: (row.value, row.part_id))
        )
        rows = tuple(
            sorted(per_part.values(), key=sort_key, reverse=reverse)
        )
        missing = tuple(sorted(self._expected_part_ids - set(per_part)))
        return AggregateResult(
            property_name=property_name,
            operation=operation,
            rows=rows,
            expected_part_ids=tuple(sorted(self._expected_part_ids)),
            missing_part_ids=missing,
            universe_complete=self._universe_complete,
        )

    def save_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            for unit in self.units:
                handle.write(json.dumps(unit.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "EvidenceStore":
        units = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        units.append(EvidenceUnit.from_dict(json.loads(line)))
                    except Exception as exc:
                        raise ValueError(f"Invalid evidence at line {line_number}") from exc
        # A JSONL evidence index is a materialized corpus, so its listed parts
        # define the complete query universe.
        expected = {unit.part_id for unit in units}
        return cls(units, expected_part_ids=expected, universe_complete=True)
