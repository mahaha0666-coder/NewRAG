"""Conservative native-PDF evidence for dimension topology candidates."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import pymupdf

from .document_schema import DrawingDocument


def extract_dark_fill_marks(
    drawings: list[dict[str, Any]], transform: pymupdf.Matrix,
) -> list[dict[str, Any]]:
    """Return localized dark filled paths without assigning arrow semantics."""

    marks = []
    for path_index, path in enumerate(drawings):
        fill = path.get("fill")
        if fill is None or max(fill) > 0.25:
            continue
        rect = pymupdf.Rect(path["rect"]) * transform
        if rect.is_empty or rect.is_infinite:
            continue
        marks.append({
            "path_index": path_index,
            "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
            "native_type": path.get("type"),
            "fill": [float(value) for value in fill],
        })
    return marks


def _point_box_distance(point: list[float], bbox: list[float]) -> float:
    dx = max(bbox[0] - point[0], 0.0, point[0] - bbox[2])
    dy = max(bbox[1] - point[1], 0.0, point[1] - bbox[3])
    return math.hypot(dx, dy)


def _arrow_candidates(
    point: list[float], orientation: str, line_length: float, marks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tolerance = max(2.5, 0.015 * line_length)
    maximum_extent = max(16.0, 0.15 * line_length)
    candidates = []
    for mark in marks:
        x0, y0, x1, y1 = mark["bbox"]
        width, height = x1 - x0, y1 - y0
        if min(width, height) <= 0.15 or max(width, height) > maximum_extent:
            continue
        if orientation == "horizontal" and width < 1.5 * height:
            continue
        if orientation == "vertical" and height < 1.5 * width:
            continue
        distance = _point_box_distance(point, mark["bbox"])
        if distance > tolerance:
            continue
        candidates.append({
            **mark,
            "distance_to_endpoint_pixels": round(distance, 4),
            "orientation_consistent": True,
        })
    return sorted(candidates, key=lambda row: (
        row["distance_to_endpoint_pixels"],
        (row["bbox"][2] - row["bbox"][0]) * (row["bbox"][3] - row["bbox"][1]),
        row["path_index"],
    ))


def classify_measurement_text(raw_text: str) -> dict[str, Any]:
    """Classify only lexical eligibility; do not infer a dimension from text alone."""

    normalized = "".join(raw_text.strip().split())
    has_digit = any(character.isdigit() for character in normalized)
    letter_tokens = [character for character in normalized if character.isalpha()]
    allowed_letters = {"M", "R", "m", "r", "Ø", "Φ", "φ", "⌀"}
    eligible = bool(normalized and has_digit and set(letter_tokens) <= allowed_letters)
    return {
        "role": "numeric_measurement_candidate" if eligible else "non_dimension_text",
        "normalized_text": normalized,
        "lexically_eligible": eligible,
        "role_verified": False,
    }


def refine_dimension_chain(
    chain: dict[str, Any], marks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require close text and orientation-consistent fill marks at both endpoints."""

    text_role = classify_measurement_text(str(chain.get("measurement_raw_text") or ""))
    bbox = chain["measurement_bbox"]
    text_height = max(1.0, bbox[3] - bbox[1])
    normalized_text_distance = chain["text_distance_pixels"] / text_height
    dimension = chain["dimension_line"]
    endpoint_marks = [
        _arrow_candidates(point, dimension["orientation"], dimension["length"], marks)
        for point in (dimension["first"], dimension["second"])
    ]
    reasons = []
    if not text_role["lexically_eligible"]:
        reasons.append("text_role_not_numeric_measurement_candidate")
    if normalized_text_distance > 0.5:
        reasons.append("text_not_tightly_aligned_to_dimension_line")
    if not endpoint_marks[0]:
        reasons.append("first_endpoint_missing_directional_fill_mark")
    if not endpoint_marks[1]:
        reasons.append("second_endpoint_missing_directional_fill_mark")
    if endpoint_marks[0] and endpoint_marks[1] and (
        endpoint_marks[0][0]["path_index"] == endpoint_marks[1][0]["path_index"]
    ):
        reasons.append("same_fill_path_used_for_both_endpoints")

    retained = not reasons
    arrows = [rows[0] for rows in endpoint_marks] if retained else []
    identity = [
        chain["id"], chain["measurement_id"],
        *[row["path_index"] for row in arrows],
    ]
    return {
        "id": "dimension_entity_" + hashlib.sha256(repr(identity).encode()).hexdigest()[:20],
        "topology_chain_id": chain["id"],
        "measurement_id": chain["measurement_id"],
        "measurement_raw_text": chain["measurement_raw_text"],
        "page": chain["page"],
        "view_id": chain["view_id"],
        "text_role": text_role,
        "normalized_text_distance": round(normalized_text_distance, 6),
        "endpoint_fill_marks": arrows,
        "retained": retained,
        "rejection_reasons": reasons,
        "decision_status": "unresolved" if retained else "rejected",
        "feature_ownership_verified": False,
        "method": "native_dimension_entity_evidence_r02d_v2",
        "needs_review": retained,
    }


def build_dimension_entity_evidence(
    document: DrawingDocument, topology: dict[str, Any],
) -> dict[str, Any]:
    source = Path(document.source_path)
    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if source_hash != document.source_sha256 or topology["source_sha256"] != source_hash:
        raise ValueError("Source PDF hash mismatch")

    page_by_number = {page.page_number: page for page in document.pages}
    marks_by_page: dict[int, list[dict[str, Any]]] = {}
    with pymupdf.open(source) as pdf:
        for page_number in {row["page"] for row in topology["chains"]}:
            page = page_by_number[page_number]
            transform = pymupdf.Matrix(*page.pixel_to_pdf_matrix)
            if transform.invert():
                raise ValueError("Singular page transform")
            marks_by_page[page_number] = extract_dark_fill_marks(
                pdf[page_number - 1].get_drawings(), transform,
            )

    candidates = [
        refine_dimension_chain(chain, marks_by_page.get(chain["page"], []))
        for chain in topology["chains"]
    ]
    retained_by_topology: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        if candidate["retained"]:
            retained_by_topology.setdefault(candidate["topology_chain_id"], []).append(candidate)
    entity_groups = []
    for topology_chain_id, rows in sorted(retained_by_topology.items()):
        entity_groups.append({
            "id": "dimension_entity_group_" + hashlib.sha256(
                topology_chain_id.encode()
            ).hexdigest()[:20],
            "topology_chain_id": topology_chain_id,
            "candidate_ids": sorted(row["id"] for row in rows),
            "measurement_candidates": [
                {
                    "measurement_id": row["measurement_id"],
                    "raw_text": row["measurement_raw_text"],
                    "normalized_text_distance": row["normalized_text_distance"],
                }
                for row in sorted(rows, key=lambda row: row["measurement_id"])
            ],
            "association_status": "ambiguous" if len(rows) > 1 else "unresolved",
            "feature_ownership_verified": False,
            "needs_review": True,
        })
    return {
        "schema_version": "native_dimension_entity_evidence_r02d.v2",
        "source_sha256": source_hash,
        "route_label_access": False,
        "candidates": candidates,
        "entity_groups": entity_groups,
        "summary": {
            "topology_chain_count": len(candidates),
            "retained_candidate_count": sum(row["retained"] for row in candidates),
            "rejected_candidate_count": sum(not row["retained"] for row in candidates),
            "retained_entity_group_count": len(entity_groups),
            "ambiguous_entity_group_count": sum(
                row["association_status"] == "ambiguous" for row in entity_groups
            ),
            "supported_feature_ownership_count": 0,
        },
        "limitations": [
            "Dark filled endpoint marks are arrow evidence candidates, not verified dimension entities",
            "The text role is lexical eligibility and is not a semantic dimension classifier",
            "Retained candidates remain unresolved and have no controlled-feature ownership",
            "Multiple OCR measurements may compete for one native dimension entity group",
            "No route labels or human relation gold were used",
        ],
    }


__all__ = [
    "build_dimension_entity_evidence",
    "classify_measurement_text",
    "extract_dark_fill_marks",
    "refine_dimension_chain",
]
