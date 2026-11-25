"""Detect review-only native-PDF dimension-line topology candidates."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import pymupdf

from .document_schema import DrawingDocument
from .local_geometry import extract_strokes
from .manufacturing_ir import ManufacturingIR


def _line_record(stroke: dict[str, Any]) -> dict[str, Any] | None:
    points = stroke.get("points") or []
    if len(points) != 2:
        return None
    first, second = points
    dx, dy = second[0] - first[0], second[1] - first[1]
    length = math.hypot(dx, dy)
    if length < 8:
        return None
    tolerance = max(1.5, 0.02 * length)
    if abs(dy) <= tolerance:
        orientation = "horizontal"
    elif abs(dx) <= tolerance:
        orientation = "vertical"
    else:
        return None
    return {
        "stroke_id": stroke["id"],
        "orientation": orientation,
        "first": [float(first[0]), float(first[1])],
        "second": [float(second[0]), float(second[1])],
        "length": length,
        "sources": stroke.get("sources") or [],
    }


def _range(line: dict[str, Any], axis: int) -> tuple[float, float]:
    return tuple(sorted((line["first"][axis], line["second"][axis])))


def _axis_distance(value: float, interval: tuple[float, float]) -> float:
    return max(interval[0] - value, 0.0, value - interval[1])


def find_dimension_topology_candidates(
    measurement_bbox: tuple[float, float, float, float],
    strokes: list[dict[str, Any]],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Find a text-adjacent dimension line with support lines at both ends."""

    lines = [record for stroke in strokes if (record := _line_record(stroke)) is not None]
    x0, y0, x1, y1 = measurement_bbox
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    text_height = y1 - y0
    text_width = x1 - x0
    candidates = []
    for dimension in lines:
        if dimension["orientation"] == "horizontal":
            along = _range(dimension, 0)
            perpendicular = dimension["first"][1]
            text_distance = min(abs(perpendicular - y0), abs(perpendicular - y1))
            if (_axis_distance(center_x, along) > max(12.0, text_width)
                    or text_distance > max(30.0, 3.0 * text_height)
                    or dimension["length"] < max(24.0, 2.0 * text_width)):
                continue
            endpoints = [(along[0], perpendicular), (along[1], perpendicular)]
            support_orientation = "vertical"
            support_axis = 1
        else:
            along = _range(dimension, 1)
            perpendicular = dimension["first"][0]
            text_distance = min(abs(perpendicular - x0), abs(perpendicular - x1))
            if (_axis_distance(center_y, along) > max(12.0, text_height)
                    or text_distance > max(30.0, 2.0 * text_width)
                    or dimension["length"] < max(24.0, 2.0 * text_height)):
                continue
            endpoints = [(perpendicular, along[0]), (perpendicular, along[1])]
            support_orientation = "horizontal"
            support_axis = 0

        endpoint_supports = []
        support_score = 0.0
        for endpoint in endpoints:
            options = []
            for support in lines:
                if support["stroke_id"] == dimension["stroke_id"]:
                    continue
                if support["orientation"] != support_orientation:
                    continue
                support_interval = _range(support, support_axis)
                fixed_axis = 0 if support_orientation == "vertical" else 1
                fixed_value = support["first"][fixed_axis]
                fixed_distance = abs(fixed_value - endpoint[fixed_axis])
                reach_distance = _axis_distance(endpoint[support_axis], support_interval)
                tolerance = max(8.0, 0.75 * text_height)
                if (fixed_distance <= tolerance and reach_distance <= tolerance
                        and support["length"] >= max(16.0, 1.5 * text_height)):
                    options.append((fixed_distance + reach_distance, support))
            if not options:
                endpoint_supports = []
                break
            score, support = min(options, key=lambda row: (row[0], row[1]["stroke_id"]))
            endpoint_supports.append(support)
            support_score += score
        if len(endpoint_supports) != 2:
            continue
        source_keys = {
            (source["path_index"], source["item_index"], source["native_kind"])
            for line in [dimension, *endpoint_supports]
            for source in line["sources"]
        }
        if len(source_keys) < 3:
            continue
        identity = [dimension["stroke_id"], *sorted(line["stroke_id"] for line in endpoint_supports)]
        candidates.append({
            "id": "dimension_topology_" + hashlib.sha256(repr(identity).encode()).hexdigest()[:20],
            "dimension_line": {
                key: dimension[key] for key in ("stroke_id", "orientation", "first", "second", "length", "sources")
            },
            "extension_lines": [
                {key: support[key] for key in ("stroke_id", "orientation", "first", "second", "length", "sources")}
                for support in endpoint_supports
            ],
            "text_distance_pixels": round(text_distance, 4),
            "support_distance_sum_pixels": round(support_score, 4),
            "source_refs": [
                {"path_index": path, "item_index": item, "native_kind": kind}
                for path, item, kind in sorted(source_keys)
            ],
            "decision_status": "unresolved",
            "feature_ownership_verified": False,
            "needs_review": True,
        })
    candidates.sort(key=lambda row: (
        row["text_distance_pixels"] + row["support_distance_sum_pixels"], row["id"]
    ))
    deduplicated = []
    seen = set()
    for candidate in candidates:
        membership = tuple(
            (row["path_index"], row["item_index"], row["native_kind"])
            for row in candidate["source_refs"]
        )
        if membership in seen:
            continue
        seen.add(membership)
        deduplicated.append(candidate)
        if len(deduplicated) == limit:
            break
    return deduplicated


def build_dimension_topology(document: DrawingDocument, ir: ManufacturingIR) -> dict[str, Any]:
    source = Path(document.source_path)
    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if source_hash != document.source_sha256:
        raise ValueError("Source PDF hash mismatch")
    page_by_number = {page.page_number: page for page in document.pages}
    view_by_page: dict[int, list[Any]] = {}
    for view in ir.views:
        if view.page is not None and view.bbox is not None:
            view_by_page.setdefault(view.page, []).append(view)
    view_strokes: dict[str, list[dict[str, Any]]] = {}
    with pymupdf.open(source) as pdf:
        for page_number, views in view_by_page.items():
            page = page_by_number[page_number]
            transform = pymupdf.Matrix(*page.pixel_to_pdf_matrix)
            if transform.invert():
                raise ValueError("Singular page transform")
            paths = pdf[page_number - 1].get_drawings()
            for view in views:
                frame = [int(math.floor(view.bbox[0])), int(math.floor(view.bbox[1])),
                         int(math.ceil(view.bbox[2])), int(math.ceil(view.bbox[3]))]
                view_strokes[view.id] = extract_strokes(paths, transform, frame, set())[0]

    chains = []
    measurements_in_views = 0
    for measurement in ir.measurements:
        if measurement.page is None or measurement.bbox is None:
            continue
        center = ((measurement.bbox[0] + measurement.bbox[2]) / 2,
                  (measurement.bbox[1] + measurement.bbox[3]) / 2)
        containing = [
            view for view in view_by_page.get(measurement.page, [])
            if (view.bbox[0] <= center[0] <= view.bbox[2]
                and view.bbox[1] <= center[1] <= view.bbox[3])
        ]
        if not containing:
            continue
        measurements_in_views += 1
        view = min(containing, key=lambda row: (
            (row.bbox[2] - row.bbox[0]) * (row.bbox[3] - row.bbox[1])
        ))
        for rank, candidate in enumerate(find_dimension_topology_candidates(
            measurement.bbox, view_strokes.get(view.id, []), limit=3
        ), start=1):
            chains.append({
                **candidate,
                "measurement_id": measurement.id,
                "measurement_raw_text": measurement.raw_text,
                "measurement_bbox": list(measurement.bbox),
                "page": measurement.page,
                "view_id": view.id,
                "rank": rank,
                "method": "native_dimension_topology_r02c_v1",
            })
    return {
        "schema_version": "native_dimension_topology_r02c.v1",
        "source_sha256": document.source_sha256,
        "route_label_access": False,
        "chains": chains,
        "summary": {
            "measurement_count": len(ir.measurements),
            "measurements_in_views": measurements_in_views,
            "measurements_with_chains": len({row["measurement_id"] for row in chains}),
            "chain_count": len(chains),
            "supported_chain_count": 0,
        },
        "limitations": [
            "The topology identifies dimension-line candidates, not controlled features",
            "Arrowheads and CAD dimension entities are not required in this version",
            "No route labels or human relation gold were used",
        ],
    }


__all__ = ["build_dimension_topology", "find_dimension_topology_candidates"]
