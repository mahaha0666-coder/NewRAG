"""Build review-only native-PDF geometry candidates inside detected views."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import pymupdf

from .document_schema import DrawingDocument
from .local_geometry import extract_strokes, point_distance, stroke_groups
from .manufacturing_ir import ManufacturingIR


def _inside(bbox: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _polyline_length(points: list[list[float]]) -> float:
    return sum(math.dist(start, end) for start, end in zip(points, points[1:]))


def positive_page_bbox(
    bbox: list[float], page_width: int, page_height: int,
) -> tuple[list[float], list[float]]:
    """Normalize zero-width line extents without hiding their raw geometry."""

    raw = [float(value) for value in bbox]
    x0, y0, x1, y1 = raw
    x0, x1 = max(0.0, min(x0, page_width)), max(0.0, min(x1, page_width))
    y0, y1 = max(0.0, min(y0, page_height)), max(0.0, min(y1, page_height))
    if x1 <= x0:
        x0, x1 = max(0.0, x0 - 0.5), min(float(page_width), x1 + 0.5)
    if y1 <= y0:
        y0, y1 = max(0.0, y0 - 0.5), min(float(page_height), y1 + 0.5)
    if x1 <= x0:
        x0, x1 = max(0.0, x0 - 1.0), min(float(page_width), x0 + 1.0)
    if y1 <= y0:
        y0, y1 = max(0.0, y0 - 1.0), min(float(page_height), y0 + 1.0)
    if not (0 <= x0 < x1 <= page_width and 0 <= y0 < y1 <= page_height):
        raise ValueError("Native geometry bbox cannot be represented inside the page")
    return [x0, y0, x1, y1], raw


def _group_candidate(view_id: str, group: list[dict[str, Any]]) -> dict[str, Any]:
    points = [point for stroke in group for point in stroke["points"]]
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    source_refs = [source for stroke in group for source in stroke["sources"]]
    identity = sorted(
        (source["path_index"], source["item_index"], source["native_kind"])
        for source in source_refs
    )
    digest = hashlib.sha256(repr([view_id, identity]).encode()).hexdigest()[:20]
    return {
        "id": f"native_geometry_{digest}",
        "kind": "native_stroke_group_candidate",
        "view_id": view_id,
        "bbox": [min(x_values), min(y_values), max(x_values), max(y_values)],
        "stroke_count": len(group),
        "total_stroke_length_pixels": round(
            sum(_polyline_length(stroke["points"]) for stroke in group), 4
        ),
        "clipped_by_view": any(stroke["clipped_by_window"] for stroke in group),
        "source_refs": source_refs,
        "physical_object_verified": False,
        "decision_status": "unresolved",
        "needs_review": True,
        "_strokes": group,
    }


def select_measurement_geometry_candidates(
    measurement_bbox: tuple[float, float, float, float],
    candidates: list[dict[str, Any]],
    view_bbox: tuple[float, float, float, float],
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Select a bounded local set without claiming dimension ownership."""

    center = (
        (measurement_bbox[0] + measurement_bbox[2]) / 2,
        (measurement_bbox[1] + measurement_bbox[3]) / 2,
    )
    text_height = measurement_bbox[3] - measurement_bbox[1]
    threshold = max(
        48.0,
        6.0 * text_height,
        0.08 * min(view_bbox[2] - view_bbox[0], view_bbox[3] - view_bbox[1]),
    )
    ranked = []
    for candidate in candidates:
        distance = min(
            point_distance(center, stroke["points"])
            for stroke in candidate.get("_strokes") or []
        )
        if distance <= threshold:
            ranked.append({
                "geometry_candidate_id": candidate["id"],
                "distance_pixels": round(distance, 4),
                "threshold_pixels": round(threshold, 4),
            })
    return sorted(ranked, key=lambda row: (
        row["distance_pixels"], row["geometry_candidate_id"]
    ))[:limit]


def build_native_view_geometry(
    document: DrawingDocument,
    ir: ManufacturingIR,
) -> dict[str, Any]:
    """Extract native stroke groups and unresolved dimension adjacency edges."""

    source = Path(document.source_path)
    if not source.is_file() or source.suffix.casefold() != ".pdf":
        raise ValueError("Native view geometry requires the frozen source PDF")
    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if source_hash != document.source_sha256:
        raise ValueError("Source PDF hash mismatch")
    page_by_number = {page.page_number: page for page in document.pages}
    view_rows = []
    nodes = []
    internal_by_view: dict[str, list[dict[str, Any]]] = {}
    with pymupdf.open(source) as pdf:
        paths_by_page = {
            page.page_number: pdf[page.page_number - 1].get_drawings()
            for page in document.pages
        }
        for view in ir.views:
            if view.page is None or view.bbox is None:
                continue
            page = page_by_number[view.page]
            frame = [
                max(0, int(math.floor(view.bbox[0]))),
                max(0, int(math.floor(view.bbox[1]))),
                min(page.width, int(math.ceil(view.bbox[2]))),
                min(page.height, int(math.ceil(view.bbox[3]))),
            ]
            transform = pymupdf.Matrix(*page.pixel_to_pdf_matrix)
            if transform.invert():
                raise ValueError("Singular page transform")
            strokes, stats = extract_strokes(
                paths_by_page[view.page], transform, frame, set()
            )
            diagonal = math.hypot(frame[2] - frame[0], frame[3] - frame[1])
            candidates = []
            for group in stroke_groups(strokes, tolerance=1.5):
                if len(group) < 2:
                    continue
                candidate = _group_candidate(view.id, group)
                width = candidate["bbox"][2] - candidate["bbox"][0]
                height = candidate["bbox"][3] - candidate["bbox"][1]
                if max(width, height) < 4 or candidate["total_stroke_length_pixels"] < max(20, 0.02 * diagonal):
                    continue
                normalized_bbox, raw_bbox = positive_page_bbox(
                    candidate["bbox"], page.width, page.height
                )
                candidate.update(
                    bbox=normalized_bbox,
                    raw_stroke_bbox=raw_bbox,
                    page=view.page,
                    view_bbox=list(view.bbox),
                )
                candidates.append(candidate)
            candidates.sort(key=lambda row: (
                row["clipped_by_view"],
                -row["stroke_count"],
                -row["total_stroke_length_pixels"],
                row["id"],
            ))
            candidates = candidates[:80]
            internal_by_view[view.id] = candidates
            nodes.extend(candidates)
            view_rows.append({
                "view_id": view.id,
                "page": view.page,
                "bbox": list(view.bbox),
                "stroke_count": len(strokes),
                "raw_group_count": len(stroke_groups(strokes, tolerance=1.5)),
                "candidate_group_count": len(candidates),
                "duplicates_merged": stats["duplicates_merged"],
                "unsupported_items": stats["unsupported_items"],
            })

    # Assign each text to the smallest containing view so overlapping layout
    # boxes do not inflate the relation count.
    view_by_id = {view.id: view for view in ir.views if view.bbox is not None}
    edges = []
    measurements_in_views = 0
    for measurement in ir.measurements:
        if measurement.page is None or measurement.bbox is None:
            continue
        center = (
            (measurement.bbox[0] + measurement.bbox[2]) / 2,
            (measurement.bbox[1] + measurement.bbox[3]) / 2,
        )
        containing = [
            view for view in view_by_id.values()
            if view.page == measurement.page and view.bbox is not None
            and _inside(view.bbox, center)
        ]
        if not containing:
            continue
        measurements_in_views += 1
        view = min(
            containing,
            key=lambda row: (row.bbox[2] - row.bbox[0]) * (row.bbox[3] - row.bbox[1]),
        )
        selected = select_measurement_geometry_candidates(
            measurement.bbox,
            internal_by_view.get(view.id, []),
            view.bbox,
        )
        for candidate in selected:
            edge_id = hashlib.sha256(
                f"{measurement.id}\x1f{candidate['geometry_candidate_id']}".encode()
            ).hexdigest()[:20]
            edges.append({
                "id": f"dimension_geometry_candidate_{edge_id}",
                "type": "dimension_geometry_candidate",
                "source_id": measurement.id,
                "target_id": candidate["geometry_candidate_id"],
                "view_id": view.id,
                "page": measurement.page,
                "source_bbox": list(measurement.bbox),
                **candidate,
                "decision_status": "unresolved",
                "method": "native_pdf_view_geometry_r02b_v2",
                "needs_review": True,
            })

    for node in nodes:
        node.pop("_strokes", None)
    grounded_measurements = len({edge["source_id"] for edge in edges})
    return {
        "schema_version": "native_view_geometry_r02b.v2",
        "source_sha256": document.source_sha256,
        "route_label_access": False,
        "views": view_rows,
        "geometry_candidates": nodes,
        "measurement_candidate_edges": edges,
        "summary": {
            "view_count": len(view_rows),
            "views_with_geometry_candidates": sum(
                row["candidate_group_count"] > 0 for row in view_rows
            ),
            "measurement_count": len(ir.measurements),
            "measurements_in_views": measurements_in_views,
            "grounded_measurements": grounded_measurements,
            "candidate_edge_count": len(edges),
            "supported_edge_count": 0,
        },
        "limitations": [
            "Stroke groups are not physical parts or manufacturing features",
            "Distance-bounded adjacency does not prove dimension ownership",
            "Text-outline strokes can remain among native PDF paths",
            "No route labels or human relationship gold were used",
        ],
    }


__all__ = [
    "build_native_view_geometry",
    "positive_page_bbox",
    "select_measurement_geometry_candidates",
]
