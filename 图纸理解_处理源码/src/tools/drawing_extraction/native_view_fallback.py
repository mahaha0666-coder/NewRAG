"""Source-matched, review-only native PDF regions when layout finds no views."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pymupdf

from .document_schema import DrawingDocument
from .local_geometry import sample_item
from .source_page_render import apply_document_page_render


MASK_SCALE = 0.8  # mask pixels per PDF point
MIN_ITEM_EXTENT_PT = 8.0
JOIN_RADIUS_PT = 12.0
MIN_SUPPORT_ITEMS = 5
MIN_DISTINCT_PATHS = 4
MIN_COMPACT_SUPPORT_PATHS = 20


def _source_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _path_items(
    page: pymupdf.Page,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {"native_paths": 0, "frame_paths_rejected": 0,
              "short_items_rejected": 0, "stroke_items_retained": 0}
    width, height = page.rect.width, page.rect.height
    identity = pymupdf.Matrix(1, 0, 0, 1, 0, 0)
    for path_index, path in enumerate(page.get_drawings()):
        counts["native_paths"] += 1
        color = path.get("color")
        if path.get("type") not in {"s", "fs"} or color is None or max(color) > 0.5:
            continue
        rect = path["rect"]
        if rect.width >= 0.8 * width or rect.height >= 0.8 * height:
            counts["frame_paths_rejected"] += 1
            continue
        for item_index, item in enumerate(path["items"]):
            points = sample_item(item, identity)
            if not points:
                continue
            coords = np.asarray(points, dtype=np.float32)
            low, high = coords.min(axis=0), coords.max(axis=0)
            length = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
            if max(high - low) < MIN_ITEM_EXTENT_PT or length < MIN_ITEM_EXTENT_PT:
                counts["short_items_rejected"] += 1
                continue
            rows.append({
                "points": coords,
                "bbox_pdf": [float(low[0]), float(low[1]),
                             float(high[0]), float(high[1])],
                "length_pdf": length,
                "ref": {"path_index": path_index, "item_index": item_index,
                        "native_kind": item[0]},
            })
            counts["stroke_items_retained"] += 1
    return rows, counts


def _stroke_mask(
    rows: list[dict[str, Any]], page_size: tuple[float, float],
) -> np.ndarray:
    mask = np.zeros((math.ceil(page_size[1] * MASK_SCALE),
                     math.ceil(page_size[0] * MASK_SCALE)), dtype=np.uint8)
    for row in rows:
        points = np.rint(row["points"] * MASK_SCALE).astype(np.int32)
        cv2.polylines(mask, [points], False, 255, 1, lineType=cv2.LINE_8)
    return mask


def _review_clusters(
    rows: list[dict[str, Any]], mask: np.ndarray,
    page_size: tuple[float, float],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    radius = max(1, round(JOIN_RADIUS_PT * MASK_SCALE))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    joined = cv2.dilate(mask, kernel)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    page_area = page_size[0] * page_size[1]
    diagnostics = {"components": component_count - 1, "small": 0,
                   "weak_support": 0, "bottom_table_or_title": 0}
    clusters: list[dict[str, Any]] = []
    for component in range(1, component_count):
        x, y, width, height, _ = (int(value) for value in stats[component])
        box = [x / MASK_SCALE, y / MASK_SCALE,
               (x + width) / MASK_SCALE, (y + height) / MASK_SCALE]
        area = (box[2] - box[0]) * (box[3] - box[1])
        if (min(box[2] - box[0], box[3] - box[1]) < 18
                or area < 0.0004 * page_area):
            diagnostics["small"] += 1
            continue
        if box[1] >= 0.70 * page_size[1]:
            diagnostics["bottom_table_or_title"] += 1
            continue
        support = []
        for row in rows:
            points = row["bbox_pdf"]
            center = ((points[0] + points[2]) / 2,
                      (points[1] + points[3]) / 2)
            px = min(labels.shape[1] - 1, max(0, round(center[0] * MASK_SCALE)))
            py = min(labels.shape[0] - 1, max(0, round(center[1] * MASK_SCALE)))
            if labels[py, px] == component:
                support.append(row)
        distinct_paths = len({row["ref"]["path_index"] for row in support})
        if len(support) < MIN_SUPPORT_ITEMS or distinct_paths < MIN_DISTINCT_PATHS:
            diagnostics["weak_support"] += 1
            continue
        raw = [min(row["bbox_pdf"][0] for row in support),
               min(row["bbox_pdf"][1] for row in support),
               max(row["bbox_pdf"][2] for row in support),
               max(row["bbox_pdf"][3] for row in support)]
        bbox_pdf = [max(0.0, raw[0] - 3), max(0.0, raw[1] - 3),
                    min(page_size[0], raw[2] + 3),
                    min(page_size[1], raw[3] + 3)]
        if bbox_pdf[0] >= bbox_pdf[2] or bbox_pdf[1] >= bbox_pdf[3]:
            diagnostics["small"] += 1
            continue
        refs = [row["ref"] for row in sorted(
            support, key=lambda item: -item["length_pdf"])]
        clusters.append({
            "bbox_pdf": bbox_pdf,
            "native_path_refs": refs,
            "support_path_count": distinct_paths,
            "support_item_count": len(support),
            "support_length_pdf_points": round(sum(
                row["length_pdf"] for row in support), 2),
        })
    clusters.sort(key=lambda row: (row["bbox_pdf"][1], row["bbox_pdf"][0]))
    return clusters, diagnostics


def _compact_table_top(
    rows: list[dict[str, Any]], content_box: list[float],
) -> float | None:
    width = content_box[2] - content_box[0]
    height = content_box[3] - content_box[1]
    rulings: dict[tuple[int, int], set[float]] = {}
    for row in rows:
        x0, y0, x1, y1 = row["bbox_pdf"]
        if (x1 - x0 < 0.5 * width or y1 - y0 > 1.0
                or y0 < content_box[1] + 0.5 * height):
            continue
        rulings.setdefault((round(x0), round(x1)), set()).add(round(y0, 1))
    tops = [min(positions) for positions in rulings.values()
            if len(positions) >= 3]
    return min(tops) if tops else None


def _source_clusters(
    page: pymupdf.Page, rows: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    content_box = [min(row["bbox_pdf"][0] for row in rows),
                   min(row["bbox_pdf"][1] for row in rows),
                   max(row["bbox_pdf"][2] for row in rows),
                   max(row["bbox_pdf"][3] for row in rows)]
    sheet_width = content_box[2] - content_box[0]
    sheet_height = content_box[3] - content_box[1]
    compact = (
        page.rotation == 0 and list(page.mediabox) == list(page.cropbox)
        and sheet_width <= 0.4 * page_size[0]
        and sheet_height <= 0.4 * page_size[1]
        and content_box[1] >= 0.6 * page_size[1]
    )
    if compact:
        table_top = _compact_table_top(rows, content_box)
        if table_top is None:
            return [], {"compact_sheet": True, "separation": "no_table_top"}, (
                "native_pdf_compact_sheet_strokes"
            )
        paths = page.get_drawings()
        frame_paths = {row["ref"]["path_index"] for row in rows
                       if (paths[row["ref"]["path_index"]]["rect"].width
                           >= 0.8 * sheet_width
                           or paths[row["ref"]["path_index"]]["rect"].height
                           >= 0.8 * sheet_height)}
        clipped = [row for row in rows
                   if row["ref"]["path_index"] not in frame_paths
                   and row["bbox_pdf"][3] < table_top - 1]
        if not clipped:
            return [], {"compact_sheet": True, "separation": "no_upper_strokes",
                        "table_top_pdf": round(table_top, 3)}, (
                "native_pdf_compact_sheet_strokes"
            )
        offset = (content_box[0] - 2, content_box[1] - 2)
        shifted = [{**row,
                    "points": row["points"] - offset,
                    "bbox_pdf": [row["bbox_pdf"][0] - offset[0],
                                 row["bbox_pdf"][1] - offset[1],
                                 row["bbox_pdf"][2] - offset[0],
                                 row["bbox_pdf"][3] - offset[1]]}
                   for row in clipped]
        local_size = (sheet_width + 4, sheet_height + 4)
        clusters, diagnostics = _review_clusters(
            shifted, _stroke_mask(shifted, local_size), local_size,
        )
        retained = []
        for cluster in clusters:
            if cluster["support_path_count"] < MIN_COMPACT_SUPPORT_PATHS:
                continue
            box = cluster["bbox_pdf"]
            cluster["bbox_pdf"] = [box[0] + offset[0], box[1] + offset[1],
                                   box[2] + offset[0], box[3] + offset[1]]
            if cluster["bbox_pdf"][3] < table_top:
                retained.append(cluster)
        return retained, {**diagnostics, "compact_sheet": True,
                          "table_top_pdf": round(table_top, 3),
                          "frame_paths_rejected": len(frame_paths)}, (
            "native_pdf_compact_sheet_strokes"
        )
    clusters, diagnostics = _review_clusters(
        rows, _stroke_mask(rows, page_size), page_size,
    )
    return clusters, {**diagnostics, "compact_sheet": False}, (
        "native_pdf_long_stroke_cluster"
    )


def _focus_box(box: list[float], image_size: tuple[int, int]) -> list[float]:
    width, height = image_size
    padding_x = max(120, min(width * 0.05, (box[2] - box[0]) * 2))
    padding_y = max(90, min(height * 0.05, (box[3] - box[1]) * 2))
    return [max(0.0, box[0] - padding_x), max(0.0, box[1] - padding_y),
            min(width, box[2] + padding_x), min(height, box[3] + padding_y)]


def _validated_source_page(
    document: DrawingDocument, image_size: tuple[int, int], page_number: int,
) -> tuple[Path, str, Any]:
    source = Path(document.source_path)
    if source.suffix.casefold() != ".pdf" or not source.is_file():
        raise ValueError("Native view fallback requires the frozen source PDF")
    source_hash = _source_sha256(source)
    if source_hash != document.source_sha256:
        raise ValueError("Source PDF hash mismatch")
    page_info = next((page for page in document.pages
                      if page.page_number == page_number), None)
    if page_info is None or image_size != (page_info.width, page_info.height):
        raise ValueError("Rendered page dimensions do not match DrawingDocument")
    return source, source_hash, page_info


def native_view_layout_support(
    document: DrawingDocument,
    image_size: tuple[int, int],
    boxes: Sequence[Sequence[float]],
    *,
    page_number: int = 1,
) -> list[dict[str, Any]]:
    """Measure native PDF stroke support for supplied rendered-page layout boxes."""

    source, _, page_info = _validated_source_page(document, image_size, page_number)
    checked_boxes: list[tuple[float, float, float, float]] = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError("Layout box must contain four coordinates")
        x0, y0, x1, y1 = (float(value) for value in box)
        if (not all(math.isfinite(value) for value in (x0, y0, x1, y1))
                or not (0 <= x0 < x1 <= image_size[0]
                        and 0 <= y0 < y1 <= image_size[1])):
            raise ValueError("Layout box is outside the rendered page")
        checked_boxes.append((x0, y0, x1, y1))
    with pymupdf.open(source) as pdf:
        if page_number < 1 or page_number > len(pdf):
            raise ValueError("Source PDF has no matching page")
        page = pdf[page_number - 1]
        apply_document_page_render(page, page_info)
        page_size = (float(page.rect.width), float(page.rect.height))
        rows, _ = _path_items(page)
        clusters = []
        if rows and checked_boxes:
            clusters, _, _ = _source_clusters(page, rows, page_size)
    pdf_to_pixels = pymupdf.Matrix(*page_info.pixel_to_pdf_matrix)
    if pdf_to_pixels.invert():
        raise ValueError("Singular rendered page transform")
    cluster_boxes = [pymupdf.Rect(*cluster["bbox_pdf"]) * pdf_to_pixels
                     for cluster in clusters]
    projected_rows = []
    for row in rows:
        x0, y0, x1, y1 = row["bbox_pdf"]
        center = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2) * pdf_to_pixels
        projected_rows.append((center.x, center.y, row))

    support = []
    for x0, y0, x1, y1 in checked_boxes:
        area = (x1 - x0) * (y1 - y0)
        coverage = max((
            max(0.0, min(x1, cluster.x1) - max(x0, cluster.x0))
            * max(0.0, min(y1, cluster.y1) - max(y0, cluster.y0)) / area
            for cluster in cluster_boxes
        ), default=0.0)
        inside = [row for x, y, row in projected_rows
                  if x0 <= x <= x1 and y0 <= y <= y1]
        support.append({
            "cluster_coverage": round(coverage, 6),
            "support_path_count": len({row["ref"]["path_index"] for row in inside}),
            "support_length_pdf_points": round(sum(
                row["length_pdf"] for row in inside), 2),
            "native_path_refs": [dict(row["ref"]) for row in inside],
        })
    return support


def native_view_review_evidence(
    document: DrawingDocument,
    image_size: tuple[int, int],
    *,
    existing_views: Sequence[object] = (),
    page_number: int = 1,
) -> dict[str, Any]:
    """Propose source-linked regions; never assert view semantics or ownership."""

    source, source_hash, page_info = _validated_source_page(
        document, image_size, page_number,
    )
    if existing_views:
        return {"status": "not_eligible_existing_views", "source_sha256": source_hash,
                "candidates": [], "diagnostics": {"existing_views": len(existing_views)}}
    with pymupdf.open(source) as pdf:
        if page_number < 1 or page_number > len(pdf):
            raise ValueError("Source PDF has no matching page")
        page = pdf[page_number - 1]
        apply_document_page_render(page, page_info)
        page_size = (float(page.rect.width), float(page.rect.height))
        rows, diagnostics = _path_items(page)
        clusters, cluster_diagnostics, basis = (
            _source_clusters(page, rows, page_size) if rows
            else ([], {}, "native_pdf_long_stroke_cluster")
        )
    if not rows:
        return {"status": "no_native_long_strokes", "source_sha256": source_hash,
                "candidates": [], "diagnostics": diagnostics}
    transform = pymupdf.Matrix(*page_info.pixel_to_pdf_matrix)
    if transform.invert():
        raise ValueError("Singular rendered page transform")
    candidates = []
    for cluster in clusters:
        rect = pymupdf.Rect(*cluster["bbox_pdf"]) * transform
        source_box = [max(0.0, rect.x0), max(0.0, rect.y0),
                      min(image_size[0], rect.x1), min(image_size[1], rect.y1)]
        if not (source_box[0] < source_box[2] and source_box[1] < source_box[3]):
            continue
        source_box = [round(value, 3) for value in source_box]
        refs = cluster["native_path_refs"]
        identity_parts: list[Any] = [source_hash]
        if page_number != 1:
            identity_parts.append(page_number)
        identity_parts.append(sorted(
            (ref["path_index"], ref["item_index"]) for ref in refs
        ))
        identity = json.dumps(identity_parts, separators=(",", ":"))
        candidates.append({
            "id": "review_native_view_" + hashlib.sha256(identity.encode()).hexdigest()[:18],
            "page": str(page_number),
            "source_box": source_box,
            "focus_box": _focus_box(source_box, image_size),
            "label": "原图线条视图区待复查",
            "reason": "版面检测未定位到视图；原始 PDF 线条形成独立区域，视图类型和范围尚待核对",
            "status": "review_required",
            "confidence": None,
            "confidence_kind": None,
            "needs_review": True,
            "verified_view": False,
            "physical_part_verified": False,
            "basis": basis,
            "bbox_pdf": [round(value, 3) for value in cluster["bbox_pdf"]],
            "native_path_refs": refs,
            "support_path_count": cluster["support_path_count"],
            "support_item_count": cluster["support_item_count"],
            "support_length_pdf_points": cluster["support_length_pdf_points"],
        })
    return {
        "status": "review_candidates" if candidates else "no_supported_clusters",
        "source_sha256": source_hash,
        "candidates": candidates,
        "diagnostics": {**diagnostics, **cluster_diagnostics,
                        "candidate_count": len(candidates)},
    }


def native_view_review_candidates(
    document: DrawingDocument,
    image_size: tuple[int, int],
    *,
    existing_views: Sequence[object] = (),
    page_number: int = 1,
) -> list[dict[str, Any]]:
    """Return review-region-shaped candidates without writing artifacts."""

    return native_view_review_evidence(
        document, image_size, existing_views=existing_views,
        page_number=page_number,
    )["candidates"]


__all__ = ["native_view_layout_support", "native_view_review_candidates",
           "native_view_review_evidence"]
