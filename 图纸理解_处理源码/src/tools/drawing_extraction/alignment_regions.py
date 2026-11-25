"""Refine aligned DWG regions and map them to PDF boxes."""

from __future__ import annotations

from typing import Any

import cv2
from PIL import Image

from .alignment import (
    Alignment,
    ViewMapping,
    _cad_bbox_values,
    _template_match_score,
    pdf_line_mask,
    rasterize_region_template,
    regions_for_mapping,
)
from .region_metadata import region_display_metadata, region_label

def _region_bbox_from_alignment(
    region: dict[str, Any],
    width: int,
    height: int,
    alignment: Alignment,
) -> tuple[float, float, float, float] | None:
    normalized = region.get("bbox_normalized_to_cad_extents")
    if not isinstance(normalized, dict):
        return None
    anchor_x, anchor_y, anchor_w, anchor_h = alignment.bbox
    x = anchor_x + float(normalized.get("x", 0.0)) * anchor_w
    y = anchor_y + float(normalized.get("y", 0.0)) * anchor_h
    w = float(normalized.get("width", 0.0)) * anchor_w
    h = float(normalized.get("height", 0.0)) * anchor_h
    return _clip_pdf_bbox((x, y, w, h), width, height)


def _clip_pdf_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x, y, w, h = bbox
    if w < 5 or h < 5:
        return None
    x = max(0.0, min(width - 1.0, x))
    y = max(0.0, min(height - 1.0, y))
    w = max(5.0, min(width - x, w))
    h = max(5.0, min(height - y, h))
    return x, y, w, h


def refine_region_pdf_bbox(
    image: Image.Image,
    payload: dict[str, Any],
    region: dict[str, Any],
    initial_bbox: tuple[float, float, float, float],
    *,
    min_score: float = 0.075,
) -> tuple[tuple[float, float, float, float], str, float | None]:
    """Refine one view's PDF bbox by matching its DWG template near the initial bbox."""

    template = rasterize_region_template(payload, region)
    if template is None:
        return initial_bbox, "global_alignment", None
    target = pdf_line_mask(image)
    image_width, image_height = image.size
    initial_x, initial_y, initial_w, initial_h = initial_bbox
    template_ratio = float(template.shape[1]) / max(float(template.shape[0]), 1.0)
    margin_x = max(28, int(round(initial_w * 0.28)))
    margin_y = max(28, int(round(initial_h * 0.34)))
    left = max(0, int(round(initial_x - margin_x)))
    top = max(0, int(round(initial_y - margin_y)))
    right = min(image_width, int(round(initial_x + initial_w + margin_x)))
    bottom = min(image_height, int(round(initial_y + initial_h + margin_y)))
    roi = target[top:bottom, left:right]
    if roi.shape[0] < 16 or roi.shape[1] < 16:
        return initial_bbox, "global_alignment", None

    size_candidates: set[tuple[int, int]] = set()
    for scale in (0.88, 0.94, 1.0, 1.06, 1.12):
        width_from_base = int(round(initial_w * scale))
        height_from_width = int(round(width_from_base / template_ratio))
        height_from_base = int(round(initial_h * scale))
        width_from_height = int(round(height_from_base * template_ratio))
        size_candidates.add((width_from_base, height_from_width))
        size_candidates.add((width_from_height, height_from_base))

    best: tuple[float, tuple[float, float, float, float]] | None = None
    for target_w, target_h in size_candidates:
        if target_w < 20 or target_h < 20 or target_w > roi.shape[1] or target_h > roi.shape[0]:
            continue
        resized = cv2.resize(template, (target_w, target_h), interpolation=cv2.INTER_AREA)
        score_location = _template_match_score(roi, resized)
        if score_location is None:
            continue
        score, location = score_location
        pdf_bbox = (float(left + location[0]), float(top + location[1]), float(target_w), float(target_h))
        if best is None or score > best[0]:
            best = (score, pdf_bbox)

    if best is None or best[0] < min_score:
        return initial_bbox, "global_alignment", None
    clipped = _clip_pdf_bbox(best[1], image_width, image_height)
    if clipped is None:
        return initial_bbox, "global_alignment", None
    return clipped, f"local_template_match:{best[0]:.3f}", best[0]


def _cad_distance_to_view(region_bbox: dict[str, Any] | None, view_bbox: dict[str, Any]) -> float | None:
    region_values = _cad_bbox_values(region_bbox)
    view_values = _cad_bbox_values(view_bbox)
    if region_values is None or view_values is None:
        return None
    min_x, min_y, max_x, max_y = region_values
    view_min_x, view_min_y, view_max_x, view_max_y = view_values
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    view_w = max(view_max_x - view_min_x, 1.0)
    view_h = max(view_max_y - view_min_y, 1.0)
    dx = max(view_min_x - center_x, 0.0, center_x - view_max_x) / view_w
    dy = max(view_min_y - center_y, 0.0, center_y - view_max_y) / view_h
    return (dx * dx + dy * dy) ** 0.5


def _best_view_mapping(region: dict[str, Any], mappings: list[ViewMapping]) -> ViewMapping | None:
    best: tuple[float, ViewMapping] | None = None
    for mapping in mappings:
        distance = _cad_distance_to_view(region.get("bbox_cad"), mapping.cad_bbox)
        if distance is None:
            continue
        if best is None or distance < best[0]:
            best = (distance, mapping)
    if best is None or best[0] > 0.65:
        return None
    return best[1]


def _map_cad_bbox_with_view(region: dict[str, Any], mapping: ViewMapping, width: int, height: int) -> tuple[float, float, float, float] | None:
    region_values = _cad_bbox_values(region.get("bbox_cad"))
    view_values = _cad_bbox_values(mapping.cad_bbox)
    if region_values is None or view_values is None:
        return None
    min_x, min_y, max_x, max_y = region_values
    view_min_x, view_min_y, view_max_x, view_max_y = view_values
    view_w = view_max_x - view_min_x
    view_h = view_max_y - view_min_y
    if view_w <= 0 or view_h <= 0:
        return None
    pdf_x, pdf_y, pdf_w, pdf_h = mapping.pdf_bbox
    x = pdf_x + ((min_x - view_min_x) / view_w) * pdf_w
    y = pdf_y + (1.0 - ((max_y - view_min_y) / view_h)) * pdf_h
    w = ((max_x - min_x) / view_w) * pdf_w
    h = ((max_y - min_y) / view_h) * pdf_h
    return _clip_pdf_bbox((x, y, w, h), width, height)


def _box_from_pdf_bbox(
    region: dict[str, Any],
    pdf_bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    alignment: Alignment,
    class_id: dict[str, int],
    payload: dict[str, Any] | None,
    *,
    alignment_source: str | None = None,
    alignment_bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, Any] | None:
    clipped = _clip_pdf_bbox(pdf_bbox, width, height)
    if clipped is None:
        return None
    x, y, w, h = clipped
    label = region_label(region)
    metadata = region_display_metadata(region, payload)
    box_alignment_bbox = alignment_bbox or alignment.bbox
    anchor_x, anchor_y, anchor_w, anchor_h = box_alignment_bbox
    return {
        "id": str(region.get("id", label)),
        "label": label,
        "class_id": class_id[label],
        "bbox": {"x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2)},
        "bbox_xyxy": [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)],
        "normalized_bbox": {
            "x": round(x / width, 6),
            "y": round(y / height, 6),
            "w": round(w / width, 6),
            "h": round(h / height, 6),
        },
        "source": "dwg_extraction_region",
        "alignment_source": alignment_source or alignment.source,
        "alignment_bbox": {
            "x": round(anchor_x, 2),
            "y": round(anchor_y, 2),
            "w": round(anchor_w, 2),
            "h": round(anchor_h, 2),
        },
        **metadata,
    }


def region_to_box(
    region: dict[str, Any],
    width: int,
    height: int,
    alignment: Alignment,
    class_id: dict[str, int],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    pdf_bbox = _region_bbox_from_alignment(region, width, height, alignment)
    if pdf_bbox is None:
        return None
    return _box_from_pdf_bbox(region, pdf_bbox, width, height, alignment, class_id, payload)


def regions_to_boxes(
    payload: dict[str, Any],
    image: Image.Image,
    alignment: Alignment,
    class_id: dict[str, int],
) -> list[dict[str, Any]]:
    """Map DWG regions to PDF boxes, refining each CAD view separately when possible."""

    width, height = image.size
    regions = regions_for_mapping(payload)
    view_mappings: list[ViewMapping] = []
    view_boxes: dict[str, dict[str, Any]] = {}

    for region in regions:
        if region.get("kind") != "subfigure":
            continue
        base_bbox = _region_bbox_from_alignment(region, width, height, alignment)
        cad_bbox = region.get("bbox_cad")
        if base_bbox is None or not isinstance(cad_bbox, dict):
            continue
        refined_bbox, source, score = refine_region_pdf_bbox(image, payload, region, base_bbox)
        region_id = str(region.get("id", "subfigure"))
        mapping = ViewMapping(
            region_id=region_id,
            cad_bbox=cad_bbox,
            pdf_bbox=refined_bbox,
            source=f"{alignment.source};{source}",
            score=score,
        )
        view_mappings.append(mapping)
        box = _box_from_pdf_bbox(
            region,
            refined_bbox,
            width,
            height,
            alignment,
            class_id,
            payload,
            alignment_source=mapping.source,
            alignment_bbox=refined_bbox,
        )
        if box is not None:
            if score is not None:
                box["local_alignment_score"] = round(score, 4)
            view_boxes[region_id] = box

    boxes: list[dict[str, Any]] = []
    for region in regions:
        if region.get("kind") == "subfigure":
            box = view_boxes.get(str(region.get("id", "subfigure")))
            if box is not None:
                boxes.append(box)
            continue

        mapping = _best_view_mapping(region, view_mappings)
        pdf_bbox = None
        source = None
        alignment_bbox = None
        if mapping is not None:
            pdf_bbox = _map_cad_bbox_with_view(region, mapping, width, height)
            source = mapping.source + ";dimension_to_nearest_view"
            alignment_bbox = mapping.pdf_bbox
        if pdf_bbox is None:
            pdf_bbox = _region_bbox_from_alignment(region, width, height, alignment)
            source = alignment.source
            alignment_bbox = alignment.bbox
        if pdf_bbox is None:
            continue
        box = _box_from_pdf_bbox(
            region,
            pdf_bbox,
            width,
            height,
            alignment,
            class_id,
            payload,
            alignment_source=source,
            alignment_bbox=alignment_bbox,
        )
        if box is not None:
            if mapping is not None:
                box["view_region_id"] = mapping.region_id
            boxes.append(box)

    return boxes
