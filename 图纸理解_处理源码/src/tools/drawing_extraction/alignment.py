"""Align DWG extraction regions to rendered PNG pages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageChops

from .region_metadata import region_display_metadata, region_display_text, region_label


@dataclass(frozen=True)
class ComponentCandidate:
    x: int
    y: int
    width: int
    height: int
    area: int
    ratio: float
    ratio_error: float
    score: float


@dataclass(frozen=True)
class Alignment:
    bbox: tuple[float, float, float, float]
    source: str
    cad_ratio: float | None
    candidates: list[ComponentCandidate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox,
            "source": self.source,
            "cad_ratio": self.cad_ratio,
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }


@dataclass(frozen=True)
class ViewMapping:
    region_id: str
    cad_bbox: dict[str, float]
    pdf_bbox: tuple[float, float, float, float]
    source: str
    score: float | None = None


def cad_aspect_ratio(payload: dict[str, Any]) -> float | None:
    extents = payload.get("cad", {}).get("cad_extents")
    if not isinstance(extents, dict):
        return None
    width = float(extents.get("max_x", 0.0)) - float(extents.get("min_x", 0.0))
    height = float(extents.get("max_y", 0.0)) - float(extents.get("min_y", 0.0))
    if width <= 0 or height <= 0:
        return None
    return width / height


def cad_extents(payload: dict[str, Any]) -> tuple[float, float, float, float] | None:
    extents = payload.get("cad", {}).get("cad_extents")
    if not isinstance(extents, dict):
        return None
    try:
        min_x = float(extents["min_x"])
        min_y = float(extents["min_y"])
        max_x = float(extents["max_x"])
        max_y = float(extents["max_y"])
    except (KeyError, TypeError, ValueError):
        return None
    if max_x <= min_x or max_y <= min_y:
        return None
    return min_x, min_y, max_x, max_y


def detect_foreground_bbox(image: Image.Image, threshold: int = 18, padding: int = 4) -> tuple[float, float, float, float]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, (255, 255, 255))).convert("L")
    mask = diff.point(lambda value: 255 if value > threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return (0.0, 0.0, float(width), float(height))
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    if right - left < 10 or bottom - top < 10:
        return (0.0, 0.0, float(width), float(height))
    return (float(left), float(top), float(right - left), float(bottom - top))


def component_candidates(
    image: Image.Image,
    cad_ratio: float,
    min_area_fraction: float = 0.0015,
) -> list[ComponentCandidate]:
    gray = np.array(image.convert("L"))
    height, width = gray.shape
    mask = (gray < 235).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    _, _, stats, _ = cv2.connectedComponentsWithStats(dilated, 8)

    min_area = max(100, int(width * height * min_area_fraction))
    candidates: list[ComponentCandidate] = []
    for x, y, w, h, area in stats[1:]:
        if area < min_area or w < 20 or h < 20:
            continue
        ratio = float(w) / float(h)
        ratio_error = abs(log(ratio / cad_ratio))
        area_bonus = min(float(area) / float(width * height), 0.08)
        score = ratio_error - area_bonus
        candidates.append(
            ComponentCandidate(
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                area=int(area),
                ratio=ratio,
                ratio_error=ratio_error,
                score=score,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.score)


def _cad_to_template_point(
    x: float,
    y: float,
    *,
    extents: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int]:
    min_x, min_y, max_x, max_y = extents
    px = (x - min_x) / (max_x - min_x) * float(width - 1)
    py = (max_y - y) / (max_y - min_y) * float(height - 1)
    return int(round(px)), int(round(py))


def _draw_cad_bbox(
    mask: np.ndarray,
    bbox: dict[str, Any],
    *,
    extents: tuple[float, float, float, float],
    color: int,
    thickness: int,
) -> None:
    try:
        p1 = _cad_to_template_point(float(bbox["min_x"]), float(bbox["min_y"]), extents=extents, width=mask.shape[1], height=mask.shape[0])
        p2 = _cad_to_template_point(float(bbox["max_x"]), float(bbox["max_y"]), extents=extents, width=mask.shape[1], height=mask.shape[0])
    except (KeyError, TypeError, ValueError):
        return
    x1, x2 = sorted((p1[0], p2[0]))
    y1, y2 = sorted((p1[1], p2[1]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    cv2.rectangle(mask, (x1, y1), (x2, y2), color, thickness)


def _cad_bbox_values(bbox: dict[str, Any] | None) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, dict):
        return None
    try:
        min_x = float(bbox["min_x"])
        min_y = float(bbox["min_y"])
        max_x = float(bbox["max_x"])
        max_y = float(bbox["max_y"])
    except (KeyError, TypeError, ValueError):
        return None
    if max_x <= min_x or max_y <= min_y:
        return None
    return min_x, min_y, max_x, max_y


def _cad_bboxes_overlap(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    values_a = _cad_bbox_values(a)
    values_b = _cad_bbox_values(b)
    if values_a is None or values_b is None:
        return False
    a_min_x, a_min_y, a_max_x, a_max_y = values_a
    b_min_x, b_min_y, b_max_x, b_max_y = values_b
    return not (a_max_x < b_min_x or b_max_x < a_min_x or a_max_y < b_min_y or b_max_y < a_min_y)


def _template_bbox_to_cad(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    extents: tuple[float, float, float, float],
    template_width: int,
    template_height: int,
) -> dict[str, float]:
    min_x, min_y, max_x, max_y = extents
    cad_x1 = min_x + (float(x) / float(max(template_width - 1, 1))) * (max_x - min_x)
    cad_x2 = min_x + (float(x + width) / float(max(template_width - 1, 1))) * (max_x - min_x)
    cad_y2 = max_y - (float(y) / float(max(template_height - 1, 1))) * (max_y - min_y)
    cad_y1 = max_y - (float(y + height) / float(max(template_height - 1, 1))) * (max_y - min_y)
    return {
        "min_x": min(cad_x1, cad_x2),
        "min_y": min(cad_y1, cad_y2),
        "max_x": max(cad_x1, cad_x2),
        "max_y": max(cad_y1, cad_y2),
    }


def _normalize_cad_bbox(bbox: dict[str, float], frame: tuple[float, float, float, float]) -> dict[str, float]:
    min_x, min_y, max_x, max_y = frame
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    return {
        "x": (bbox["min_x"] - min_x) / width,
        "y": 1.0 - ((bbox["max_y"] - min_y) / height),
        "width": (bbox["max_x"] - bbox["min_x"]) / width,
        "height": (bbox["max_y"] - bbox["min_y"]) / height,
    }


def rasterize_dwg_template(
    payload: dict[str, Any],
    *,
    max_side: int = 650,
    include_dimensions: bool = True,
) -> np.ndarray | None:
    """Rasterize DWG JSON entities into a sparse binary template in CAD extents."""

    extents = cad_extents(payload)
    ratio = cad_aspect_ratio(payload)
    if extents is None or ratio is None:
        return None
    if ratio >= 1.0:
        width = max_side
        height = max(64, int(round(max_side / ratio)))
    else:
        height = max_side
        width = max(64, int(round(max_side * ratio)))

    mask = np.zeros((height, width), dtype=np.uint8)
    for entity in payload.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entity_type = entity.get("type")
        details = entity.get("details") if isinstance(entity.get("details"), dict) else {}
        if entity_type == "LINE" and isinstance(details.get("start"), list) and isinstance(details.get("end"), list):
            try:
                start = details["start"]
                end = details["end"]
                p1 = _cad_to_template_point(float(start[0]), float(start[1]), extents=extents, width=width, height=height)
                p2 = _cad_to_template_point(float(end[0]), float(end[1]), extents=extents, width=width, height=height)
            except (TypeError, ValueError, IndexError):
                continue
            cv2.line(mask, p1, p2, 255, 2)
            continue

        bbox = entity.get("bbox_cad")
        if isinstance(bbox, dict) and entity_type in {"HATCH", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC", "SPLINE"}:
            _draw_cad_bbox(mask, bbox, extents=extents, color=210, thickness=1)

    if include_dimensions:
        for dimension in payload.get("dimensions", []):
            if isinstance(dimension, dict) and isinstance(dimension.get("bbox_cad"), dict):
                _draw_cad_bbox(mask, dimension["bbox_cad"], extents=extents, color=255, thickness=1)

    if int(np.count_nonzero(mask)) < 20:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(mask, kernel, iterations=1)


def rasterize_region_template(
    payload: dict[str, Any],
    region: dict[str, Any],
    *,
    max_side: int = 520,
) -> np.ndarray | None:
    """Rasterize one CAD view region into a local binary template."""

    region_values = _cad_bbox_values(region.get("bbox_cad"))
    if region_values is None:
        return None
    min_x, min_y, max_x, max_y = region_values
    ratio = (max_x - min_x) / (max_y - min_y)
    if ratio >= 1.0:
        width = max_side
        height = max(32, int(round(max_side / ratio)))
    else:
        height = max_side
        width = max(32, int(round(max_side * ratio)))

    extents = (min_x, min_y, max_x, max_y)
    mask = np.zeros((height, width), dtype=np.uint8)
    for entity in payload.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entity_bbox = entity.get("bbox_cad")
        entity_type = entity.get("type")
        details = entity.get("details") if isinstance(entity.get("details"), dict) else {}
        if entity_type == "LINE" and isinstance(details.get("start"), list) and isinstance(details.get("end"), list):
            try:
                start = details["start"]
                end = details["end"]
                line_bbox = {
                    "min_x": min(float(start[0]), float(end[0])),
                    "min_y": min(float(start[1]), float(end[1])),
                    "max_x": max(float(start[0]), float(end[0])),
                    "max_y": max(float(start[1]), float(end[1])),
                }
            except (TypeError, ValueError, IndexError):
                continue
            if not _cad_bboxes_overlap(region.get("bbox_cad"), line_bbox):
                continue
            try:
                p1 = _cad_to_template_point(float(start[0]), float(start[1]), extents=extents, width=width, height=height)
                p2 = _cad_to_template_point(float(end[0]), float(end[1]), extents=extents, width=width, height=height)
            except (TypeError, ValueError, IndexError):
                continue
            cv2.line(mask, p1, p2, 255, 2)
            continue

        if isinstance(entity_bbox, dict) and entity_type in {"HATCH", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC", "SPLINE"}:
            if _cad_bboxes_overlap(region.get("bbox_cad"), entity_bbox):
                _draw_cad_bbox(mask, entity_bbox, extents=extents, color=210, thickness=1)

    if int(np.count_nonzero(mask)) < 20:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(mask, kernel, iterations=1)


def view_cluster_regions(
    payload: dict[str, Any],
    *,
    max_regions: int = 8,
    min_area_fraction: float = 0.002,
) -> list[dict[str, Any]]:
    """Split modelspace geometry into coarse CAD view clusters."""

    extents = cad_extents(payload)
    if extents is None:
        return []
    template = rasterize_dwg_template(payload, max_side=900, include_dimensions=False)
    if template is None:
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    grouped = cv2.morphologyEx(template, cv2.MORPH_CLOSE, kernel, iterations=2)
    grouped = cv2.dilate(grouped, kernel, iterations=1)
    _, _, stats, _ = cv2.connectedComponentsWithStats((grouped > 0).astype(np.uint8), 8)

    template_area = float(template.shape[0] * template.shape[1])
    candidates = []
    for x, y, width, height, area in stats[1:]:
        if area < template_area * min_area_fraction or width < 20 or height < 20:
            continue
        ratio = float(width) / max(float(height), 1.0)
        if ratio > 80 or ratio < 0.0125:
            continue
        candidates.append((int(area), int(x), int(y), int(width), int(height)))
    candidates.sort(reverse=True)
    if len(candidates) <= 1:
        return []

    regions = []
    for index, (_, x, y, width, height) in enumerate(candidates[:max_regions], start=1):
        bbox = _template_bbox_to_cad(
            x,
            y,
            width,
            height,
            extents=extents,
            template_width=template.shape[1],
            template_height=template.shape[0],
        )
        role = "main_view" if index == 1 else "detail_view"
        regions.append(
            {
                "id": f"{role}_cluster_{index}",
                "kind": "subfigure",
                "role": role,
                "description": "DWG geometry view cluster split from modelspace entities.",
                "bbox_cad": bbox,
                "bbox_normalized_to_cad_extents": _normalize_cad_bbox(bbox, extents),
                "entity_types": payload.get("cad", {}).get("entity_counts", {}),
                "alignment_note": "Synthetic DWG view cluster; mapped through PDF-DWG alignment.",
            }
        )
    return regions


def regions_for_mapping(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return view clusters plus annotation regions for PDF mapping."""

    source_regions = [region for region in payload.get("regions", []) if isinstance(region, dict)]
    annotations = [region for region in source_regions if region.get("kind") == "annotation"]
    view_clusters = view_cluster_regions(payload)
    if view_clusters:
        return view_clusters + annotations
    return source_regions


def pdf_line_mask(image: Image.Image) -> np.ndarray:
    gray = np.array(image.convert("L"))
    mask = (gray < 235).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(mask, kernel, iterations=1)


def _template_match_score(target: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int]] | None:
    if template.shape[0] > target.shape[0] or template.shape[1] > target.shape[1]:
        return None
    if np.count_nonzero(template) < 20 or np.count_nonzero(target) < 20:
        return None
    result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    _, max_value, _, max_location = cv2.minMaxLoc(result)
    return float(max_value), (int(max_location[0]), int(max_location[1]))


def choose_template_alignment(
    image: Image.Image,
    payload: dict[str, Any],
    *,
    min_score: float = 0.09,
) -> Alignment | None:
    cad_ratio = cad_aspect_ratio(payload)
    if cad_ratio is None:
        return None
    template = rasterize_dwg_template(payload)
    if template is None:
        return None

    target_full = pdf_line_mask(image)
    foreground = detect_foreground_bbox(image)
    candidates = component_candidates(image, cad_ratio, min_area_fraction=0.0006)
    if not candidates:
        fx, fy, fw, fh = foreground
        candidates.append(
            ComponentCandidate(
                x=int(fx),
                y=int(fy),
                width=int(fw),
                height=int(fh),
                area=int(fw * fh),
                ratio=float(fw) / max(float(fh), 1.0),
                ratio_error=abs(log((float(fw) / max(float(fh), 1.0)) / cad_ratio)),
                score=999.0,
            )
        )

    image_width, image_height = image.size
    match_scale = min(1.0, 1800.0 / float(max(image_width, image_height)))
    if match_scale < 1.0:
        target = cv2.resize(
            target_full,
            (max(1, int(round(image_width * match_scale))), max(1, int(round(image_height * match_scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        target = target_full
    template_ratio = float(template.shape[1]) / float(template.shape[0])
    best: tuple[float, tuple[float, float, float, float]] | None = None

    for candidate in candidates[:8]:
        if candidate.ratio_error > 0.45:
            continue
        for scale in (0.96, 1.0, 1.04):
            target_w = int(round(candidate.width * scale))
            target_h = int(round(target_w / template_ratio))
            if target_w < 40 or target_h < 40 or target_w >= image_width or target_h >= image_height:
                continue
            match_w = max(12, int(round(target_w * match_scale)))
            match_h = max(12, int(round(target_h * match_scale)))
            resized = cv2.resize(template, (match_w, match_h), interpolation=cv2.INTER_AREA)
            margin_x = max(20, int(round(target_w * 0.14)))
            margin_y = max(20, int(round(target_h * 0.14)))
            left = max(0, int(round((candidate.x - margin_x) * match_scale)))
            top = max(0, int(round((candidate.y - margin_y) * match_scale)))
            right = min(target.shape[1], int(round((candidate.x + candidate.width + margin_x) * match_scale)))
            bottom = min(target.shape[0], int(round((candidate.y + candidate.height + margin_y) * match_scale)))
            roi = target[top:bottom, left:right]
            score_location = _template_match_score(roi, resized)
            if score_location is None:
                continue
            score, location = score_location
            bbox = (
                float((left + location[0]) / match_scale),
                float((top + location[1]) / match_scale),
                float(match_w / match_scale),
                float(match_h / match_scale),
            )
            if best is None or score > best[0]:
                best = (score, bbox)

    if best is None or best[0] < min_score:
        return None
    score, bbox = best
    _, _, matched_width, matched_height = bbox
    image_area = float(image_width * image_height)
    matched_area = matched_width * matched_height
    if (
        matched_area < image_area * 0.008
        or matched_width < image_width * 0.05
        or matched_height < image_height * 0.035
    ):
        return None
    return Alignment(
        bbox,
        f"pdf_dwg_template_match:{score:.3f}",
        cad_ratio,
        candidates,
    )


def choose_alignment(
    image: Image.Image,
    payload: dict[str, Any],
    padding: int = 8,
    max_ratio_error: float = 0.18,
) -> Alignment:
    template_alignment = choose_template_alignment(image, payload)
    if template_alignment is not None:
        return template_alignment

    foreground = detect_foreground_bbox(image)
    cad_ratio = cad_aspect_ratio(payload)
    if cad_ratio is None:
        return Alignment(foreground, "png_foreground_bbox", cad_ratio, [])

    candidates = component_candidates(image, cad_ratio)
    if not candidates:
        return Alignment(foreground, "png_foreground_bbox", cad_ratio, candidates)

    best = candidates[0]
    if best.ratio_error > max_ratio_error:
        return Alignment(foreground, "png_foreground_bbox", cad_ratio, candidates)

    width, height = image.size
    left = max(0, best.x - padding)
    top = max(0, best.y - padding)
    right = min(width, best.x + best.width + padding)
    bottom = min(height, best.y + best.height + padding)
    return Alignment(
        (float(left), float(top), float(right - left), float(bottom - top)),
        "png_cad_component_bbox",
        cad_ratio,
        candidates,
    )


def choose_alignment_for_path(image_path: Path, payload: dict[str, Any]) -> Alignment:
    with Image.open(image_path) as image:
        return choose_alignment(image.convert("RGB"), payload)


def region_to_box(
    region: dict[str, Any],
    width: int,
    height: int,
    alignment: Alignment,
    class_id: dict[str, int],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    from .alignment_regions import region_to_box as map_region

    return map_region(region, width, height, alignment, class_id, payload)


def regions_to_boxes(
    payload: dict[str, Any],
    image: Image.Image,
    alignment: Alignment,
    class_id: dict[str, int],
) -> list[dict[str, Any]]:
    from .alignment_regions import regions_to_boxes as map_regions

    return map_regions(payload, image, alignment, class_id)
