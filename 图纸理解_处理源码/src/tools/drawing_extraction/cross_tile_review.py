"""Review-only envelopes for views clipped by a page-layout tile boundary."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pymupdf

from .document_schema import DrawingDocument


LAYOUT_RESULT = re.compile(r"^paddle_0001\.json#/results/(\d+)/")
MIN_RIGHT_GAIN = 80.0
MIN_CROSSING_DEPTH = 80.0


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("malformed_artifact")
    return value


def _valid_box(box: Any, size: tuple[int, int]) -> bool:
    return (isinstance(box, list) and len(box) == 4
            and all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in box)
            and 0 <= box[0] < box[2] <= size[0]
            and 0 <= box[1] < box[3] <= size[1])


def _intersection_fraction(first: list[float], second: list[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    smaller = min((first[2] - first[0]) * (first[3] - first[1]),
                  (second[2] - second[0]) * (second[3] - second[1]))
    return width * height / max(smaller, 1.0)


def _vertical_coverage(box: list[float], whole_boxes: list[list[float]]) -> float:
    intervals = sorted((max(box[1], row[1]), min(box[3], row[3]))
                       for row in whole_boxes if min(box[3], row[3]) > max(box[1], row[1]))
    covered = 0.0
    end = box[1]
    for start, stop in intervals:
        covered += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return covered / (box[3] - box[1])


def _focus(box: list[float], size: tuple[int, int]) -> list[float]:
    padding_x, padding_y = max(90.0, .04 * size[0]), max(70.0, .04 * size[1])
    return [max(0.0, box[0] - padding_x), max(0.0, box[1] - padding_y),
            min(size[0], box[2] + padding_x), min(size[1], box[3] + padding_y)]


def _tile_edge(candidate: dict[str, Any], payload: dict[str, Any],
               page_width: int) -> tuple[float | None, list[str]]:
    tiles = {
        tile["pass_id"]: tile["frame"]
        for tile in (payload.get("coverage_ledger") or {}).get("layout_tiles") or []
        if isinstance(tile, dict) and isinstance(tile.get("frame"), list)
    }
    passes = payload.get("pass_ids") or []
    hits: list[tuple[float, str]] = []
    for ref in candidate.get("layout_refs") or []:
        match = LAYOUT_RESULT.match(ref) if isinstance(ref, str) else None
        if match is None or int(match.group(1)) >= len(passes):
            continue
        pass_id = passes[int(match.group(1))]
        frame = tiles.get(pass_id)
        if (frame and len(frame) == 4 and frame[2] < page_width - 32
                and abs(candidate["bbox"][2] - frame[2]) <= 3):
            hits.append((float(frame[2]), pass_id))
    if not hits:
        return None, []
    edge = max(right for right, _ in hits)
    return edge, sorted({pass_id for right, pass_id in hits if right == edge})


def _native_crossings(
    native_page: dict[str, Any], matrix: list[float],
    box: list[float], seam: float, whole_right: float,
) -> list[dict[str, Any]]:
    transform = ~pymupdf.Matrix(*matrix)
    crossings = []
    for path_index, row in enumerate(native_page.get("native_drawings") or []):
        if not isinstance(row, dict):
            continue
        color, source_box = row.get("color"), row.get("bbox")
        if (row.get("type") != "s" or row.get("item_count") != 1
                or row.get("fill") is not None or not isinstance(color, list)
                or not color or max(color) > .35
                or not isinstance(source_box, list) or len(source_box) != 4):
            continue
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in source_box):
            continue
        stroke = pymupdf.Rect(*source_box) * transform
        center_y = (stroke.y0 + stroke.y1) / 2
        if (stroke.width < 150 or stroke.height > 2.5
                or stroke.x0 < box[0] - 40 or stroke.x0 > seam - 24
                or stroke.x1 < seam + MIN_CROSSING_DEPTH
                or stroke.x1 > whole_right + 25
                or center_y < box[1] + 8 or center_y > box[3] - 8):
            continue
        crossings.append({
            "source_ref": {"path_index": path_index, "item_index": 0},
            "bbox": [round(stroke.x0, 2), round(stroke.y0, 2),
                     round(stroke.x1, 2), round(stroke.y1, 2)],
            "right": float(stroke.x1), "y": center_y,
        })
    return sorted(crossings, key=lambda row: (-row["right"], row["y"]))


def _separate_bands(crossings: list[dict[str, Any]], box_height: float) -> int:
    if not crossings:
        return 0
    rightmost = crossings[0]["right"]
    near_end = sorted(row["y"] for row in crossings
                      if row["right"] >= rightmost - 24)
    separation = max(24.0, min(84.0, .12 * box_height))
    bands = 0
    last = -math.inf
    for y in near_end:
        if y - last >= separation:
            bands += 1
            last = y
    return bands


def _probe_one(
    candidate: dict[str, Any], whole_by_id: dict[str, dict[str, Any]],
    tiled: list[dict[str, Any]], payload: dict[str, Any],
    native_page: dict[str, Any], matrix: list[float],
    image_size: tuple[int, int], source_hash: str,
) -> dict[str, Any]:
    box = candidate["bbox"]
    row: dict[str, Any] = {
        "candidate_id": candidate["id"], "original_bbox": box,
        "status": "rejected", "reject_reasons": [],
        "review_only": True, "physical_boundary_verified": False,
        "route_hard_constraint_eligible": False,
    }
    reasons = row["reject_reasons"]
    if candidate.get("status") != "review_required":
        reasons.append("no_cross_path_review_conflict")
    seam, tile_pass_ids = _tile_edge(candidate, payload, image_size[0])
    if seam is None:
        reasons.append("right_edge_not_on_layout_tile_seam")
    matched = [whole_by_id[identifier] for identifier in candidate.get("matched_ids") or []
               if identifier in whole_by_id]
    if not matched:
        reasons.append("no_matching_whole_page_candidate")
    if reasons:
        return row

    whole_boxes = [item["bbox"] for item in matched]
    coverage = _vertical_coverage(box, whole_boxes)
    whole_right = min(item[2] for item in whole_boxes)
    row.update({
        "tile_pass_ids": tile_pass_ids, "tile_right_edge": seam,
        "whole_candidate_ids": [item["id"] for item in matched],
        "matched_whole_vertical_coverage": round(coverage, 4),
        "whole_right_limit": round(whole_right, 2),
    })
    if coverage < .85:
        reasons.append("whole_page_candidates_do_not_cover_same_vertical_region")
    if whole_right - seam < MIN_RIGHT_GAIN:
        reasons.append("whole_page_path_has_no_substantial_rightward_extension")
    if reasons:
        return row

    crossings = _native_crossings(native_page, matrix, box, seam, whole_right)
    bands = _separate_bands(crossings, box[3] - box[1])
    row["native_crossing_count"] = len(crossings)
    row["native_separate_band_count"] = bands
    row["native_support"] = [
        {key: item[key] for key in ("source_ref", "bbox")}
        for item in crossings[:12]
    ]
    if bands < 2:
        reasons.append("fewer_than_two_separate_native_strokes_cross_seam")
        return row
    proposed_right = min(whole_right, crossings[0]["right"] + 8)
    if proposed_right - seam < MIN_RIGHT_GAIN:
        reasons.append("native_supported_extension_is_too_short")
    if proposed_right - seam > .25 * image_size[0]:
        reasons.append("extension_exceeds_quarter_page_width")
    proposed = [box[0], box[1], round(proposed_right, 2), box[3]]
    corridor = [seam, box[1], proposed[2], box[3]]
    if any(_intersection_fraction(corridor, other["bbox"]) > .08
           for other in tiled if other["id"] != candidate["id"]):
        reasons.append("extension_overlaps_another_tiled_view")
    blocked = (payload.get("coverage_ledger") or {}).get("layout_regions") or []
    if any(isinstance(item, dict) and item.get("label") in
           {"table", "text", "paragraph_title"}
           and _valid_box(item.get("bbox"), image_size)
           and _intersection_fraction(corridor, item["bbox"]) > .3
           for item in blocked):
        reasons.append("extension_overlaps_table_or_text_region")
    if reasons:
        return row

    identity = f"{source_hash}:{candidate['id']}:{proposed[2]:.2f}"
    row.update({
        "id": "review_cross_tile_" + hashlib.sha256(identity.encode()).hexdigest()[:18],
        "status": "review_required", "page": 1,
        "source_box": proposed, "focus_box": _focus(proposed, image_size),
        "proposed_bbox": proposed,
        "right_gain_pixels": round(proposed[2] - box[2], 2),
        "whole_path_tail_not_native_supported_pixels": round(
            max(0.0, whole_right - proposed[2]), 2),
        "label": "跨切片视图边界待复查",
        "reason": "局部检测止于切片边界；整页检测与原图线条支持右侧延伸",
    })
    return row


def _validate_sources(
    document: DrawingDocument, artifact: Path,
    paddle_payload: dict[str, Any], comparison: dict[str, Any],
) -> tuple[dict[str, Any], tuple[int, int], list[float]]:
    page = next((item for item in document.pages if item.page_number == 1), None)
    if page is None:
        raise ValueError("missing_first_page")
    image_size = (page.width, page.height)
    image = Path(page.image_path).resolve()
    if not image.is_relative_to(artifact.resolve()):
        raise ValueError("image_outside_artifact")
    if _digest(Path(document.source_path)) != document.source_sha256:
        raise ValueError("source_hash_mismatch")
    if _digest(image) != page.image_sha256:
        raise ValueError("rendered_image_hash_mismatch")
    if (comparison.get("source_sha256") != document.source_sha256
            or comparison.get("image_sha256") != page.image_sha256
            or comparison.get("image_size") != list(image_size)
            or comparison.get("page") != 1):
        raise ValueError("comparison_source_or_image_mismatch")
    stored_comparison = _read_json(artifact / "multipath_candidates_0001.json")
    if stored_comparison != comparison:
        raise ValueError("comparison_not_from_current_artifact")
    stored_payload = _read_json(artifact / "paddle_0001.json")
    current_payload = {key: value for key, value in paddle_payload.items()
                       if key != "_multipath_comparison"}
    if stored_payload != current_payload:
        raise ValueError("ocr_payload_not_from_current_artifact")
    stored_drawing = _read_json(artifact / "drawing.json")
    stored_page = next((item for item in stored_drawing.get("pages") or []
                        if item.get("page_number") == 1), None)
    if (stored_drawing.get("source_sha256") != document.source_sha256
            or stored_page is None
            or stored_page.get("image_sha256") != page.image_sha256):
        raise ValueError("drawing_artifact_source_or_image_mismatch")
    native = _read_json(artifact / "native_pdf.json")
    native_page = next((item for item in native.get("pages") or []
                        if item.get("page") == 1), None)
    if native.get("source_sha256") != document.source_sha256 or native_page is None:
        raise ValueError("native_geometry_source_mismatch")
    if (not isinstance(page.pixel_to_pdf_matrix, list)
            or len(page.pixel_to_pdf_matrix) != 6
            or not all(math.isfinite(value) for value in page.pixel_to_pdf_matrix)):
        raise ValueError("invalid_page_coordinate_transform")
    return native_page, image_size, page.pixel_to_pdf_matrix


def build_cross_tile_review(
    document: DrawingDocument, artifact: Path,
    paddle_payload: dict[str, Any], comparison: dict[str, Any],
) -> dict[str, Any]:
    """Return candidate envelopes without changing view or route evidence."""
    base: dict[str, Any] = {
        "schema_version": "cross_tile_review.v1", "page": 1,
        "source_sha256": document.source_sha256,
        "status": "partial", "proposals": [], "rejections": [],
        "verified_view_count": 0, "route_hard_constraint_eligible": False,
    }
    try:
        native_page, size, matrix = _validate_sources(
            document, artifact, paddle_payload, comparison)
        if comparison.get("status") == "partial":
            base["reason"] = "multipath_comparison_partial"
            return base
        candidates = comparison.get("candidates") or []
        if any(not isinstance(row, dict) or not _valid_box(row.get("bbox"), size)
               for row in candidates):
            raise ValueError("invalid_candidate_box")
        tiled = [row for row in candidates if row.get("path") == "tiled_layout"]
        whole = {row["id"]: row for row in candidates
                 if row.get("path") == "whole_page_layout"}
        if len(whole) != sum(row.get("path") == "whole_page_layout"
                             for row in candidates):
            raise ValueError("duplicate_whole_page_candidate_id")
        rows = [_probe_one(row, whole, tiled, paddle_payload, native_page,
                           matrix, size, document.source_sha256) for row in tiled]
    except (OSError, ValueError, KeyError, TypeError) as error:
        base["reason"] = str(error) if isinstance(error, ValueError) else "malformed_or_missing_evidence"
        return base
    base.update({
        "status": "review_required" if any(row["status"] == "review_required"
                                        for row in rows) else "no_extension_candidates",
        "image_sha256": comparison["image_sha256"],
        "proposals": [row for row in rows if row["status"] == "review_required"],
        "rejections": [row for row in rows if row["status"] == "rejected"],
    })
    return base


__all__ = ["build_cross_tile_review"]
