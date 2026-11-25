"""Source-linked review candidates for thin missed or merged drawing views.

Native rectangles and blank bands are geometric clues, not verified views.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from PIL import Image

from .document_schema import DrawingDocument


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _focus(box: list[float], size: tuple[int, int]) -> list[float]:
    padding_x = max(90.0, .04 * size[0])
    padding_y = max(70.0, .04 * size[1])
    return [max(0.0, box[0] - padding_x), max(0.0, box[1] - padding_y),
            min(size[0], box[2] + padding_x), min(size[1], box[3] + padding_y)]


def _overlap_fraction(box: list[float], other: list[float]) -> float:
    intersection = max(0.0, min(box[2], other[2]) - max(box[0], other[0])) * max(
        0.0, min(box[3], other[3]) - max(box[1], other[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return intersection / max(area, 1.0)


def _native_lines(
    page: pymupdf.Page, transform: pymupdf.Matrix,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    horizontal: list[dict[str, Any]] = []
    vertical: list[dict[str, Any]] = []
    for path_index, path in enumerate(page.get_drawings()):
        color = path.get("color")
        if (path.get("type") != "s" or path.get("fill") is not None
                or color is None or max(color) > .5
                or path.get("dashes") not in (None, "[] 0")):
            continue
        for item_index, item in enumerate(path["items"]):
            if item[0] != "l":
                continue
            first, second = item[1] * transform, item[2] * transform
            x0, x1 = sorted((first.x, second.x))
            y0, y1 = sorted((first.y, second.y))
            ref = {"path_index": path_index, "item_index": item_index,
                   "native_kind": "l"}
            tone = max(color)
            if y1 - y0 <= 2 and x1 - x0 >= 60:
                horizontal.append({"x0": x0, "x1": x1, "y": (y0 + y1) / 2,
                                   "ref": ref, "tone": tone})
            elif x1 - x0 <= 2 and y1 - y0 >= 8:
                vertical.append({"x": (x0 + x1) / 2, "y0": y0, "y1": y1,
                                 "ref": ref, "tone": tone})
    return horizontal, vertical


def _thin_rectangles(
    horizontal: list[dict[str, Any]], vertical: list[dict[str, Any]],
    size: tuple[int, int],
) -> list[dict[str, Any]]:
    width, height = size
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for line in horizontal:
        if (line["x1"] - line["x0"] >= max(220, .10 * width)
                and line["x1"] - line["x0"] <= .80 * width
                and line["y"] < .68 * height):
            groups[round(line["x0"] / 3), round(line["x1"] / 3)].append(line)
    rectangles = []
    for lines in groups.values():
        lines.sort(key=lambda row: row["y"])
        for index, upper in enumerate(lines):
            for lower in lines[index + 1:]:
                span = lower["y"] - upper["y"]
                if span > min(200, .12 * height):
                    break
                if span < max(8, .006 * height):
                    continue
                box = [max(upper["x0"], lower["x0"]), upper["y"],
                       min(upper["x1"], lower["x1"]), lower["y"]]
                if box[2] - box[0] < max(220, .10 * width) or (box[2] - box[0]) / span < 8:
                    continue
                walls = []
                for x in (box[0], box[2]):
                    choices = [line for line in vertical
                               if abs(line["x"] - x) <= 3
                               and line["y0"] <= box[1] + .1 * span
                               and line["y1"] >= box[3] - .1 * span]
                    if not choices:
                        break
                    walls.append(min(choices, key=lambda row: abs(row["x"] - x)))
                if len(walls) != 2:
                    continue
                rectangles.append({
                    "bbox": [round(value, 2) for value in box],
                    "native_path_refs": [upper["ref"], lower["ref"],
                                         *(wall["ref"] for wall in walls)],
                    "aspect_ratio": round((box[2] - box[0]) / span, 2),
                })
    rectangles.sort(key=lambda row: (row["bbox"][1], row["bbox"][0],
                                     -(row["bbox"][2] - row["bbox"][0])))
    unique = []
    for rectangle in rectangles:
        if not any(_overlap_fraction(rectangle["bbox"], row["bbox"]) > .85
                   and _overlap_fraction(row["bbox"], rectangle["bbox"]) > .85
                   for row in unique):
            unique.append(rectangle)
    return unique


def _table_rows(rectangles: list[dict[str, Any]]) -> set[int]:
    rejected: set[int] = set()
    for index, row in enumerate(rectangles):
        box = row["bbox"]
        height = box[3] - box[1]
        peers = [other_index for other_index, other in enumerate(rectangles)
                 if abs(other["bbox"][0] - box[0]) <= 3
                 and abs(other["bbox"][2] - box[2]) <= 3
                 and abs((other["bbox"][3] - other["bbox"][1]) - height) <= .1 * height]
        if len(peers) < 3:
            continue
        peers.sort(key=lambda other_index: rectangles[other_index]["bbox"][1])
        for start in range(len(peers) - 2):
            triple = peers[start:start + 3]
            if all(abs(rectangles[first]["bbox"][3] - rectangles[second]["bbox"][1])
                   <= max(3, .1 * height) for first, second in zip(triple, triple[1:])):
                rejected.update(triple)
    return rejected


def _thin_proposals(
    horizontal: list[dict[str, Any]], vertical: list[dict[str, Any]],
    views: list[dict[str, Any]], size: tuple[int, int], source_sha: str,
    page_number: int = 1,
) -> list[dict[str, Any]]:
    rectangles = _thin_rectangles(horizontal, vertical, size)
    table_rows = _table_rows(rectangles)
    proposals = []
    for index, rectangle in enumerate(rectangles):
        if index in table_rows:
            continue
        box = rectangle["bbox"]
        overlap = max((_overlap_fraction(box, view["view_bbox"]) for view in views),
                      default=0.0)
        if overlap >= .25:
            continue
        identity_parts = [source_sha]
        if page_number != 1:
            identity_parts.append(page_number)
        identity_parts.append(rectangle["native_path_refs"])
        identity = json.dumps(identity_parts,
                              sort_keys=True, separators=(",", ":"))
        proposals.append({
            "id": "review_thin_view_" + hashlib.sha256(identity.encode()).hexdigest()[:18],
            "kind": "uncovered_thin_view", "page": str(page_number), "source_box": box,
            "focus_box": _focus(box, size), "label": "细长视图区待复查",
            "reason": "原 PDF 有未被现有视图框覆盖的闭合细长区域；工程视图类型和边界尚未确认",
            "status": "review_required", "confidence": None,
            "review_only": True, "verified_view": False,
            "physical_part_verified": False, "route_hard_constraint_eligible": False,
            "basis": "original_pdf_closed_thin_rectangle",
            "native_path_refs": rectangle["native_path_refs"],
            "aspect_ratio": rectangle["aspect_ratio"],
            "max_existing_view_overlap": round(overlap, 3),
        })
    return proposals


def _complete_panels(
    box: list[float], page_size: tuple[int, int],
    horizontal: list[dict[str, Any]], vertical: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    width, height = box[2] - box[0], box[3] - box[1]
    sidewalls = [line for line in vertical
                 if box[0] + .02 * width < line["x"] < box[2] - .02 * width
                 and box[1] <= line["y0"] < line["y1"] <= box[3]
                 and line["y1"] - line["y0"] >= .14 * height]
    panels = []
    for index, first in enumerate(sidewalls):
        for second in sidewalls[index + 1:]:
            left, right = sorted((first, second), key=lambda row: row["x"])
            span = right["x"] - left["x"]
            wall_height = min(left["y1"] - left["y0"], right["y1"] - right["y0"])
            if (not .16 * width <= span <= .7 * width
                    or span * wall_height < .02 * page_size[0] * page_size[1]
                    or abs(left["y0"] - right["y0"]) > 4
                    or abs(left["y1"] - right["y1"]) > 4):
                continue
            ends = []
            for y in (left["y0"], left["y1"]):
                support = [line for line in horizontal
                           if abs(line["y"] - y) <= 4
                           and line["x0"] <= left["x"] + .15 * span
                           and line["x1"] >= right["x"] - .15 * span
                           and line["x1"] <= right["x"] + 5]
                if not support:
                    break
                ends.append(max(support, key=lambda row: row["x1"] - row["x0"]))
            if len(ends) != 2:
                continue
            candidate = {
                "bbox": [round(left["x"], 2), round(left["y0"], 2),
                         round(right["x"], 2), round(left["y1"], 2)],
                "native_path_refs": [left["ref"], right["ref"],
                                     ends[0]["ref"], ends[1]["ref"]],
            }
            if not any(_overlap_fraction(candidate["bbox"], prior["bbox"]) > .85
                       and _overlap_fraction(prior["bbox"], candidate["bbox"]) > .85
                       for prior in panels):
                panels.append(candidate)
    return panels


def _clear_vertical_band(
    image: np.ndarray, first: list[float], second: list[float],
) -> dict[str, Any] | None:
    left, right = sorted((first, second), key=lambda row: row[0])
    x0, x1 = math.ceil(left[2]), math.floor(right[0])
    y0, y1 = math.ceil(max(left[1], right[1])), math.floor(min(left[3], right[3]))
    if x1 - x0 < 50 or y1 - y0 < 100:
        return None
    dark_columns = np.count_nonzero(image[y0:y1, x0:x1] < 225, axis=0)
    bands = np.flatnonzero(np.diff(np.r_[False, dark_columns <= 1, False])).reshape(-1, 2)
    if not len(bands):
        return None
    start, end = max(bands, key=lambda row: row[1] - row[0])
    if end - start < 50:
        return None
    return {"bbox": [x0 + int(start), y0, x0 + int(end), y1],
            "clear_width": int(end - start)}


def _clear_horizontal_band(
    image: np.ndarray, box: list[float], upper_y: float, lower_y: float,
) -> dict[str, Any] | None:
    width, height = box[2] - box[0], box[3] - box[1]
    y0, y1 = math.ceil(upper_y), math.floor(lower_y)
    x0, x1 = math.ceil(box[0] + .2 * width), math.floor(box[2] - .2 * width)
    if y1 - y0 < max(70, .06 * height):
        return None
    dark_rows = np.count_nonzero(image[y0:y1, x0:x1] < 225, axis=1)
    empty_fraction = float(np.mean(dark_rows <= 1))
    if empty_fraction < .7:
        return None
    return {"bbox": [x0, y0, x1, y1], "empty_row_fraction": round(empty_fraction, 4)}


def _crossing_sidewalls(
    box: list[float], band: dict[str, Any], vertical: list[dict[str, Any]],
) -> bool:
    width = box[2] - box[0]
    y0, y1 = band["bbox"][1], band["bbox"][3]
    crossing = [line for line in vertical
                if box[0] < line["x"] < box[2]
                and line["y0"] <= y0 - 20 and line["y1"] >= y1 + 20]
    for index, left in enumerate(crossing):
        for right in crossing[index + 1:]:
            if (abs(left["x"] - right["x"]) >= .5 * width
                    and abs(left["y0"] - right["y0"]) <= 20
                    and abs(left["y1"] - right["y1"]) <= 20):
                return True
    return False


def _merged_proposals(
    image_path: Path, size: tuple[int, int], views: list[dict[str, Any]],
    horizontal: list[dict[str, Any]], vertical: list[dict[str, Any]],
    source_sha: str, page_number: int = 1,
) -> list[dict[str, Any]]:
    broad = [view for view in views
             if view.get("basis") == "layout_candidate"
             and view["view_bbox"][2] - view["view_bbox"][0] >= .5 * size[0]
             and view["view_bbox"][3] - view["view_bbox"][1] >= .3 * size[1]]
    if not broad:
        return []
    panels_by_view = [(view, _complete_panels(
        view["view_bbox"], size, horizontal, vertical)) for view in broad]
    if not any(len(panels) >= 2 for _, panels in panels_by_view):
        return []
    with Image.open(image_path) as opened:
        image = np.asarray(opened.convert("L"))
    proposals = []
    for view, panels in panels_by_view:
        box = view["view_bbox"]
        width, height = box[2] - box[0], box[3] - box[1]
        for index, first in enumerate(panels):
            for second in panels[index + 1:]:
                left, right = sorted((first, second), key=lambda row: row["bbox"][0])
                if (abs(left["bbox"][1] - right["bbox"][1]) > .04 * height
                        or abs(left["bbox"][3] - right["bbox"][3]) > .04 * height):
                    continue
                vertical_band = _clear_vertical_band(image, left["bbox"], right["bbox"])
                if vertical_band is None:
                    continue
                lower_lines = [line for line in horizontal
                               if box[0] < line["x0"] < line["x1"] < box[2]
                               and line["y"] > max(left["bbox"][3], right["bbox"][3]) + .08 * height
                               and line["y"] < box[3] - .03 * height
                               and line["x1"] - line["x0"] >= .14 * width]
                if not lower_lines:
                    continue
                lower_edge = min(lower_lines, key=lambda line: line["y"])
                horizontal_band = _clear_horizontal_band(
                    image, box, max(left["bbox"][3], right["bbox"][3]), lower_edge["y"])
                if horizontal_band is None or _crossing_sidewalls(box, horizontal_band, vertical):
                    continue
                divider_x = (vertical_band["bbox"][0] + vertical_band["bbox"][2]) / 2
                divider_y = (horizontal_band["bbox"][1] + horizontal_band["bbox"][3]) / 2
                refs = [*left["native_path_refs"], *right["native_path_refs"], lower_edge["ref"]]
                identity_parts = [source_sha]
                if page_number != 1:
                    identity_parts.append(page_number)
                identity_parts.extend((view["id"], refs))
                identity = json.dumps(identity_parts,
                                      sort_keys=True, separators=(",", ":"))
                proposals.append({
                    "id": "review_merged_view_" + hashlib.sha256(identity.encode()).hexdigest()[:18],
                    "kind": "merged_view_gap", "page": str(page_number), "view_id": view["id"],
                    "source_box": box, "candidate_boxes": [
                        [box[0], box[1], divider_x, divider_y],
                        [divider_x, box[1], box[2], divider_y],
                        [box[0], divider_y, box[2], box[3]],
                    ],
                    "panel_boxes": [left["bbox"], right["bbox"]],
                    "vertical_band": vertical_band, "horizontal_band": horizontal_band,
                    "native_path_refs": refs,
                    "label": "宽视图包含独立区域候选",
                    "reason": "原 PDF 的两组独立轮廓及空白带提示宽框可能合并多个视图；分区边界和内容待复查",
                    "status": "review_required", "confidence": None,
                    "review_only": True, "verified_view": False,
                    "physical_part_verified": False, "route_hard_constraint_eligible": False,
                })
    return proposals


def _stacked_proposals(
    image_path: Path, size: tuple[int, int], views: list[dict[str, Any]],
    horizontal: list[dict[str, Any]], vertical: list[dict[str, Any]],
    source_sha: str, page_number: int = 1,
) -> list[dict[str, Any]]:
    broad = [view for view in views
             if view.get("basis") == "layout_candidate"
             and view["view_bbox"][2] - view["view_bbox"][0] >= .5 * size[0]
             and view["view_bbox"][3] - view["view_bbox"][1] >= .3 * size[1]]
    if not broad:
        return []
    with Image.open(image_path) as opened:
        image = np.asarray(opened.convert("L"))
    proposals = []
    for view in broad:
        box = view["view_bbox"]
        width, height = box[2] - box[0], box[3] - box[1]
        long_lines = [line for line in horizontal
                      if line["tone"] <= .35
                      and line["x1"] - line["x0"] >= .2 * width
                      and box[1] <= line["y"] <= box[3]
                      and line["x0"] >= box[0] - .05 * width
                      and line["x1"] <= box[2] + .05 * width]
        levels = sorted({round(line["y"]) for line in long_lines})
        if len(levels) < 4:
            continue
        gaps = sorted(((lower - upper, upper, lower)
                       for upper, lower in zip(levels, levels[1:])), reverse=True)
        gap, upper, lower = gaps[0]
        if (gap < max(140, .2 * height)
                or gap < 1.15 * gaps[1][0]):
            continue
        divider = (upper + lower) / 2
        upper_levels = [level for level in levels if level <= upper]
        lower_levels = [level for level in levels if level >= lower]
        if (min(divider - box[1], box[3] - divider) < .2 * height
                or len(upper_levels) < 2 or len(lower_levels) < 2
                or upper_levels[-1] - upper_levels[0] < max(40, .06 * height)
                or lower_levels[-1] - lower_levels[0] < max(40, .06 * height)):
            continue
        upper_edges = [line for line in long_lines if round(line["y"]) == upper]
        lower_edges = [line for line in long_lines if round(line["y"]) == lower]
        common_span = max((min(first["x1"], second["x1"])
                           - max(first["x0"], second["x0"])
                           for first in upper_edges for second in lower_edges),
                          default=0.0)
        if common_span < .2 * width:
            continue
        band = {"bbox": [box[0], upper + .08 * gap,
                         box[2], lower - .08 * gap]}
        if _crossing_sidewalls(box, band, vertical):
            continue
        x0, x1 = math.ceil(box[0] + .1 * width), math.floor(box[2] - .1 * width)
        row_ink = np.mean(image[upper:lower, x0:x1] < 128, axis=1)
        if (not len(row_ink) or np.median(row_ink) > .03
                or np.quantile(row_ink, .9) > .08):
            continue
        refs = list(dict.fromkeys(
            (row["ref"]["path_index"], row["ref"]["item_index"])
            for row in [*upper_edges, *lower_edges]
        ))
        native_refs = [{"path_index": path_index, "item_index": item_index,
                        "native_kind": "l"} for path_index, item_index in refs]
        identity = json.dumps(
            [source_sha, page_number, view["id"], native_refs],
            sort_keys=True, separators=(",", ":"),
        )
        proposals.append({
            "id": "review_stacked_view_" + hashlib.sha256(identity.encode()).hexdigest()[:18],
            "kind": "stacked_view_gap", "page": str(page_number),
            "view_id": view["id"], "source_box": box,
            "candidate_boxes": [[box[0], box[1], box[2], divider],
                                [box[0], divider, box[2], box[3]]],
            "horizontal_band": {"bbox": [box[0], upper, box[2], lower],
                                "clear_height": gap,
                                "median_dark_fraction": round(float(np.median(row_ink)), 4)},
            "native_path_refs": native_refs,
            "label": "上下视图分区待复查",
            "reason": "原 PDF 深色长线分为上下两组，间隔处缺少连续侧壁；视图边界和类型仍需核实",
            "status": "review_required", "confidence": None,
            "review_only": True, "verified_view": False,
            "physical_part_verified": False, "route_hard_constraint_eligible": False,
            "basis": "original_pdf_dark_native_lines_and_whitespace",
        })
    return proposals


def _broad_boundary_reviews(
    size: tuple[int, int], views: list[dict[str, Any]],
    existing: list[dict[str, Any]], source_sha: str, page_number: int,
) -> list[dict[str, Any]]:
    reviewed_ids = {row["view_id"] for row in existing}
    proposals = []
    for view in views:
        box = view["view_bbox"]
        if (view["id"] in reviewed_ids
                or view.get("basis") != "layout_candidate"
                or box[2] - box[0] < .5 * size[0]
                or box[3] - box[1] < .3 * size[1]
                or len({row["path_index"] for row in
                        view.get("native_path_refs", [])}) < 8):
            continue
        identity = json.dumps(
            [source_sha, page_number, view["id"], box],
            sort_keys=True, separators=(",", ":"),
        )
        proposals.append({
            "id": "review_broad_view_" + hashlib.sha256(identity.encode()).hexdigest()[:18],
            "kind": "broad_view_boundary_unverified",
            "page": str(page_number), "view_id": view["id"],
            "source_box": box, "candidate_boxes": [],
            "source_sha256": source_sha,
            "native_path_refs": view.get("native_path_refs", [])[:12],
            "label": "宽视图区边界待复查",
            "reason": "原 PDF 中版面宽框跨越大面积绘图区；现有几何证据不足以确认它是单个视图",
            "status": "review_required", "confidence": None,
            "review_only": True, "verified_view": False,
            "physical_part_verified": False, "route_hard_constraint_eligible": False,
            "basis": "current_pdf_broad_layout_boundary_unverified",
        })
    return proposals


def build_view_gap_review(
    document: DrawingDocument, views: list[dict[str, Any]],
    *, page_number: int = 1,
) -> dict[str, Any]:
    """Screen the current original PDF and rendered page, never prior artifacts."""
    base: dict[str, Any] = {
        "schema_version": "view_gap_review.v1", "status": "not_run",
        "source_sha256": document.source_sha256,
        "thin_proposals": [], "merged_proposals": [], "boundary_proposals": [],
    }
    page = next((row for row in document.pages
                 if row.page_number == page_number), None)
    source = Path(document.source_path)
    if (page is None or not views or source.suffix.casefold() != ".pdf"
            or not source.is_file()):
        base["reason"] = "no_current_pdf_page_or_views"
        return base
    image = Path(page.image_path)
    if not image.is_file():
        base["reason"] = "rendered_page_unavailable"
        return base
    if _digest(source) != document.source_sha256:
        raise ValueError("View gap review source PDF hash mismatch")
    if _digest(image) != page.image_sha256:
        raise ValueError("View gap review rendered page hash mismatch")
    with Image.open(image) as opened:
        if opened.size != (page.width, page.height):
            raise ValueError("View gap review rendered page size mismatch")
    transform = pymupdf.Matrix(*page.pixel_to_pdf_matrix)
    if transform.invert():
        raise ValueError("View gap review singular page transform")
    with pymupdf.open(source) as pdf:
        if page_number > len(pdf):
            raise ValueError("View gap review page is absent from source PDF")
        horizontal, vertical = _native_lines(pdf[page_number - 1], transform)
    size = (page.width, page.height)
    thin = _thin_proposals(
        horizontal, vertical, views, size, document.source_sha256, page_number,
    )
    merged = _merged_proposals(image, size, views,
                               [row for row in horizontal if row["tone"] <= .35],
                               [row for row in vertical if row["tone"] <= .35
                                and row["y1"] - row["y0"] >= 60],
                               document.source_sha256, page_number)
    merged.extend(_stacked_proposals(
        image, size, views, horizontal,
        [row for row in vertical if row["tone"] <= .35
         and row["y1"] - row["y0"] >= 60],
        document.source_sha256, page_number,
    ))
    boundary = _broad_boundary_reviews(
        size, views, merged, document.source_sha256, page_number,
    )
    base.update({
        "status": "review_required" if thin or merged or boundary else "no_gap_candidates",
        "page": str(page_number),
        "image_sha256": page.image_sha256,
        "coordinate_space": "rendered_page_pixels",
        "thin_proposals": thin, "merged_proposals": merged,
        "boundary_proposals": boundary,
        "diagnostics": {"horizontal_lines": len(horizontal),
                        "vertical_lines": len(vertical),
                        "thin_candidate_count": len(thin),
                        "merged_candidate_count": len(merged),
                        "broad_boundary_review_count": len(boundary)},
    })
    return base


__all__ = ["build_view_gap_review"]
