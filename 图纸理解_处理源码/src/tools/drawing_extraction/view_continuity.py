"""Flag horizontal view splits contradicted by original PDF sidewalls."""

from __future__ import annotations

import hashlib
from typing import Any


def _paired_sidewalls(
    box: list[float], divider: float, strokes: list[dict[str, Any]],
    filled_paths: set[int],
) -> dict[str, Any] | None:
    width, height = box[2] - box[0], box[3] - box[1]
    margin = max(22.0, .07 * height)
    vertical = []
    for stroke in strokes:
        points = stroke["points"]
        if (len(points) != 2 or any(ref["path_index"] in filled_paths
                                    for ref in stroke["sources"])):
            continue
        first, second = points
        x0, x1 = sorted((first[0], second[0]))
        y0, y1 = sorted((first[1], second[1]))
        if (x1 - x0 > 2.5 or y1 - y0 < .15 * height
                or not box[0] <= x0 <= box[2]
                or not box[1] <= y0 < divider - margin
                or not divider + margin < y1 <= box[3]):
            continue
        vertical.append({"x": (x0 + x1) / 2, "y0": y0, "y1": y1,
                         "stroke_id": stroke["id"], "source_refs": stroke["sources"]})
    pairs = []
    for index, first in enumerate(vertical):
        for second in vertical[index + 1:]:
            left, right = sorted((first, second), key=lambda row: row["x"])
            if (right["x"] - left["x"] < .65 * width
                    or left["x"] > box[0] + .25 * width
                    or right["x"] < box[2] - .25 * width
                    or abs(left["y0"] - right["y0"]) > .04 * height
                    or abs(left["y1"] - right["y1"]) > .04 * height):
                continue
            pairs.append((abs(left["y0"] - right["y0"])
                          + abs(left["y1"] - right["y1"]),
                          -(right["x"] - left["x"]), left, right))
    if not pairs:
        return None
    _, _, left, right = min(pairs, key=lambda row: row[:2])
    return {
        "sidewall_bbox": [round(left["x"], 2), round(min(left["y0"], right["y0"]), 2),
                          round(right["x"], 2), round(max(left["y1"], right["y1"]), 2)],
        "stroke_ids": [left["stroke_id"], right["stroke_id"]],
        "native_path_refs": [*left["source_refs"], *right["source_refs"]],
    }


def reconcile_horizontal_splits(
    views: list[dict[str, Any]], strokes: list[dict[str, Any]],
    filled_paths: set[int], source_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace a contradicted split with one unresolved, source-linked envelope."""
    consumed: set[str] = set()
    replacements: dict[str, dict[str, Any]] = {}
    reviews = []
    for upper in views:
        if upper["id"] in consumed or upper.get("basis") != "layout_and_paired_native_contours":
            continue
        first = upper["view_bbox"]
        for lower in views:
            if (lower["id"] == upper["id"] or lower["id"] in consumed
                    or lower.get("basis") != "layout_and_paired_native_contours"):
                continue
            second = lower["view_bbox"]
            if (abs(first[0] - second[0]) > 3 or abs(first[2] - second[2]) > 3
                    or abs(first[3] - second[1]) > 3
                    or not set(upper.get("raw_result_refs") or [])
                    & set(lower.get("raw_result_refs") or [])):
                continue
            box = [min(first[0], second[0]), first[1],
                   max(first[2], second[2]), second[3]]
            divider = (first[3] + second[1]) / 2
            sidewalls = _paired_sidewalls(box, divider, strokes, filled_paths)
            if sidewalls is None:
                continue
            identity = repr((source_sha256, upper["id"], lower["id"], box))
            identifier = "drawing_view_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
            evidence_ids = list(dict.fromkeys([*upper["evidence_ids"],
                                               *lower["evidence_ids"]]))
            raw_refs = list(dict.fromkeys([*upper["raw_result_refs"],
                                           *lower["raw_result_refs"]]))
            native_refs = list(dict.fromkeys(
                tuple(sorted(ref.items())) for ref in [
                    *upper["native_path_refs"], *lower["native_path_refs"],
                    *sidewalls["native_path_refs"],
                ]
            ))
            replacement = {
                **upper, "id": identifier, "view_bbox": box,
                "label": "未命名视图候选", "status": "unresolved",
                "confidence": None, "confidence_kind": None,
                "evidence_ids": evidence_ids, "raw_result_refs": raw_refs,
                "native_path_refs": [dict(ref) for ref in native_refs],
                "basis": "paired_native_sidewalls_continuity_review",
                "split_conflict": {"original_view_ids": [upper["id"], lower["id"]],
                                   "original_boxes": [first, second],
                                   "divider_y": round(divider, 3), **sidewalls},
            }
            replacements[upper["id"]] = replacement
            consumed.update((upper["id"], lower["id"]))
            reviews.append({
                "id": "view_split_conflict_" + hashlib.sha256(identity.encode()).hexdigest()[:18],
                "kind": "view_split_conflict", "label": "连续轮廓跨越视图拆分线",
                "reason": "成对原图侧壁跨越自动拆分线；整体边界和视图归属待核实",
                "source_box": box, "view_id": identifier,
                "candidate_boxes": [first, second],
                "native_path_refs": sidewalls["native_path_refs"],
                "status": "review_required", "route_hard_constraint_eligible": False,
            })
            break
    reconciled = []
    for view in views:
        if view["id"] in replacements:
            reconciled.append(replacements[view["id"]])
        elif view["id"] not in consumed:
            reconciled.append(view)
    return reconciled, reviews


__all__ = ["reconcile_horizontal_splits"]
