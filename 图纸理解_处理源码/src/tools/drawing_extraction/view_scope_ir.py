"""Source-matched view, annotation-scope and native-geometry candidates.

This read model deliberately stops at visible evidence. A native stroke group
is not a physical part, and a nearby section marker is not a verified cut.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import pymupdf

from .dimension_entity_evidence import build_dimension_entity_evidence, extract_dark_fill_marks
from .dimension_first_contact import find_first_contact_targets
from .dimension_topology import build_dimension_topology
from .view_continuity import reconcile_horizontal_splits
from .view_gap_review import build_view_gap_review
from .document_schema import DrawingDocument
from .local_geometry import dense_parallel_ids, extract_strokes, stroke_groups
from .manufacturing_ir import (
    IRRecord, ManufacturingIR, coarse_embedded_drawing_table_pages,
)
from .native_view_fallback import (
    native_view_layout_support, native_view_review_candidates,
)
from .source_page_render import apply_document_page_render


SECTION_NAME = re.compile(r"^([A-Z])[-－—–]\1(?:剖视图|剖面图|剖视|剖面)?$", re.IGNORECASE)
DIRECTION_NAME = re.compile(r"^([A-Z])向(局部视图|局部|视图)?$", re.IGNORECASE)
ORTHOGRAPHIC_NAME = re.compile(
    r"^(?:(?:正面|背面|内侧|外侧|左侧|右侧|前侧|后侧)(?:视(?:图)?)?"
    r"|(?:主|正|背|左|右|前|后|端面|端|俯|仰)视(?:图)?)$"
)
SCALE_NAME = re.compile(r"^\d{1,2}:\d{1,3}$")
ROMAN_DETAIL_NAME = re.compile(r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X)$")
DIAMETER = re.compile(r"[⌀ØΦφ∅]")
THREAD = re.compile(r"(?:^|\W)M\s*\d", re.IGNORECASE)
ANGLE = re.compile(r"\d+(?:\.\d+)?\s*[°º]")
RADIUS = re.compile(r"(?:^|\W)(?:SR|R)\s*\d", re.IGNORECASE)
MULTIPLE = re.compile(r"\d+\s*[×xX]\s*")
MIN_UNLABELED_VIEW_AREA_FRACTION = .005


def _source_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _box_area(box: list[float] | tuple[float, ...]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_center(box: list[float] | tuple[float, ...]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _intersection_fraction(box: list[float], view: list[float]) -> float:
    intersection = max(0.0, min(box[2], view[2]) - max(box[0], view[0])) * max(
        0.0, min(box[3], view[3]) - max(box[1], view[1])
    )
    return intersection / max(_box_area(box), 1.0)


def _normalize_label(text: str) -> str:
    value = re.sub(r"\\(?:Delta|triangle)\b", "△", text, flags=re.IGNORECASE)
    return re.sub(r"[\s$△Δ{}]", "", value).upper()


def _view_labels(document: DrawingDocument, page_number: int) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for entity in document.raw_ocr:
        name = _normalize_label(entity.raw_text or "")
        section = SECTION_NAME.fullmatch(name)
        direction = DIRECTION_NAME.fullmatch(name)
        if section:
            view_type = "section_view"
            name = f"{section.group(1)}-{section.group(1)}"
        elif direction:
            view_type = "direction_view"
            name = f"{direction.group(1)}向{direction.group(2) or ''}"
        elif ORTHOGRAPHIC_NAME.fullmatch(name):
            view_type = "orthographic_view"
        elif SCALE_NAME.fullmatch(name) or ROMAN_DETAIL_NAME.fullmatch(name):
            view_type = "detail_view"
        else:
            continue
        for location in entity.locations:
            if location.page_number != page_number or location.bbox is None:
                continue
            box = list(location.bbox)
            center = _box_center(box)
            duplicate = next((
                item for item in labels
                if item["name"] == name and math.dist(center, _box_center(item["box"])) <= 14
            ), None)
            if duplicate:
                duplicate["evidence_ids"].append(entity.id)
                duplicate["raw_result_refs"].extend(entity.raw_result_refs)
                continue
            labels.append({
                "name": name, "type": view_type, "box": box,
                "evidence_ids": [entity.id],
                "raw_result_refs": list(entity.raw_result_refs),
            })
    return labels


def _nearby_view_labels(box: list[float], labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    region_height = box[3] - box[1]
    return sorted((
        label for label in labels
        if box[0] <= _box_center(label["box"])[0] <= box[2]
        and box[1] - max(120, min(180, .25 * region_height)) <= _box_center(label["box"])[1]
        <= box[1] + max(75, min(160, .06 * region_height))
    ), key=lambda label: _box_center(label["box"])[0])


def _distinct_named_views(
    first: list[float], second: list[float], labels: list[dict[str, Any]],
) -> bool:
    first_names = {label["name"] for label in _nearby_view_labels(first, labels)}
    second_names = {label["name"] for label in _nearby_view_labels(second, labels)}
    return bool(first_names and second_names and first_names.isdisjoint(second_names))


def _layout_boxes(
    ir: ManufacturingIR, page_number: int, page_width: int,
    labels: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    labels = labels or []
    rows = [
        {"box": list(record.bbox), "source_ids": list(record.source_ids),
         "raw_result_refs": list(record.raw_result_refs), "ir_ids": [record.id],
         "_fragments": [list(record.bbox)]}
        for record in ir.views if record.page == page_number and record.bbox is not None
    ]
    rows.sort(key=lambda row: (row["box"][1], row["box"][0]))
    merged: list[dict[str, Any]] = []
    for row in rows:
        box = row["box"]
        partner = next((
            candidate for candidate in merged
            if any(_same_view_fragment(box, fragment)
                   for fragment in candidate["_fragments"])
            or (
                max(box[2] - box[0], candidate["box"][2] - candidate["box"][0]) < page_width * .25
                and .5 <= (box[2] - box[0]) / (candidate["box"][2] - candidate["box"][0]) <= 2
                and (min(box[2], candidate["box"][2]) - max(box[0], candidate["box"][0]))
                / min(box[2] - box[0], candidate["box"][2] - candidate["box"][0]) >= .8
                and 0 <= box[1] - candidate["box"][3] <= 150
                and not _distinct_named_views(box, candidate["box"], labels)
            )
        ), None)
        if partner is None:
            merged.append(row)
            continue
        previous = partner["box"]
        partner["box"] = [min(previous[0], box[0]), min(previous[1], box[1]),
                          max(previous[2], box[2]), max(previous[3], box[3])]
        for key in ("source_ids", "raw_result_refs", "ir_ids"):
            partner[key].extend(row[key])
        partner["_fragments"].append(box)
    for row in merged:
        del row["_fragments"]
    return merged


def _same_view_fragment(first: list[float], second: list[float]) -> bool:
    """Merge overlapping OCR layout tiles only when their scale is comparable."""
    first_width, first_height = first[2] - first[0], first[3] - first[1]
    second_width, second_height = second[2] - second[0], second[3] - second[1]
    width_ratio = min(first_width, second_width) / max(first_width, second_width)
    height_ratio = min(first_height, second_height) / max(first_height, second_height)
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    fraction = intersection / min(_box_area(first), _box_area(second))
    if max(width_ratio, height_ratio) >= .7 and fraction >= .2:
        return True
    smaller, larger = (first, second) if _box_area(first) <= _box_area(second) else (second, first)
    crosses_boundary = any((
        smaller[0] < larger[0], smaller[1] < larger[1],
        smaller[2] > larger[2], smaller[3] > larger[3],
    ))
    return min(width_ratio, height_ratio) >= .3 and fraction >= .6 and crosses_boundary


def _layout_review_reason(
    layout: dict[str, Any], layouts: list[dict[str, Any]],
    labels: list[dict[str, Any]], image_size: tuple[int, int],
) -> tuple[str, str] | None:
    box = layout["box"]
    if _nearby_view_labels(box, labels):
        return None
    if _box_area(box) < MIN_UNLABELED_VIEW_AREA_FRACTION * image_size[0] * image_size[1]:
        return ("small_layout", "小型无视图名版面区；可能是工程符号或局部细节，未作为独立视图")
    width = box[2] - box[0]
    peers = []
    for other in layouts:
        if other is layout:
            continue
        frame = other["box"]
        other_width, other_height = frame[2] - frame[0], frame[3] - frame[1]
        horizontal_overlap = max(0.0, min(box[2], frame[2]) - max(box[0], frame[0]))
        vertical_overlap = max(0.0, min(box[3], frame[3]) - max(box[1], frame[1]))
        if (other_width >= 1.2 * width
                and horizontal_overlap >= .25 * min(width, other_width)
                and vertical_overlap >= .5 * other_height):
            peers.append(frame)
    if any(
        first[3] <= second[1] or second[3] <= first[1]
        for index, first in enumerate(peers) for second in peers[index + 1:]
    ):
        return ("ambiguous_layout", "版面切片同时跨越上下独立视图区；无法唯一确定视图边界")
    return None


def _long_horizontal_pairs(
    box: list[float], strokes: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    width, height = box[2] - box[0], box[3] - box[1]
    lines = []
    for stroke in strokes:
        if len(stroke["points"]) != 2:
            continue
        start, end = stroke["points"]
        length = abs(end[0] - start[0])
        y = (start[1] + end[1]) / 2
        if (length < .36 * width or abs(end[1] - start[1]) > 1.5
                or not box[1] <= y <= box[3]):
            continue
        lines.append({"y": y, "x0": min(start[0], end[0]),
                      "x1": max(start[0], end[0]), "sources": stroke["sources"]})
    lines.sort(key=lambda line: line["y"])
    pairs = []
    for index, first in enumerate(lines):
        for second in lines[index + 1:]:
            distance = second["y"] - first["y"]
            if distance < 8 or distance > min(110, .27 * height):
                continue
            overlap = max(0.0, min(first["x1"], second["x1"]) - max(first["x0"], second["x0"]))
            first_length = first["x1"] - first["x0"]
            second_length = second["x1"] - second["x0"]
            if overlap / min(first_length, second_length) < .92:
                continue
            if abs(first_length - second_length) / max(first_length, second_length) > .08:
                continue
            pairs.append((first, second))
    pairs.sort(key=lambda pair: (pair[0]["y"], pair[1]["y"]))
    selected = []
    for pair in pairs:
        if selected and pair[0]["y"] <= selected[-1][1]["y"] + .12 * height:
            continue
        selected.append(pair)
    return selected


def _stroke_refs_in_box(strokes: list[dict[str, Any]], box: list[float], limit: int = 12) -> list[dict[str, Any]]:
    ranked = []
    for stroke in strokes:
        points = stroke["points"]
        xs, ys = [point[0] for point in points], [point[1] for point in points]
        if (min(xs) < box[0] or max(xs) > box[2] or min(ys) < box[1] or max(ys) > box[3]):
            continue
        length = sum(math.dist(first, second) for first, second in zip(points, points[1:]))
        if length >= 8:
            ranked.append((length, stroke))
    ranked.sort(key=lambda item: -item[0])
    return [ref for _, stroke in ranked[:limit] for ref in stroke["sources"][:1]]


def _tight_split_box(
    zone: list[float], label_box: list[float], strokes: list[dict[str, Any]],
) -> tuple[list[float], list[dict[str, Any]]]:
    samples = []
    for stroke in strokes:
        points = stroke["points"]
        xs, ys = [point[0] for point in points], [point[1] for point in points]
        if (min(xs) < zone[0] + 5 or max(xs) > zone[2] - 5
                or min(ys) < zone[1] + 15 or max(ys) > zone[3] - 12):
            continue
        length = sum(math.dist(first, second) for first, second in zip(points, points[1:]))
        if length >= 12:
            samples.append((stroke, xs, ys))
    if len(samples) < 4:
        return [zone[0], min(zone[1], label_box[1]), zone[2], zone[3]], []
    xs = [value for _, values, _ in samples for value in values]
    ys = [value for _, _, values in samples for value in values]
    box = [max(zone[0], min(xs) - 12), min(zone[1], label_box[1], min(ys) - 12),
           min(zone[2], max(xs) + 12), min(zone[3], max(ys) + 12)]
    refs = [ref for stroke, _, _ in sorted(samples, key=lambda row: -(
        max(row[1]) - min(row[1]) + max(row[2]) - min(row[2])
    ))[:12] for ref in stroke["sources"][:1]]
    return box, refs


def _detected_views(
    document: DrawingDocument, ir: ManufacturingIR, page_number: int,
    image_size: tuple[int, int], strokes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    width, _ = image_size
    labels = _view_labels(document, page_number)
    views = []
    layouts = _layout_boxes(ir, page_number, width, labels)
    for layout in layouts:
        box = layout["box"]
        region_width, region_height = box[2] - box[0], box[3] - box[1]
        nearby = _nearby_view_labels(box, labels)
        if _layout_review_reason(layout, layouts, labels, image_size):
            continue
        if len(nearby) >= 2 and region_width > width * .25:
            centers = [_box_center(label["box"])[0] for label in nearby]
            boundaries = [box[0], *[(left + right) / 2 for left, right in zip(centers, centers[1:])], box[2]]
            for index, label in enumerate(nearby):
                zone = [boundaries[index], box[1], boundaries[index + 1], box[3]]
                view_box, refs = _tight_split_box(zone, label["box"], strokes)
                views.append({
                    "page": str(page_number), "view_bbox": view_box,
                    "type": label["type"], "label": label["name"] if label["type"] != "detail_view" else f"局部视图 {label['name']}",
                    "confidence": .76 if refs else None,
                    "status": "candidate" if refs else "unresolved",
                    "confidence_kind": "uncalibrated_geometry_heuristic" if refs else None,
                    "evidence_ids": [*layout["source_ids"], *label["evidence_ids"]],
                    "raw_result_refs": [*layout["raw_result_refs"], *label["raw_result_refs"]],
                    "label_box": label["box"], "native_path_refs": refs,
                    "basis": "layout_label_and_native_strokes" if refs else "layout_label_partition",
                })
            continue
        pairs = _long_horizontal_pairs(box, strokes) if region_width > width * .55 else []
        if len(pairs) >= 2 and pairs[1][0]["y"] - pairs[0][1]["y"] > .22 * region_height:
            boundary = (pairs[0][1]["y"] + pairs[1][0]["y"]) / 2
            for index, (top, bottom) in enumerate((
                (box[1], boundary), (boundary, box[3]),
            )):
                pair = pairs[index]
                views.append({
                    "page": str(page_number), "view_bbox": [box[0], top, box[2], bottom],
                    "type": "orthographic_view",
                    "label": "投影视图候选",
                    "confidence": .73, "status": "candidate",
                    "confidence_kind": "uncalibrated_geometry_heuristic",
                    "evidence_ids": list(layout["source_ids"]),
                    "raw_result_refs": list(layout["raw_result_refs"]),
                    "label_box": None,
                    "native_path_refs": [ref for line in pair for ref in line["sources"]],
                    "basis": "layout_and_paired_native_contours",
                })
            continue
        best = nearby[0] if len(nearby) == 1 else None
        view_type = best["type"] if best else "orthographic_view" if region_width > width * .55 else "unknown"
        view_box = [box[0], min(box[1], max(0, best["box"][1] - 4))
                    if best else box[1], box[2], box[3]]
        views.append({
            "page": str(page_number), "view_bbox": view_box,
            "type": view_type,
            "label": (f"局部视图 {best['name']}" if best and view_type == "detail_view"
                      else best["name"] if best
                      else "投影视图候选" if view_type == "orthographic_view"
                      else "未命名视图候选"),
            "confidence": .79 if best and strokes else .55 if strokes else None,
            "confidence_kind": "uncalibrated_geometry_heuristic" if strokes else None,
            "status": "candidate" if strokes else "unresolved",
            "evidence_ids": [*layout["source_ids"], *(best["evidence_ids"] if best else [])],
            "raw_result_refs": [*layout["raw_result_refs"], *(best["raw_result_refs"] if best else [])],
            "label_box": best["box"] if best else None,
            "native_path_refs": _stroke_refs_in_box(strokes, box),
            "basis": "layout_label_and_native_strokes" if best and strokes else "layout_candidate",
        })
    row_height = max(100, image_size[1] * .09)
    views.sort(key=lambda view: (round(view["view_bbox"][1] / row_height),
                                 view["view_bbox"][0]))
    projection_number = 0
    for view in views:
        if view["type"] == "orthographic_view" and view["label"] == "投影视图候选":
            projection_number += 1
            view["label"] = f"投影视图候选 {projection_number}"
    for index, view in enumerate(views, start=1):
        view["id"] = "drawing_view_" + hashlib.sha256(
            repr((document.source_sha256, page_number, view["type"], view["label"],
                  view["view_bbox"], index)).encode()
        ).hexdigest()[:20]
        view["evidence_ids"] = list(dict.fromkeys(view["evidence_ids"]))
        view["raw_result_refs"] = list(dict.fromkeys(view["raw_result_refs"]))
    return views


def _coarse_table_native_views(
    document: DrawingDocument, annotations: list[dict[str, Any]],
    page_number: int, image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    coarse_table_page = page_number in coarse_embedded_drawing_table_pages(document)
    engineering = [row for row in annotations
                   if row.get("page") == str(page_number)
                   and row.get("category") in {
                       "dimension", "diameter", "hole", "radius", "angle", "tolerance",
                   } and isinstance(row.get("source_box"), list)]
    if not engineering:
        return []
    views = []
    for candidate in native_view_review_candidates(
        document, image_size, page_number=page_number,
    ):
        compact_sheet = candidate.get("basis") == "native_pdf_compact_sheet_strokes"
        if not coarse_table_page and not compact_sheet:
            continue
        if (candidate["support_path_count"] < 10
                or candidate["support_length_pdf_points"] < 400):
            continue
        box = candidate["source_box"]
        padding = max(60.0, min(120.0, (box[2] - box[0]) * .06))
        matched = [row for row in engineering if (
            box[0] - padding <= _box_center(row["source_box"])[0] <= box[2] + padding
            and box[1] - padding <= _box_center(row["source_box"])[1] <= box[3] + padding
        )]
        if not matched:
            continue
        views.append({
            "id": "drawing_view_" + hashlib.sha256(
                f"{document.source_sha256}:{page_number}:{candidate['id']}".encode()
            ).hexdigest()[:20],
            "page": str(page_number), "view_bbox": box,
            "type": "unknown", "label": f"原图视图待核实 {len(views) + 1}",
            "confidence": None, "confidence_kind": None,
            "status": "unresolved", "review_only": True,
            "route_hard_constraint_eligible": False,
            "evidence_ids": [row["id"] for row in matched],
            "raw_result_refs": list(dict.fromkeys(
                ref for row in matched
                for ref in row.get("provenance", {}).get("raw_result_refs", [])
            )),
            "label_box": None,
            "native_path_refs": candidate["native_path_refs"],
            "basis": ("native_pdf_compact_sheet_strokes_and_localized_measurements"
                      if compact_sheet else
                      "native_pdf_strokes_and_localized_measurements_in_coarse_table"),
            "source_candidate_id": candidate["id"],
        })
    return views


def _native_supported_small_views(
    document: DrawingDocument, ir: ManufacturingIR, page_number: int,
    image_size: tuple[int, int], existing_views: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = _view_labels(document, page_number)
    layouts = _layout_boxes(ir, page_number, image_size[0], labels)
    minimum_area = .002 * image_size[0] * image_size[1]
    small = [layout for layout in layouts
             if _box_area(layout["box"]) >= minimum_area
             and (review := _layout_review_reason(layout, layouts, labels, image_size))
             and review[0] == "small_layout"
             and max((_intersection_fraction(layout["box"], view["view_bbox"])
                      for view in existing_views), default=0.0) < .2]
    if not small:
        return []
    support = native_view_layout_support(
        document, image_size, [layout["box"] for layout in small],
        page_number=page_number,
    )
    views = []
    for layout, evidence in zip(small, support):
        if (evidence["cluster_coverage"] < .8
                or evidence["support_path_count"] < 20
                or evidence["support_length_pdf_points"] < 500):
            continue
        box = layout["box"]
        views.append({
            "id": "drawing_view_" + hashlib.sha256(
                repr((document.source_sha256, page_number, box,
                      "native_supported_small_layout")).encode()
            ).hexdigest()[:20],
            "page": str(page_number), "view_bbox": box,
            "type": "unknown", "label": f"小型视图待核实 {len(views) + 1}",
            "confidence": None, "confidence_kind": None,
            "status": "unresolved", "review_only": True,
            "route_hard_constraint_eligible": False,
            "evidence_ids": list(dict.fromkeys(layout["source_ids"])),
            "raw_result_refs": list(dict.fromkeys(layout["raw_result_refs"])),
            "label_box": None,
            "native_path_refs": evidence["native_path_refs"],
            "basis": "small_layout_and_native_pdf_path_support",
            "native_support": {
                "cluster_coverage": evidence["cluster_coverage"],
                "support_path_count": evidence["support_path_count"],
                "support_length_pdf_points": evidence["support_length_pdf_points"],
            },
        })
    return views


def _engineering_category(record: IRRecord | None, annotation: dict[str, Any]) -> str:
    text = annotation.get("label") or ""
    kind = record.kind.lower() if record else annotation.get("provenance", {}).get("field_kind", "")
    if kind == "gdt":
        return "gdt"
    if kind == "tolerance":
        return "tolerance"
    if kind == "surface_finish" or annotation["category"] == "roughness":
        return "roughness"
    if kind == "technical_notes":
        return "technical_note"
    if kind == "threads" or THREAD.search(text):
        return "thread"
    if kind in {"dimension", "measurement"}:
        if DIAMETER.search(text):
            return "hole" if MULTIPLE.search(text) or "孔" in text else "diameter"
        if RADIUS.search(text):
            return "radius"
        if ANGLE.search(text):
            return "angle"
        if re.search(r"±|\^\s*\{?[-+]?\d|\b公差\b", text):
            return "tolerance"
    return annotation["category"]


def _view_scope(
    box: list[float], views: list[dict[str, Any]], page_number: int = 1,
) -> tuple[str | None, list[dict[str, Any]]]:
    ranked = sorted((
        (fraction, view) for view in views if view["page"] == str(page_number)
        if (fraction := _intersection_fraction(box, view["view_bbox"])) > .15
    ), key=lambda row: (-row[0], _box_area(row[1]["view_bbox"])))
    candidates = [{
        "kind": "view", "id": view["id"], "label": view["label"],
        "confidence": round(fraction * view["confidence"], 3)
        if view["confidence"] is not None else None,
        "confidence_kind": view["confidence_kind"],
        "status": view["status"],
        "reason": "标注文字框与视图区的几何交叠；未证明所属零件",
    } for fraction, view in ranked[:3]]
    if len(ranked) == 1 and ranked[0][0] >= .9 and ranked[0][1]["status"] == "candidate":
        return ranked[0][1]["id"], candidates
    if (len(ranked) > 1 and ranked[0][0] >= .9 and ranked[1][0] < .35
            and ranked[0][1]["status"] == "candidate"):
        return ranked[0][1]["id"], candidates
    return None, candidates


@lru_cache(maxsize=3)
def _native_evidence(
    document_json: str, ir_json: str, page_number: int = 1,
) -> dict[str, Any]:
    document = DrawingDocument.model_validate_json(document_json)
    ir = ManufacturingIR.model_validate_json(ir_json)
    page = next((item for item in document.pages if item.page_number == page_number), None)
    if page is None:
        return {"status": "missing_page", "strokes": [], "marks": [],
                "topology": {}, "entities": {}, "contacts": {}}
    source = Path(document.source_path)
    if not source.is_file() or _source_digest(source) != document.source_sha256:
        raise ValueError("Source PDF hash mismatch")
    try:
        with pymupdf.open(source) as pdf:
            source_page = pdf[page_number - 1]
            apply_document_page_render(source_page, page)
            drawings = source_page.get_drawings()
    except (pymupdf.FileDataError, pymupdf.EmptyFileError):
        return {"status": "invalid_pdf", "strokes": [], "marks": [],
                "topology": {}, "entities": {}, "contacts": {}}
    if not drawings:
        return {"status": "no_native_geometry", "strokes": [], "marks": [],
                "topology": {}, "entities": {}, "contacts": {}}
    transform = ~pymupdf.Matrix(*page.pixel_to_pdf_matrix)
    # Keep original indices: every returned reference must address get_drawings().
    indexed_paths = [(index, path) for index, path in enumerate(drawings)
                     if path.get("color") is not None and max(path["color"]) <= .5]
    strokes, _ = extract_strokes(drawings, transform, [0, 0, page.width, page.height],
                                 set(range(len(drawings))) - {index for index, _ in indexed_paths})
    if not strokes:
        return {"status": "no_native_geometry", "strokes": [], "marks": [],
                "topology": {}, "entities": {}, "contacts": {}}
    filled_paths = {index for index, path in enumerate(drawings)
                    if path.get("fill") is not None or "f" in str(path.get("type", ""))}
    marks = extract_dark_fill_marks(drawings, transform)
    for mark in marks:
        mark["source_refs"] = [
            {"path_index": mark["path_index"], "item_index": index,
             "native_kind": item[0]}
            for index, item in enumerate(drawings[mark["path_index"]]["items"])
        ]
    topology = build_dimension_topology(document, ir)
    entities = build_dimension_entity_evidence(document, topology)
    stroke_bounds = []
    for stroke in strokes:
        xs = [point[0] for point in stroke["points"]]
        ys = [point[1] for point in stroke["points"]]
        stroke_bounds.append((min(xs), min(ys), max(xs), max(ys), stroke))
    contacts = {}
    for chain in topology["chains"]:
        rows = []
        for index, extension in enumerate(chain["extension_lines"]):
            xs = [point[0] for point in (extension["first"], extension["second"])]
            ys = [point[1] for point in (extension["first"], extension["second"])]
            window = [min(xs) - 8, min(ys) - 8, max(xs) + 8, max(ys) + 8]
            nearby_strokes = [stroke for x0, y0, x1, y1, stroke in stroke_bounds
                              if x0 <= window[2] and x1 >= window[0]
                              and y0 <= window[3] and y1 >= window[1]
                              and all(ref["path_index"] not in filled_paths
                                      for ref in stroke["sources"])]
            ray_anchor, targets = find_first_contact_targets(
                extension, chain["dimension_line"], nearby_strokes, limit=4,
            )
            candidates = []
            for target in targets:
                points = target["points"]
                x_values = [point[0] for point in points]
                y_values = [point[1] for point in points]
                candidates.append({
                    "id": target["stroke_id"],
                    "kind": "native_stroke_contact_candidate",
                    "bbox": [max(0, min(x_values) - .5), max(0, min(y_values) - .5),
                             min(page.width, max(x_values) + .5),
                             min(page.height, max(y_values) + .5)],
                    "stroke_points": points,
                    "contact_point": target["contact_point"],
                    "forward_distance_pixels": target["forward_distance_pixels"],
                    "transverse_distance_pixels": target["transverse_distance_pixels"],
                    "source_refs": target["source_refs"],
                    "physical_object_verified": False,
                })
            rows.append({
                "extension_index": index, "ray_anchor": ray_anchor,
                "extension_source_refs": extension["sources"],
                "contact_status": "candidate" if candidates else "unresolved",
                "object_candidates": candidates,
            })
        contacts[chain["id"]] = rows
    return {"status": "available", "strokes": strokes, "marks": marks,
            "topology": topology, "entities": entities, "contacts": contacts,
            "filled_paths": filled_paths}


def _text_centered_on_line(box: list[float], line: dict[str, Any]) -> bool:
    text_center = _box_center(box)
    first, second = line["first"], line["second"]
    axis = 0 if line["orientation"] == "horizontal" else 1
    midpoint = (first[axis] + second[axis]) / 2
    return abs(text_center[axis] - midpoint) <= .2 * line["length"]


def _centered_line_candidates(box: list[float], native: dict[str, Any]) -> list[dict[str, Any]]:
    width, height = box[2] - box[0], box[3] - box[1]
    center = _box_center(box)
    ranked = []
    for stroke in native.get("strokes", []):
        if len(stroke["points"]) != 2 or any(
            ref["path_index"] in native.get("filled_paths", set())
            for ref in stroke["sources"]
        ):
            continue
        first, second = stroke["points"]
        dx, dy = second[0] - first[0], second[1] - first[1]
        length = math.hypot(dx, dy)
        if abs(dy) <= 1.5 and length >= max(24, 2 * width):
            orientation, axis = "horizontal", 0
            distance = min(abs(first[1] - box[1]), abs(first[1] - box[3]))
            span = (min(first[0], second[0]), max(first[0], second[0]))
            maximum_distance = max(30, 3 * height)
        elif abs(dx) <= 1.5 and length >= max(24, 2 * height):
            orientation, axis = "vertical", 1
            distance = min(abs(first[0] - box[0]), abs(first[0] - box[2]))
            span = (min(first[1], second[1]), max(first[1], second[1]))
            maximum_distance = max(30, 2 * width)
        else:
            continue
        midpoint_distance = abs(center[axis] - (span[0] + span[1]) / 2)
        if (distance > maximum_distance or not span[0] <= center[axis] <= span[1]
                or midpoint_distance > .2 * length):
            continue
        ranked.append((distance, midpoint_distance, {
            "id": stroke["id"], "orientation": orientation,
            "first": first, "second": second, "length": round(length, 4),
            "source_refs": stroke["sources"],
            "text_distance_pixels": round(distance, 4),
            "midpoint_distance_pixels": round(midpoint_distance, 4),
            "status": "unresolved",
            "reason": "文字与线段位置对应；双端延长线、箭头及对象关系未闭环",
        }))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]["id"]))
    return [row for _, _, row in ranked[:3]]


def _geometry_evidence(
    annotation_id: str, box: list[float], native: dict[str, Any],
) -> dict[str, Any]:
    chains = [row for row in native.get("topology", {}).get("chains", [])
              if row["measurement_id"] == annotation_id]
    rejected = [row for row in chains
                if not _text_centered_on_line(box, row["dimension_line"])]
    chains = [row for row in chains if row not in rejected]
    entities = [row for row in native.get("entities", {}).get("candidates", [])
                if row["measurement_id"] == annotation_id and row["retained"]]
    if not chains:
        return {"status": "unresolved", "object_status": "unresolved",
                "arrow_status": "unresolved", "candidates": [],
                "rejected_topology_ids": [row["id"] for row in rejected],
                "line_only_candidates": _centered_line_candidates(box, native)
                if rejected else []}
    rows = []
    by_topology = {row["topology_chain_id"]: row for row in entities}
    for chain in chains:
        entity = by_topology.get(chain["id"])
        mark_refs = {mark["path_index"]: mark["source_refs"]
                     for mark in native.get("marks", [])}
        rows.append({
            "id": chain["id"], "view_layout_id": chain["view_id"],
            "dimension_line": chain["dimension_line"],
            "extension_lines": chain["extension_lines"],
            "arrow_candidates": [
                {**mark, "source_refs": mark_refs.get(mark["path_index"], [])}
                for mark in entity["endpoint_fill_marks"]
            ] if entity else [],
            "arrow_status": "candidate" if entity else "unresolved",
            "source_refs": chain["source_refs"],
            "extension_contacts": native.get("contacts", {}).get(chain["id"], []),
            "text_distance_pixels": chain["text_distance_pixels"],
            "status": "candidate", "object_status": "unresolved",
        })
    return {"status": "candidate", "object_status": "unresolved",
            "arrow_status": "candidate" if entities else "unresolved",
            "candidates": rows}


def _geometry_cues(
    views: list[dict[str, Any]], native: dict[str, Any], source_sha256: str,
) -> dict[str, Any]:
    """Expose bounded native patterns without naming a physical feature."""
    if native["status"] != "available":
        return {"status": "unresolved", "centerline_status": "unresolved",
                "native_dash_metadata_count": 0, "segmented_line_patterns": [],
                "parallel_hatch_candidates": [], "contour_candidates": []}

    filled_paths = native["filled_paths"]
    measurement_strokes = {
        line["stroke_id"]
        for chain in native["topology"]["chains"]
        for line in [chain["dimension_line"], *chain["extension_lines"]]
    }
    solid_strokes = [
        stroke for stroke in native["strokes"]
        if stroke["id"] not in measurement_strokes
        and all(ref["path_index"] not in filled_paths for ref in stroke["sources"])
    ]
    view_boxes = {view["id"]: view["view_bbox"] for view in views}
    view_pages = {view["id"]: view["page"] for view in views}

    def cue(kind: str, view_id: str, strokes: list[dict[str, Any]],
            *, detail: dict[str, Any] | None = None) -> dict[str, Any]:
        points = [point for stroke in strokes for point in stroke["points"]]
        refs = [ref for stroke in strokes for ref in stroke["sources"]]
        identity = (source_sha256, kind, view_id, sorted(stroke["id"] for stroke in strokes))
        frame = view_boxes[view_id]
        return {
            "id": "geometry_cue_" + hashlib.sha256(repr(identity).encode()).hexdigest()[:18],
            "kind": kind, "view_id": view_id, "page": view_pages[view_id],
            "bbox": [max(frame[0], min(point[0] for point in points) - 1),
                     max(frame[1], min(point[1] for point in points) - 1),
                     min(frame[2], max(point[0] for point in points) + 1),
                     min(frame[3], max(point[1] for point in points) + 1)],
            "stroke_count": len(strokes), "source_refs": refs[:12],
            "status": "candidate", "physical_feature_verified": False,
            "part_id": None, "feature_id": None, **(detail or {}),
        }

    segmented: list[dict[str, Any]] = []
    hatches: list[dict[str, Any]] = []
    contours: list[dict[str, Any]] = []
    for view in views:
        box = view["view_bbox"]
        local = [
            stroke for stroke in solid_strokes
            if all(box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
                   for point in stroke["points"])
        ]
        # Separate CAD stroke segments can form a dashed-looking line even
        # when the PDF path style itself contains no dash metadata.
        axis_groups: dict[tuple[str, int], list[tuple[float, float, dict[str, Any]]]] = defaultdict(list)
        for stroke in local:
            if len(stroke["points"]) != 2:
                continue
            first, second = stroke["points"]
            dx, dy = second[0] - first[0], second[1] - first[1]
            length = math.hypot(dx, dy)
            if not 3 <= length <= 100:
                continue
            if abs(dy) <= .7:
                orientation, fixed, values = "horizontal", (first[1] + second[1]) / 2, (first[0], second[0])
            elif abs(dx) <= .7:
                orientation, fixed, values = "vertical", (first[0] + second[0]) / 2, (first[1], second[1])
            else:
                continue
            axis_groups[(orientation, round(fixed / 1.5))].append((min(values), max(values), stroke))
        view_runs = []
        for (orientation, _), intervals in axis_groups.items():
            intervals.sort(key=lambda row: row[0])
            current: list[tuple[float, float, dict[str, Any]]] = []
            for interval in intervals:
                gap = interval[0] - current[-1][1] if current else math.inf
                if current and not .75 <= gap <= 35:
                    span = current[-1][1] - current[0][0]
                    threshold = max(80, .18 * (box[2] - box[0] if orientation == "horizontal"
                                                 else box[3] - box[1]))
                    if len(current) >= 4 and span >= threshold:
                        view_runs.append((span, orientation, [row[2] for row in current]))
                    current = []
                current.append(interval)
            if current:
                span = current[-1][1] - current[0][0]
                threshold = max(80, .18 * (box[2] - box[0] if orientation == "horizontal"
                                             else box[3] - box[1]))
                if len(current) >= 4 and span >= threshold:
                    view_runs.append((span, orientation, [row[2] for row in current]))
        view_runs.sort(key=lambda row: (-row[0], row[1]))
        segmented.extend(cue("segmented_collinear_pattern", view["id"], strokes,
                             detail={"orientation": orientation,
                                     "centerline_verified": False})
                         for _, orientation, strokes in view_runs[:2])

        short = [stroke for stroke in local if len(stroke["points"]) == 2
                 and 3 <= math.dist(*stroke["points"]) <= 100]
        repeated = dense_parallel_ids(short)
        bins: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for stroke in short:
            if stroke["id"] not in repeated:
                continue
            first, second = stroke["points"]
            midpoint = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
            angle = math.degrees(math.atan2(second[1] - first[1],
                                            second[0] - first[0])) % 180
            if not (25 <= angle <= 65 or 115 <= angle <= 155):
                continue
            angle_bucket = round(angle / 15)
            bins[(int(midpoint[0] // 70), int(midpoint[1] // 70), angle_bucket)].append(stroke)
        ranked_hatches = sorted((
            rows for rows in bins.values() if len(rows) >= 5
        ), key=lambda rows: (-len(rows), rows[0]["id"]))
        hatches.extend(cue("parallel_hatch_pattern", view["id"], rows,
                           detail={"hatching_verified": False})
                       for rows in ranked_hatches[:2])

        connected = []
        for group in stroke_groups(local, tolerance=1.5):
            if len(group) < 3:
                continue
            points = [point for stroke in group for point in stroke["points"]]
            xs, ys = [point[0] for point in points], [point[1] for point in points]
            span = max(max(xs) - min(xs), max(ys) - min(ys))
            if span >= max(30, .12 * min(box[2] - box[0], box[3] - box[1])):
                connected.append((len(group), span, group))
        connected.sort(key=lambda row: (-row[0], -row[1]))
        contours.extend(cue("connected_contour_candidate", view["id"], group,
                            detail={"closed_contour_verified": False})
                        for _, _, group in connected[:2])

    return {
        "status": "candidate", "source": "original_pdf_native_paths",
        "centerline_status": "unresolved",
        "centerline_reason": "原 PDF 路径未提供虚线样式；断续共线模式不足以确认中心线",
        "native_dash_metadata_count": sum(
            stroke.get("dashes") not in {None, "[] 0"} for stroke in native["strokes"]
        ),
        "segmented_line_patterns": segmented,
        "parallel_hatch_candidates": hatches,
        "contour_candidates": contours,
    }


def _section_relations(
    document: DrawingDocument, views: list[dict[str, Any]], marks: list[dict[str, Any]],
    page_number: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    page = next((item for item in document.pages
                 if item.page_number == page_number), None)
    if page is None:
        return [], []
    targets = {view["label"][0]: view for view in views
               if view["type"] in {"section_view", "direction_view"}}
    grouped: list[dict[str, Any]] = []
    for entity in document.raw_ocr:
        letter = _normalize_label(entity.raw_text or "")
        if len(letter) != 1 or letter not in targets:
            continue
        for location in entity.locations:
            if location.page_number != page_number or location.bbox is None:
                continue
            box = list(location.bbox)
            center = _box_center(box)
            if center[0] > page.width * .95:
                continue
            row = next((candidate for candidate in grouped
                        if candidate["letter"] == letter
                        and math.dist(_box_center(candidate["box"]), center) <= 14), None)
            if row is None:
                row = {"letter": letter, "box": box, "ids": [], "refs": []}
                grouped.append(row)
            row["ids"].append(entity.id)
            row["refs"].extend(entity.raw_result_refs)
    relations = []
    markers = []

    def source_view_candidates(box: list[float]) -> list[dict[str, Any]]:
        center = _box_center(box)
        distances = []
        for view in views:
            if view["page"] != str(page_number):
                continue
            frame = view["view_bbox"]
            distance = math.hypot(max(frame[0] - center[0], 0, center[0] - frame[2]),
                                  max(frame[1] - center[1], 0, center[1] - frame[3]))
            if distance <= 45:
                distances.append({
                    "id": view["id"], "label": view["label"],
                    "distance_pixels": round(distance, 3),
                    "confidence": round(max(0, 1 - distance / 60) *
                                        (view["confidence"] or .5), 3),
                    "confidence_kind": "uncalibrated_geometry_heuristic",
                    "status": "candidate",
                })
        distances.sort(key=lambda row: (row["distance_pixels"], row["id"]))
        return distances[:3]

    for row in grouped:
        box = row["box"]
        center = _box_center(box)
        nearby = sorted((
            (math.hypot(max(mark["bbox"][0] - center[0], 0, center[0] - mark["bbox"][2]),
                        max(mark["bbox"][1] - center[1], 0, center[1] - mark["bbox"][3])), mark)
            for mark in marks
            if max(mark["bbox"][2] - mark["bbox"][0],
                   mark["bbox"][3] - mark["bbox"][1]) <= 30
        ), key=lambda item: item[0])
        arrow = [nearby[0][1]] if nearby and nearby[0][0] <= 55 else []
        if not arrow:
            continue
        target = targets[row["letter"]]
        marker_identity = ((document.source_sha256, row["letter"], box)
                           if page_number == 1 else
                           (document.source_sha256, page_number, row["letter"], box))
        marker_id = "section_marker_" + hashlib.sha256(
            repr(marker_identity).encode()
        ).hexdigest()[:18]
        source_details = source_view_candidates(box)
        source_views = [candidate["id"] for candidate in source_details]
        marker = {
            "id": marker_id, "label": row["letter"], "page": str(page_number),
            "source_box": box, "target_view_id": target["id"],
            "source_view_candidates": source_views,
            "source_view_candidate_details": source_details,
            "evidence_ids": list(dict.fromkeys(row["ids"])),
            "raw_result_refs": list(dict.fromkeys(row["refs"])),
            "native_path_refs": [ref for mark in arrow for ref in mark["source_refs"]],
            "arrow_boxes": [mark["bbox"] for mark in arrow],
            "status": "candidate",
        }
        markers.append(marker)
        relations.append({
            "id": "drawing_relation_" + hashlib.sha256(
                f"{marker_id}:{target['id']}".encode()
            ).hexdigest()[:18],
            "type": "section_marker_to_view" if target["type"] == "section_view" else "direction_marker_to_view",
            "source_id": marker_id, "target_id": target["id"],
            "status": "candidate", "confidence": .55,
            "confidence_kind": "uncalibrated_geometry_heuristic",
            "evidence_ids": marker["evidence_ids"],
            "native_path_refs": marker["native_path_refs"],
            "source_view_candidates": source_views,
            "source_view_candidate_details": source_details,
            "source_label": f"{row['letter']} 剖切/方向标记候选",
            "target_label": target["label"],
            "source_box": box, "target_box": target["view_bbox"],
            "reason": "同字母标识与原 PDF 填充箭头相邻；剖切线、方向和轮廓对应尚未闭环",
        })
    return relations, markers


def build_view_scope_ir(
    document: DrawingDocument, ir: ManufacturingIR,
    annotations: list[dict[str, Any]],
    *, page_number: int = 1,
) -> dict[str, Any]:
    """Build a display IR from current OCR and the original, hash-checked PDF."""
    if page_number < 1:
        raise ValueError("Drawing IR page number must be positive")
    if document.source_sha256 != ir.source_sha256:
        raise ValueError("Drawing IR source hashes disagree")
    source = Path(document.source_path)
    if source.is_file() and _source_digest(source) != document.source_sha256:
        raise ValueError("Drawing IR source PDF hash mismatch")
    page = next((row for row in document.pages if row.page_number == page_number), None)
    coverage = next((row for row in ir.page_coverage
                     if row.get("page") == page_number), None)
    if page is None and coverage is None:
        raise ValueError("Drawing IR page is absent from current source")
    size = (page.width, page.height) if page else (
        int(coverage["width"]), int(coverage["height"])
    )
    native = {"status": "source_unavailable", "strokes": [], "marks": [],
              "topology": {}, "entities": {}, "contacts": {}}
    if page and source.is_file() and source.suffix.lower() == ".pdf":
        minimal_document = DrawingDocument(
            id=document.id, source_path=document.source_path,
            source_sha256=document.source_sha256, pages=[page],
        )
        def inside_page(record: IRRecord) -> bool:
            box = record.bbox
            return (record.page == page_number and box is not None
                    and 0 <= box[0] < box[2] <= page.width
                    and 0 <= box[1] < box[3] <= page.height)

        minimal_ir = ManufacturingIR(
            source_path=ir.source_path, source_sha256=ir.source_sha256,
            drawing_document_id=ir.drawing_document_id,
            page_coverage=[row for row in ir.page_coverage
                           if row.get("page") == page_number],
            views=[record for record in ir.views if inside_page(record)],
            measurements=[record for record in ir.measurements if inside_page(record)],
        )
        native = _native_evidence(
            minimal_document.model_dump_json(), minimal_ir.model_dump_json(), page_number,
        )
    views = _detected_views(document, ir, page_number, size, native["strokes"])
    views, continuity_reviews = reconcile_horizontal_splits(
        views, native["strokes"], native.get("filled_paths", set()),
        document.source_sha256,
    )
    if not views and native["status"] == "available":
        views = _coarse_table_native_views(document, annotations, page_number, size)
    if native["status"] == "available":
        views.extend(_native_supported_small_views(
            document, ir, page_number, size, views,
        ))
    view_gap_review = (build_view_gap_review(document, views, page_number=page_number)
                       if native["status"] == "available" else {
                           "schema_version": "view_gap_review.v1", "status": "not_run",
                           "reason": "no_native_geometry", "thin_proposals": [],
                           "merged_proposals": [], "boundary_proposals": [],
                       })
    merged_view_ids = {row["view_id"] for row in [
        *view_gap_review["merged_proposals"],
        *view_gap_review["boundary_proposals"],
    ]}
    for view in views:
        if view["id"] in merged_view_ids:
            view["status"] = "unresolved"
            view["confidence"] = None
            view["confidence_kind"] = None
    layout_labels = _view_labels(document, page_number)
    layouts = _layout_boxes(ir, page_number, size[0], layout_labels)
    review_layouts = []
    for layout in layouts:
        review = _layout_review_reason(layout, layouts, layout_labels, size)
        if review is not None:
            review_layouts.append((layout, *review))
    geometry_cues = _geometry_cues(views, native, document.source_sha256)
    record_by_id = {record.id: record for record in [
        *ir.metadata, *ir.features, *ir.measurements, *ir.requirements,
    ]}
    scoped = []
    unresolved = [*continuity_reviews, *[{
        "id": f"review_{view['id']}", "kind": "native_view_scope_unresolved",
        "reason": "原图线条与局部尺寸支持视图候选；视图类型和零件归属尚未验证",
        "source_box": view["view_bbox"], "view_id": view["id"],
        "native_path_refs": view["native_path_refs"],
        "evidence_ids": view["evidence_ids"],
        "candidates": [],
    } for view in views if view.get("review_only")],
                  *view_gap_review["thin_proposals"],
                  *view_gap_review["merged_proposals"],
                  *view_gap_review["boundary_proposals"], *[{
        "id": "small_layout_" + hashlib.sha256(
            repr((document.source_sha256, layout["box"])
                 if page_number == 1 else
                 (document.source_sha256, page_number, layout["box"])).encode()
        ).hexdigest()[:18],
        "label": "工程符号候选" if review_kind == "small_layout" else "跨视图区块",
        "kind": review_kind,
        "reason": reason,
        "source_box": layout["box"],
        "evidence_ids": layout["source_ids"],
        "candidates": [],
    } for layout, review_kind, reason in review_layouts]]
    for annotation in annotations:
        if annotation.get("page") is not None and str(annotation["page"]) != str(page_number):
            continue
        box = annotation["source_box"]
        view_id, candidates = _view_scope(box, views, page_number)
        record = record_by_id.get(annotation["id"])
        category = _engineering_category(record, annotation)
        is_dimension_record = record is not None and record.kind in {"dimension", "measurement"}
        scope = {
            "status": "candidate" if view_id else "unresolved",
            "view_id": view_id, "part_id": None, "feature_id": None,
            "view_status": "candidate" if view_id else "unresolved",
            "part_status": "unresolved", "feature_status": "unresolved",
            "candidates": candidates,
        }
        scoped.append({
            **annotation,
            "category": category,
            "view_id": view_id, "scope": scope,
            "geometry_evidence": _geometry_evidence(annotation["id"], box, native)
            if is_dimension_record and category in {
                "dimension", "diameter", "hole", "radius", "angle", "tolerance",
            } else {"status": "unresolved", "object_status": "unresolved",
                    "arrow_status": "unresolved", "candidates": []},
        })
        unresolved.append({
            "id": f"scope_{annotation['id']}", "annotation_id": annotation["id"],
            "reason": "零件与几何特征归属尚无完整的标注线至对象证据链"
            if view_id else "视图、零件与几何特征归属尚无唯一证据",
            "candidates": candidates,
        })
    relations, markers = _section_relations(
        document, views, native["marks"], page_number,
    ) if page else ([], [])
    detail_views = [view for view in views if view["type"] == "detail_view"]
    if detail_views:
        unresolved.append({
            "id": "local_detail_reference",
            "reason": "局部索引到局部视图尚无完整矢量引出线与轮廓对应证据",
            "candidates": [{"kind": "view", "id": view["id"], "label": view["label"],
                            "confidence": view["confidence"],
                            "reason": "局部视图已定位；源索引未验证"}
                           for view in detail_views],
        })
    return {
        "schema_version": "drawing_ir.v1", "source_sha256": document.source_sha256,
        "page": str(page_number),
        "coordinate_space": "rendered_page_pixels", "geometry_status": native["status"],
        "views": views, "annotations": scoped, "relations": relations,
        "section_markers": markers, "unresolved": unresolved,
        "geometry_cues": geometry_cues, "view_gap_review": view_gap_review,
        "metrics": {
            "view_count": len(views),
            "native_review_view_count": sum(bool(view.get("review_only")) for view in views),
            "small_layout_regions_for_review": sum(
                review_kind == "small_layout" for _, review_kind, _ in review_layouts
            ),
            "ambiguous_layout_regions_for_review": sum(
                review_kind == "ambiguous_layout" for _, review_kind, _ in review_layouts
            ),
            "view_split_conflicts_for_review": len(continuity_reviews),
            "uncovered_thin_regions_for_review": len(view_gap_review["thin_proposals"]),
            "merged_view_regions_for_review": len(view_gap_review["merged_proposals"]),
            "broad_view_boundaries_for_review": len(view_gap_review["boundary_proposals"]),
            "annotations_total": len(scoped),
            "annotations_with_view_candidate": sum(row["view_id"] is not None for row in scoped),
            "annotations_with_dimension_line_candidate": sum(
                row["geometry_evidence"]["status"] == "candidate" for row in scoped
            ),
            "verified_part_bindings": 0,
            "local_detail_relations_supported": 0,
            "segmented_line_pattern_count": len(geometry_cues["segmented_line_patterns"]),
            "parallel_hatch_candidate_count": len(geometry_cues["parallel_hatch_candidates"]),
            "contour_candidate_count": len(geometry_cues["contour_candidates"]),
        },
    }


__all__ = ["build_view_scope_ir"]
