"""Structure detected part-table cells without treating detection as recognition."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from src.tools.drawing_extraction.document_tables import HEADER_ALIASES, extract_cells


PART_FIELDS = ("item_no", "part_code", "name", "quantity", "material", "spec", "remark", "unit")
FIELD_NAMES = {
    "item_number": "item_no", "part_number": "part_code", "part_name": "name",
    "specification": "spec", **{key: key for key in PART_FIELDS},
}
FIELD_ALIASES = {
    "".join(alias.split()).casefold(): FIELD_NAMES[field]
    for field, aliases in HEADER_ALIASES.items() for alias in aliases
}
FIELD_ALIASES.update({"unit": "unit", "units": "unit", "uom": "unit", "单位": "unit"})
REVIEW_STATES = {"unreadable", "conflicted", "unmapped"}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _score(value: Any) -> float | None:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or not 0 <= value <= 1):
        return None
    return float(value)


def _box(value: Any, image_size: Sequence[float]) -> list[float] | None:
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(not isinstance(number, (int, float)) or isinstance(number, bool)
                   or not math.isfinite(number) for number in value)):
        return None
    x0, y0, x1, y1 = map(float, value)
    if not (0 <= x0 < x1 <= image_size[0] and 0 <= y0 < y1 <= image_size[1]):
        return None
    return [x0, y0, x1, y1]


def _union(boxes: Sequence[Sequence[float]]) -> list[float] | None:
    if not boxes:
        return None
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def _inside(box: Sequence[float], scope: Sequence[float]) -> bool:
    return (scope[0] <= box[0] < box[2] <= scope[2]
            and scope[1] <= box[1] < box[3] <= scope[3])


def _overlap(first: Sequence[float], second: Sequence[float]) -> float:
    area = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return area / max(1e-9, min((first[2] - first[0]) * (first[3] - first[1]),
                               (second[2] - second[0]) * (second[3] - second[1])))


def _field(column: Mapping[str, Any], index: int) -> str:
    key = _text(column.get("key"))
    if key in FIELD_NAMES:
        return FIELD_NAMES[key]
    alias = "".join(_text(column.get("label") or key).split()).casefold()
    return FIELD_ALIASES.get(alias, key or f"column_{index}")


def _comparable(value: str, field: str) -> str:
    # Identifier spacing can differ between a cell and its OCR fragments.
    return "".join(value.split()) if field in {"item_no", "part_code"} else " ".join(value.split())


def _source_ids(observations: Sequence[Mapping[str, Any]], cell: Mapping[str, Any]) -> list[str]:
    values = [_text(observation.get("source_id")) for observation in observations]
    values.extend(_text(value) for value in cell.get("evidence_ids") or [])
    values.append(_text(cell.get("source_id")))
    return list(dict.fromkeys(value for value in values if value))


def _raw_refs(observations: Sequence[Mapping[str, Any]], cell: Mapping[str, Any]) -> list[str]:
    values = [_text(observation.get("raw_ref")) for observation in observations]
    values.extend(_text(value) for value in cell.get("raw_result_refs") or [])
    values.append(_text(cell.get("raw_ref")))
    return list(dict.fromkeys(value for value in values if value))


def _compose_readings(
    observations: Sequence[Mapping[str, Any]], field: str, image_size: Sequence[float],
) -> tuple[list[dict[str, Any]], bool]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    readings = []
    contradictory_overlap = False
    for index, observation in enumerate(observations):
        if (not _text(observation.get("text"))
                or observation.get("response_status") in {"failed", "partial", "incomplete"}):
            continue
        box = _box(observation.get("bbox"), image_size)
        group = _text(observation.get("reading_id") or observation.get("pass_id"))
        if not group and box is None:
            group = f"unpositioned_{index}"
        groups[group or "positioned_observations"].append(observation)
    for group_id, records in groups.items():
        kept = []
        group_has_conflict = False
        for observation in records:
            box = _box(observation.get("bbox"), image_size)
            duplicate = False
            for previous in kept:
                previous_box = _box(previous.get("bbox"), image_size)
                if box is None or previous_box is None or _overlap(box, previous_box) < 0.65:
                    continue
                if _comparable(_text(previous.get("text")), field) == _comparable(
                        _text(observation.get("text")), field):
                    duplicate = True
                else:
                    contradictory_overlap = True
                    group_has_conflict = True
            if not duplicate:
                kept.append(observation)
        if group_has_conflict:
            for observation in kept:
                readings.append({"text": _text(observation.get("text")),
                                 "pass_id": group_id, "observations": [observation]})
            continue
        positioned = [record for record in kept if _box(record.get("bbox"), image_size)]
        if len(positioned) != len(kept):
            readings.extend({"text": _text(record.get("text")), "pass_id": group_id,
                             "observations": [record]} for record in kept)
            continue
        lines: list[list[Mapping[str, Any]]] = []
        for record in sorted(positioned, key=lambda value: (
                (_box(value["bbox"], image_size)[1] + _box(value["bbox"], image_size)[3]) / 2,
                _box(value["bbox"], image_size)[0])):
            box = _box(record["bbox"], image_size)
            assert box is not None
            center = (box[1] + box[3]) / 2
            line = next((line for line in lines if abs(center - statistics.median(
                (_box(item["bbox"], image_size)[1] + _box(item["bbox"], image_size)[3]) / 2
                for item in line)) <= 0.45 * (box[3] - box[1])), None)
            if line is None:
                lines.append([record])
            else:
                line.append(record)
        text = "\n".join(" ".join(_text(item.get("text")) for item in sorted(
            line, key=lambda value: _box(value["bbox"], image_size)[0])) for line in lines)
        readings.append({"text": text, "pass_id": group_id, "observations": kept})
    return readings, contradictory_overlap


def _cell_evidence(
    cell: Mapping[str, Any] | None, *, field: str, source_box: list[float] | None,
    table_bbox: Sequence[float], image_size: Sequence[float],
) -> dict[str, Any]:
    if cell is None:
        return {"value": None, "raw_text": "", "status": "unmapped", "source_box": source_box,
                "confidence": None, "evidence_ids": [], "raw_result_refs": [],
                "review_reasons": ["detected_cell_has_no_recognition_result"], "alternatives": [],
                "observations": []}
    observations = [dict(value) for value in cell.get("observations") or [] if isinstance(value, Mapping)]
    text = _text(cell.get("text"))
    raw_value = cell.get("raw_text")
    raw_text = raw_value if isinstance(raw_value, str) else text
    readings, contradictory_overlap = _compose_readings(observations, field, image_size)
    if text:
        readings.insert(0, {"text": text, "pass_id": _text(cell.get("pass_id")),
                            "observations": []})
    alternatives_by_value: dict[str, dict[str, Any]] = {}
    for reading in readings:
        value = _comparable(reading["text"], field)
        alternative = alternatives_by_value.setdefault(value, {
            "value": reading["text"], "raw_text": reading["text"], "confidence": None,
            "evidence_ids": [], "raw_result_refs": [], "pass_ids": [],
        })
        sources = reading["observations"]
        alternative["evidence_ids"] = list(dict.fromkeys([
            *alternative["evidence_ids"], *_source_ids(sources, {})]))
        alternative["raw_result_refs"] = list(dict.fromkeys([
            *alternative["raw_result_refs"], *_raw_refs(sources, {})]))
        if reading["pass_id"] and reading["pass_id"] not in alternative["pass_ids"]:
            alternative["pass_ids"].append(reading["pass_id"])
        scores = [_score(source.get("confidence")) for source in sources]
        scores = [score for score in scores if score is not None]
        if alternative["confidence"] is not None:
            scores.append(alternative["confidence"])
        if scores:
            alternative["confidence"] = min(scores)
    reasons = []
    if source_box is None:
        reasons.append("invalid_or_missing_cell_geometry")
    elif not _inside(source_box, table_bbox):
        reasons.append("cell_outside_detected_table")
    if source_box is not None:
        for observation in observations:
            observed_box = _box(observation.get("bbox"), image_size)
            if observed_box is None:
                continue
            center_x = (observed_box[0] + observed_box[2]) / 2
            center_y = (observed_box[1] + observed_box[3]) / 2
            if not (source_box[0] <= center_x <= source_box[2]
                    and source_box[1] <= center_y <= source_box[3]):
                reasons.append("recognition_evidence_outside_cell")
                break
    alternatives = list(alternatives_by_value.values())
    if len(alternatives) > 1 or contradictory_overlap:
        status, value = "conflicted", None
        reasons.append("different_readings_for_same_cell")
    elif alternatives:
        status, value = "recognized", text or alternatives[0]["value"]
    elif cell.get("blank_evidence") is True:
        status, value = "blank", None
    else:
        status, value = "unreadable", None
        reasons.append("detected_cell_text_not_reliably_read")
    if source_box is None and status == "recognized":
        status, value = "unmapped", None
    if "recognition_evidence_outside_cell" in reasons and status == "recognized":
        status, value = "unmapped", None
    source_ids, raw_refs = _source_ids(observations, cell), _raw_refs(observations, cell)
    if status == "recognized" and not (source_ids or raw_refs):
        reasons.append("missing_recognition_provenance")
    confidence = _score(cell.get("confidence"))
    if status == "recognized" and confidence is None and len(alternatives) == 1:
        confidence = alternatives[0]["confidence"]
    if status in REVIEW_STATES:
        confidence = None
    return {"value": value, "raw_text": raw_text, "status": status, "source_box": source_box,
            "confidence": confidence, "evidence_ids": source_ids, "raw_result_refs": raw_refs,
            "review_reasons": reasons, "alternatives": alternatives,
            "observations": observations}


def _html_grid(table_html: str | None) -> tuple[list[dict[int, dict[str, Any]]], list[str]]:
    if not table_html:
        return [], []
    try:
        raw_rows = extract_cells(table_html)
        grid: list[dict[int, dict[str, Any]]] = [{} for _ in raw_rows]
        for row_index, row in enumerate(raw_rows):
            column = 0
            for cell_index, cell in enumerate(row):
                while column in grid[row_index]:
                    column += 1
                rowspan, colspan = int(cell["rowspan"]), int(cell["colspan"])
                if not 1 <= rowspan <= 100 or not 1 <= colspan <= 100:
                    raise ValueError("Unbounded table span")
                source = {"row_index": row_index, "cell_index": cell_index,
                          "rowspan": rowspan, "colspan": colspan,
                          "raw_text": cell["text"], "is_header": cell["is_header"]}
                for target_row in range(row_index, min(len(grid), row_index + rowspan)):
                    for target_column in range(column, column + colspan):
                        if target_column in grid[target_row]:
                            raise ValueError("Overlapping table spans")
                        grid[target_row][target_column] = source
                column += colspan
        return grid, []
    except (KeyError, TypeError, ValueError):
        return [], ["recognized_html_has_invalid_spans"]


def structure_part_table(
    *, table_id: str, page: int, source_sha256: str, image_size: Sequence[float],
    table_bbox: Sequence[float], columns: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]], table_html: str | None = None,
    revision: str | None = None,
) -> list[dict[str, Any]]:
    """Return one traceable part row for each detected physical row, in source order.

    ``columns`` contain a field ``key`` and page-pixel ``bbox``. ``rows`` contain
    ``row_index``, ``bbox`` and ``cells`` with ``column_index``, ``bbox``, ``text``,
    ``raw_text``, ``observations`` and an explicit ``blank_evidence`` flag. Each
    observation can retain ``source_id``, ``text``, ``bbox``, ``pass_id``, actual
    ``confidence`` and ``raw_ref``. Different passes are compared cell by cell;
    their texts never become fabricated field values or neighboring-row anchors.

    Optional recognized HTML supplies span provenance only. Its text does not
    replace missing cell recognition or establish alignment by itself.
    """
    if (not table_id or not source_sha256 or type(page) is not int or page < 1
            or len(image_size) != 2
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) or value <= 0 for value in image_size)):
        raise ValueError("Invalid part-table source identity or page dimensions")
    table_box = _box(table_bbox, image_size)
    if table_box is None:
        raise ValueError("Invalid detected part-table bounds")
    column_keys = [_field(column, index) for index, column in enumerate(columns)]
    html_grid, html_errors = _html_grid(table_html)
    if html_grid and len(html_grid) != len(rows):
        html_errors.append("recognized_html_and_physical_row_counts_differ")
        html_grid = []
    result = []
    for position, row in enumerate(rows):
        cells_by_column: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for cell in row.get("cells") or []:
            index = cell.get("column_index")
            if type(index) is int and index >= 0:
                cells_by_column[index].append(cell)
        row_box = _box(row.get("bbox"), image_size)
        valid_boxes = [box for cells in cells_by_column.values() for cell in cells
                       if (box := _box(cell.get("bbox"), image_size)) is not None]
        whole_box = _union([*([row_box] if row_box else []), *valid_boxes])
        fields, extra_fields = {}, {}
        indexes = sorted(set(range(len(columns))) | set(cells_by_column))
        seen_keys: set[str] = set()
        for index in indexes:
            key = column_keys[index] if index < len(column_keys) else f"column_{index}"
            duplicate_key = key in seen_keys
            seen_keys.add(key)
            cells = cells_by_column.get(index) or []
            cell = cells[0] if cells else None
            cell_box = _box(cell.get("bbox"), image_size) if cell else None
            if cell_box is None and not cells and row_box is not None and index < len(columns):
                column_box = _box(columns[index].get("bbox"), image_size)
                if column_box is not None:
                    cell_box = _box([max(column_box[0], row_box[0]), max(column_box[1], row_box[1]),
                                     min(column_box[2], row_box[2]), min(column_box[3], row_box[3])], image_size)
            evidence = _cell_evidence(cell, field=key, source_box=cell_box,
                                      table_bbox=table_box, image_size=image_size)
            evidence["column_index"] = index
            if index < len(columns):
                evidence["column_label"] = _text(columns[index].get("label")) or None
            if html_grid and index in html_grid[position]:
                evidence["recognized_html_span"] = dict(html_grid[position][index])
            if len(cells) > 1:
                evidence["status"] = "unmapped"
                evidence["value"] = None
                evidence["confidence"] = None
                evidence["review_reasons"].append("multiple_physical_cells_share_column_index")
                evidence["unmapped_cells"] = [dict(value) for value in cells]
            if duplicate_key:
                evidence["review_reasons"].append("multiple_columns_share_field_key")
                extra_fields[f"{key}__column_{index}"] = evidence
            elif key in PART_FIELDS:
                fields[key] = evidence
            else:
                extra_fields[key] = evidence
        all_fields = [*fields.values(), *extra_fields.values()]
        counts = {state: sum(field["status"] == state for field in all_fields)
                  for state in ("recognized", "blank", "unreadable", "conflicted", "unmapped")}
        review_reasons = list(dict.fromkeys([
            *(reason for field in all_fields for reason in field["review_reasons"]),
            *html_errors,
        ]))
        if whole_box is None:
            review_reasons.append("detected_row_has_no_valid_source_geometry")
        if not all_fields:
            review_reasons.append("detected_row_has_no_mapped_cells")
        identity = json.dumps([source_sha256, table_id, page, row.get("row_index", position), position],
                              ensure_ascii=True, separators=(",", ":"))
        output = {
            "id": "part_table_row_" + hashlib.sha256(identity.encode()).hexdigest()[:24],
            **{field: fields[field]["value"] if field in fields else None for field in PART_FIELDS},
            "page": str(page), "source_box": whole_box, "table_id": table_id,
            "table_box": table_box, "source_sha256": source_sha256,
            "source_row_index": row.get("row_index", position),
            "source": "drawing_part_table_recognition", "location_kind": "bom_table",
            "physical_extent_verified": False, "field_evidence": fields,
            "field_boxes": {field: evidence["source_box"] for field, evidence in fields.items()
                            if evidence["source_box"] is not None},
            "extra_fields": extra_fields,
            "evidence_ids": list(dict.fromkeys(value for field in all_fields for value in field["evidence_ids"])),
            "raw_result_refs": list(dict.fromkeys(value for field in all_fields for value in field["raw_result_refs"])),
            "recognition_quality": {"detected_field_count": len(all_fields),
                                    **{f"{state}_field_count": count for state, count in counts.items()},
                                    "review_field_count": sum(bool(field["review_reasons"]) or field["status"] in REVIEW_STATES
                                                              for field in all_fields)},
            "needs_review": bool(review_reasons), "uncertain": bool(review_reasons),
            "review_reasons": review_reasons, "structure_review_reasons": list(html_errors),
        }
        if revision is not None:
            output["recognition_revision"] = revision
        part_code_box = output["field_boxes"].get("part_code")
        if part_code_box is not None:
            output.update(anchor_box=part_code_box, anchor_kind="part_code_text")
        result.append(output)
    return result
