"""Recover BOM rows from recognized HTML while preserving source cells."""
from collections import defaultdict
from html.parser import HTMLParser
import re
import statistics
from typing import Any


class TableCells(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
        if self.depth != 1:
            return
        if tag == "tr":
            self.row = []
            self.rows.append(self.row)
        elif tag in {"td", "th"} and self.row is not None:
            attrs = dict(attrs)
            self.cell = {"text": "", "rowspan": attrs.get("rowspan", "1"),
                         "colspan": attrs.get("colspan", "1"), "is_header": tag == "th"}
            self.row.append(self.cell)
        elif tag == "br" and self.cell is not None:
            self.cell["text"] += "\n"

    def handle_data(self, data):
        if self.depth == 1 and self.cell is not None:
            self.cell["text"] += data

    def handle_endtag(self, tag):
        if self.depth == 1:
            if tag in {"td", "th"}:
                self.cell = None
            elif tag == "tr":
                self.row = None
        if tag == "table":
            self.depth = max(0, self.depth - 1)


def extract_cells(text: str) -> list:
    parser = TableCells()
    parser.feed(text)
    parser.close()
    return parser.rows


HEADER_ALIASES = {
    "item_number": {"item", "itemnumber", "itemno.", "\u5e8f\u53f7", "\u4ef6\u53f7"},
    "part_number": {"partnumber", "partno.", "\u4ee3\u53f7", "\u56fe\u53f7"},
    "part_name": {"partname", "description", "\u540d\u79f0"},
    "quantity": {"quantity", "qty", "qty.", "\u6570\u91cf"},
    "material": {"material", "\u6750\u6599"},
    "specification": {"specification", "\u89c4\u683c"},
    "remark": {"remark", "remarks", "\u5907\u6ce8"},
}

PART_NUMBER = re.compile(r"^(?=.*\d)(?=.*[.#/])[A-Za-z0-9_.#/+\-]+$")
MATERIAL = re.compile(
    r"^(?:SUS\s*)?(?:304|316L?|Q235-?[A-Z]?|Q345-?[A-Z]?|ZG\d+(?:-\d+)?|20#|45#)",
    re.IGNORECASE,
)
SPECIFICATION = re.compile(r"(?:^t\s*\d|\d\s*[x*×]\s*\d|方管|圆管|槽钢)", re.IGNORECASE)


def normalize_part_number(value: str) -> str:
    """Normalize OCR spacing/escaping without changing the visible identifier."""
    return "".join(str(value).replace("\\#", "#").replace("＃", "#").split())


def _spatial_bbox(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = item.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _spatial_center(item: dict[str, Any]) -> tuple[float, float] | None:
    bbox = _spatial_bbox(item)
    return None if bbox is None else ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _spatial_iou(first: tuple[float, float, float, float],
                 second: tuple[float, float, float, float]) -> float:
    x_overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    y_overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = x_overlap * y_overlap
    if not intersection:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _spatial_part_code(value: str) -> bool:
    normalized = normalize_part_number(value)
    return bool(PART_NUMBER.fullmatch(normalized)) and sum(char.isalpha() for char in normalized) >= 2


def _spatial_number(value: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}", value.strip()))


def _spatial_row_values(code: dict[str, Any], item: dict[str, Any], quantity: dict[str, Any],
                        nearby: list[dict[str, Any]]) -> dict[str, Any]:
    code_bbox = _spatial_bbox(code)
    item_bbox = _spatial_bbox(item)
    quantity_bbox = _spatial_bbox(quantity)
    assert code_bbox is not None and item_bbox is not None and quantity_bbox is not None
    code_x = (code_bbox[0] + code_bbox[2]) / 2
    quantity_x = (quantity_bbox[0] + quantity_bbox[2]) / 2
    line_y = (code_bbox[1] + code_bbox[3]) / 2
    tolerance = max(18.0, 0.8 * (code_bbox[3] - code_bbox[1]))
    between = []
    after_quantity = []
    for candidate in nearby:
        bbox = _spatial_bbox(candidate)
        text = str(candidate.get("text") or "").strip()
        center = _spatial_center(candidate)
        if bbox is None or center is None or not text or abs(center[1] - line_y) > tolerance:
            continue
        if candidate is code or candidate is item or candidate is quantity:
            continue
        if _spatial_number(text) or _spatial_part_code(text):
            continue
        if code_bbox[2] < center[0] < quantity_x:
            between.append(candidate)
        elif center[0] > quantity_x:
            after_quantity.append(candidate)
    name = next((str(candidate["text"]).strip() for candidate in sorted(
        between, key=lambda value: _spatial_center(value)[0])
        if str(candidate.get("text") or "").strip()), None)
    material = next((str(candidate["text"]).strip() for candidate in after_quantity + between
                     if MATERIAL.match(str(candidate.get("text") or "").strip())), None)
    specification = next((str(candidate["text"]).strip() for candidate in after_quantity + between
                          if SPECIFICATION.search(str(candidate.get("text") or "").strip())), None)
    remark = next(("本图" for candidate in after_quantity + between
                   if "本图" in str(candidate.get("text") or "")), None)
    evidence = [item.get("entity_id"), code.get("entity_id"), quantity.get("entity_id")]
    evidence.extend(candidate.get("entity_id") for candidate in between + after_quantity)
    evidence = [value for value in evidence if value]
    raw_refs = [item.get("raw_result_ref"), code.get("raw_result_ref"), quantity.get("raw_result_ref")]
    raw_refs.extend(candidate.get("raw_result_ref") for candidate in between + after_quantity)
    raw_refs = [value for value in raw_refs if value]
    row_bbox = [min(item_bbox[0], code_bbox[0]), min(item_bbox[1], code_bbox[1]),
                max(quantity_bbox[2], code_bbox[2]), max(item_bbox[3], code_bbox[3])]
    return {
        "values": {"item_number": str(item["text"]).strip(),
                   "part_number": normalize_part_number(str(code["text"])),
                   "part_name": name, "quantity": str(quantity["text"]).strip(),
                   "material": material, "specification": specification, "remark": remark},
        "raw_cells": [{"text": str(value), "rowspan": "1", "colspan": "1", "is_header": False}
                      for value in (item.get("text"), code.get("text"), name or "",
                                    quantity.get("text"), specification or "", material or "",
                                    remark or "")],
        "bbox": row_bbox, "evidence_ids": list(dict.fromkeys(evidence)),
        "raw_result_refs": list(dict.fromkeys(raw_refs)),
    }


def spatial_bom_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover only strongly repeated BOM rows from OCR coordinates.

    This is deliberately stricter than a generic line parser: rows need a part
    code, a numeric item and quantity, stable columns, and a contiguous item
    sequence of at least two rows. A lone title-block drawing number is ignored.
    """
    positioned = []
    for item in items:
        text = str(item.get("text") or "").strip()
        if text and _spatial_bbox(item) is not None and _spatial_center(item) is not None:
            positioned.append({**item, "text": text})
    codes = []
    for item in sorted(positioned, key=lambda value: -float(value.get("score", 0.0))):
        if not _spatial_part_code(item["text"]):
            continue
        bbox = _spatial_bbox(item)
        if bbox is not None and bbox[3] - bbox[1] > 1.5 * (bbox[2] - bbox[0]):
            continue
        if any(_spatial_iou(_spatial_bbox(existing), bbox) > 0.50 for existing in codes):
            continue
        codes.append(item)
    candidates = []
    for code in codes:
        code_bbox = _spatial_bbox(code)
        code_center = _spatial_center(code)
        assert code_bbox is not None and code_center is not None
        tolerance = max(18.0, 0.8 * (code_bbox[3] - code_bbox[1]))
        column_gap = max(180.0, code_bbox[2] - code_bbox[0])
        left = [item for item in positioned
                if _spatial_number(item["text"])
                and _spatial_center(item)[0] < code_center[0]
                and code_center[0] - _spatial_center(item)[0] <= column_gap
                and abs(_spatial_center(item)[1] - code_center[1]) <= tolerance]
        right = [item for item in positioned
                 if _spatial_number(item["text"])
                 and _spatial_center(item)[0] > code_center[0]
                 and abs(_spatial_center(item)[1] - code_center[1]) <= tolerance]
        if not left or not right:
            continue
        item = min(left, key=lambda value: code_center[0] - _spatial_center(value)[0])
        quantity = min(right, key=lambda value: _spatial_center(value)[0] - code_center[0])
        candidates.append({**_spatial_row_values(code, item, quantity, positioned),
                           "item_x": _spatial_center(item)[0],
                           "code_x": code_center[0], "quantity_x": _spatial_center(quantity)[0],
                           "line_y": code_center[1]})
    deduplicated = []
    for candidate in sorted(candidates, key=lambda value: value["line_y"]):
        values = candidate["values"]
        if not any(existing["values"]["part_number"] == values["part_number"]
                   and abs(existing["line_y"] - candidate["line_y"]) < 20 for existing in deduplicated):
            deduplicated.append(candidate)
    groups = []
    for candidate in deduplicated:
        matched = next((group for group in groups
                        if abs(statistics.median(row["item_x"] for row in group) - candidate["item_x"]) < 80
                        and abs(statistics.median(row["code_x"] for row in group) - candidate["code_x"]) < 100
                        and abs(statistics.median(row["quantity_x"] for row in group) - candidate["quantity_x"]) < 100
                        and abs(statistics.median(row["line_y"] for row in group) - candidate["line_y"]) < 1500), None)
        if matched is None:
            groups.append([candidate])
        else:
            matched.append(candidate)
    recovered = []
    for group in groups:
        group.sort(key=lambda value: value["line_y"])
        if len(group) < 2:
            continue
        numbers = [int(row["values"]["item_number"]) for row in group]
        contiguous = (all(first - second == 1 for first, second in zip(numbers, numbers[1:]))
                      or all(second - first == 1 for first, second in zip(numbers, numbers[1:])))
        stable_columns = max(statistics.pstdev([row[key] for row in group])
                             for key in ("item_x", "code_x", "quantity_x")) < 100
        if not contiguous or len(set(numbers)) != len(numbers) or not stable_columns:
            continue
        for row in group:
            row.pop("item_x", None)
            row.pop("code_x", None)
            row.pop("quantity_x", None)
            row.pop("line_y", None)
        recovered.extend(group)
    return recovered


def localized_bom_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decode complete isolated rows without requiring a contiguous sequence."""
    positioned = []
    for item in items:
        text = str(item.get("text") or "").strip()
        if text and _spatial_bbox(item) is not None and _spatial_center(item) is not None:
            positioned.append({**item, "text": text})
    rows = []
    codes = []
    for code in sorted(positioned, key=lambda value: -float(value.get("score", 0.0))):
        if not _spatial_part_code(code["text"]):
            continue
        code_box = _spatial_bbox(code)
        code_center = _spatial_center(code)
        assert code_box is not None and code_center is not None
        if any(_spatial_iou(_spatial_bbox(existing), code_box) > 0.50 for existing in codes):
            continue
        codes.append(code)
        # A neighboring BOM row is commonly separated by roughly one text
        # height; do not borrow its sequence number when the local row is
        # missing a token.
        tolerance = max(10.0, 0.55 * (code_box[3] - code_box[1]))
        column_gap = max(180.0, code_box[2] - code_box[0])
        left = [item for item in positioned
                if _spatial_number(item["text"])
                and _spatial_center(item)[0] < code_center[0]
                and code_center[0] - _spatial_center(item)[0] <= column_gap
                and abs(_spatial_center(item)[1] - code_center[1]) <= tolerance]
        right = [item for item in positioned
                 if _spatial_number(item["text"])
                 and _spatial_center(item)[0] > code_center[0]
                 and abs(_spatial_center(item)[1] - code_center[1]) <= tolerance]
        if not left or not right:
            continue
        item = min(left, key=lambda value: code_center[0] - _spatial_center(value)[0])
        quantity = min(right, key=lambda value: _spatial_center(value)[0] - code_center[0])
        row = _spatial_row_values(code, item, quantity, positioned)
        part_name = str(row["values"].get("part_name") or "").strip()
        quantity_text = str(row["values"].get("quantity") or "").strip()
        header_name = re.sub(r"\s+", "", part_name).casefold()
        if (not part_name or header_name in {"material", "quantity", "qty", "name",
                                              "partname", "description", "名称", "数量", "材料", "材质", "规格", "备注",
                                              "标准", "标准件", "合计", "总计"}
                or not re.fullmatch(r"\d{1,3}", quantity_text)
                or int(quantity_text) <= 0):
            continue
        if not any(existing["values"] == row["values"] for existing in rows):
            rows.append(row)
    return rows


def vertical_bom_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decode BOMs whose rendered columns run vertically down the page.

    Every field must share one X column and occupy a bounded Y band relative
    to the part number; unrelated dimensions cannot satisfy all three anchors.
    """
    positioned = []
    for item in items:
        text = str(item.get("text") or "").strip()
        if text and _spatial_bbox(item) is not None and _spatial_center(item) is not None:
            positioned.append({**item, "text": text})
    codes = []
    for code in sorted(positioned, key=lambda value: -float(value.get("score", 0.0))):
        if not _spatial_part_code(code["text"]):
            continue
        box = _spatial_bbox(code)
        assert box is not None
        if box[3] - box[1] <= 1.5 * (box[2] - box[0]):
            continue
        if any(_spatial_iou(_spatial_bbox(existing), box) > 0.50 for existing in codes):
            continue
        codes.append(code)
    rows = []
    headers = {"material", "quantity", "qty", "name", "partname", "description",
               "名称", "数量", "材料", "材质", "规格", "备注"}
    for code in codes:
        code_box = _spatial_bbox(code)
        code_center = _spatial_center(code)
        assert code_box is not None and code_center is not None
        x_tolerance = max(18.0, 0.9 * (code_box[2] - code_box[0]))
        column = [item for item in positioned
                  if abs(_spatial_center(item)[0] - code_center[0]) <= x_tolerance]
        item_candidates = [item for item in column if _spatial_number(item["text"])
                           and code_box[1] - 280 <= _spatial_center(item)[1] < code_box[1] - 8]
        name_candidates = [item for item in column
                           if code_box[3] + 12 <= _spatial_center(item)[1] <= code_box[3] + 230
                           and not _spatial_number(item["text"])
                           and not _spatial_part_code(item["text"])]
        quantity_candidates = [item for item in column if _spatial_number(item["text"])
                               and code_box[3] + 120 <= _spatial_center(item)[1] <= code_box[3] + 430]
        if not item_candidates or not name_candidates or not quantity_candidates:
            continue
        item = min(item_candidates, key=lambda value: (
            -float(value.get("score", 0.0)),
            abs(_spatial_center(value)[1] - (code_box[1] - 45))))
        name = min(name_candidates, key=lambda value: (
            -float(value.get("score", 0.0)),
            abs(_spatial_center(value)[1] - (code_box[3] + 100))))
        quantity = min(quantity_candidates, key=lambda value: (
            -float(value.get("score", 0.0)),
            abs(_spatial_center(value)[1] - (code_box[3] + 260))))
        part_name = str(name["text"]).strip()
        quantity_text = str(quantity["text"]).strip()
        header_name = re.sub(r"\s+", "", part_name).casefold()
        if (header_name in headers or not re.fullmatch(r"\d{1,3}", quantity_text)
                or int(quantity_text) <= 0):
            continue
        after_quantity = [candidate for candidate in column
                          if _spatial_center(candidate)[1] > _spatial_center(quantity)[1] + 20
                          and not _spatial_number(candidate["text"])
                          and not _spatial_part_code(candidate["text"])]
        material = next((str(candidate["text"]).strip() for candidate in after_quantity
                         if MATERIAL.match(str(candidate["text"]).strip())), None)
        item_box = _spatial_bbox(item)
        quantity_box = _spatial_bbox(quantity)
        assert item_box is not None and quantity_box is not None
        values = {"item_number": item["text"].strip(),
                  "part_number": normalize_part_number(code["text"]),
                  "part_name": part_name, "quantity": quantity_text,
                  "material": material, "specification": None, "remark": None}
        evidence = [item.get("entity_id"), code.get("entity_id"), name.get("entity_id"),
                    quantity.get("entity_id")]
        refs = [item.get("raw_result_ref"), code.get("raw_result_ref"),
                name.get("raw_result_ref"), quantity.get("raw_result_ref")]
        row = {"values": values,
               "raw_cells": [{"text": str(value), "rowspan": "1", "colspan": "1", "is_header": False}
                             for value in (item["text"], code["text"], part_name, quantity_text,
                                           "", material or "", "")],
               "bbox": [min(item_box[0], code_box[0]), code_box[1],
                        max(quantity_box[2], code_box[2]), max(quantity_box[3], quantity_box[3])],
               "evidence_ids": list(dict.fromkeys(value for value in evidence if value)),
               "raw_result_refs": list(dict.fromkeys(value for value in refs if value)),
               "_item_entity_id": item.get("entity_id"),
               "_item_raw_result_ref": item.get("raw_result_ref"),
               "_item_options": sorted(
                   [(candidate["text"].strip(), candidate.get("entity_id"),
                     candidate.get("raw_result_ref"), float(candidate.get("score", 0.0)))
                    for candidate in item_candidates], key=lambda value: -value[3])}
        if not any(existing["values"] == values for existing in rows):
            rows.append(row)
    # OCR can confuse adjacent sequence numerals (for example 6 and 9).  Use
    # a unique sequence assignment when a same-column alternative exists;
    # never duplicate an item number silently.
    used = {row["values"]["item_number"] for row in rows
            if sum(other["values"]["item_number"] == row["values"]["item_number"] for other in rows) == 1}
    for row in rows:
        current = row["values"]["item_number"]
        if sum(other["values"]["item_number"] == current for other in rows) <= 1:
            continue
        alternative = next((option for option in row.get("_item_options", [])
                            if option[0] not in used), None)
        if alternative is None:
            continue
        row["values"]["item_number"] = alternative[0]
        row["raw_cells"][0]["text"] = alternative[0]
        if row.get("_item_entity_id"):
            row["evidence_ids"] = [value for value in row["evidence_ids"]
                                    if value != row["_item_entity_id"]]
        if row.get("_item_raw_result_ref"):
            row["raw_result_refs"] = [value for value in row["raw_result_refs"]
                                       if value != row["_item_raw_result_ref"]]
        if alternative[1]:
            row["evidence_ids"].append(alternative[1])
        if alternative[2]:
            row["raw_result_refs"].append(alternative[2])
        used.add(alternative[0])
    for row in rows:
        row.pop("_item_options", None)
        row.pop("_item_entity_id", None)
        row.pop("_item_raw_result_ref", None)
    return rows


def _signature_rows(rows: list) -> list[dict]:
    """Recover repeated BOM-shaped rows only when a header is unavailable."""
    groups = defaultdict(list)
    for row_index, row in enumerate(rows):
        values = [cell["text"].strip() for cell in row]
        for start in range(len(values) - 3):
            item_number, part_number, part_name, quantity = values[start:start + 4]
            part_number = normalize_part_number(part_number)
            if not (item_number.isdigit() and PART_NUMBER.fullmatch(part_number)
                    and part_name and not part_name.isdigit() and quantity.isdigit()):
                continue
            tail = values[start + 4:start + 8]
            material = next((value for value in tail if MATERIAL.match(value)), None)
            specification = next((value for value in tail
                                  if value != material and SPECIFICATION.search(value)), None)
            remark = next((value for value in tail if value == "本图"), None)
            groups[start].append({
                "row_index": row_index, "group_start": start,
                "values": {"item_number": item_number, "part_number": part_number,
                           "part_name": part_name, "quantity": quantity,
                           "material": material, "specification": specification,
                           "remark": remark},
                "raw_cells": row[start:start + 8], "parse_mode": "headerless_signature",
            })

    decoded = []
    for candidates in groups.values():
        runs = []
        for candidate in candidates:
            if not runs or candidate["row_index"] != runs[-1][-1]["row_index"] + 1:
                runs.append([])
            runs[-1].append(candidate)
        for run in runs:
            numbers = [int(row["values"]["item_number"]) for row in run]
            monotonic = (all(first < second for first, second in zip(numbers, numbers[1:]))
                         or all(first > second for first, second in zip(numbers, numbers[1:])))
            if len(run) >= 2 and len(set(numbers)) == len(numbers) and monotonic:
                decoded.extend(run)
    return sorted(decoded, key=lambda row: (row["row_index"], row["group_start"]))


def bom_rows(rows: list) -> list[dict]:
    """Decode only contiguous numeric BOM rows with unambiguous header spans.

    Rowspans are retained in raw cells. They do not shift cells in rows above
    the header, which is where Paddle commonly emits the actual BOM entries.
    """
    def columns(row):
        result, column = {}, 0
        for cell in row:
            rowspan = int(cell["rowspan"])
            if not 1 <= rowspan <= 100:
                return None
            span = int(cell["colspan"])
            if not 1 <= span <= 100:
                return None
            result[(column, column + span)] = cell["text"].strip()
            column += span
        return result

    decoded = {}
    for header_index, row in enumerate(rows):
        header = columns(row)
        if header is None:
            continue
        roles, ambiguous = {}, False
        for position, text in header.items():
            normalized = "".join(text.split()).casefold()
            for role, aliases in HEADER_ALIASES.items():
                if normalized in aliases:
                    if role in roles:
                        ambiguous = True
                    roles[role] = position
        if ambiguous or not {"item_number", "part_name", "quantity"} <= roles.keys():
            continue
        # Without a grid expander, a rowspan can only be interpreted safely for
        # data rows that precede the header and therefore cannot cross it.
        directions = (-1,) if any(int(cell["rowspan"]) != 1 for cell in row) else (-1, 1)
        for direction in directions:
            index = header_index + direction
            while 0 <= index < len(rows):
                data = columns(rows[index])
                if data is None or not set(roles.values()) <= data.keys():
                    break
                values = {role: data[position] or None for role, position in roles.items()}
                if not (values["item_number"] or "").isdigit() or not values["part_name"]:
                    break
                decoded[index] = {"row_index": index, "header_row_index": header_index,
                                  "values": values, "raw_cells": rows[index],
                                  "parse_mode": "header"}
                index += direction
    return ([decoded[index] for index in sorted(decoded)]
            if decoded else _signature_rows(rows))


def bom_part_number_coverage(table_html: list[str]) -> dict[str, int]:
    """Compare part numbers visible in table cells with decoded BOM rows."""
    observed, decoded = set(), set()
    for text in table_html:
        rows = extract_cells(text)
        observed.update(
            normalize_part_number(cell["text"].strip())
            for row in rows
            for cell in row
            if PART_NUMBER.fullmatch(normalize_part_number(cell["text"].strip()))
        )
        decoded.update(
            normalize_part_number(row["values"]["part_number"])
            for row in bom_rows(rows)
            if row["values"].get("part_number")
        )
    return {
        "observed_part_numbers": len(observed),
        "decoded_part_numbers": len(decoded),
        "unmatched_part_numbers": len(observed - decoded),
    }
