"""Bind literal title values to labelled, closed native table cells.

The table boundary is evidence, not proximity alone. Multiple values, missing
boundaries or unresolved preview transforms remain unknown. No PDF answers,
file-name-derived values or custom parameter guesses are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.dwg_route_validation_20261015.parser_probe import digest, write_new_json

LABELS = {"图号": "drawing_number", "版本": "revision", "材料": "material",
          "数量": "quantity", "编码": "item_code", "常用名": "common_name"}
TOLERANCE = 1e-6


def label_role(raw: str) -> str | None:
    for label, role in LABELS.items():
        if raw.lstrip().startswith(label):
            return role
    return None


def covered(intervals: list[tuple[float, float]], low: float, high: float) -> bool:
    endpoint = low
    for start, stop in sorted(intervals):
        if stop < endpoint - TOLERANCE:
            continue
        if start > endpoint + TOLERANCE:
            return False
        endpoint = max(endpoint, stop)
        if endpoint >= high - TOLERANCE:
            return True
    return False


def grid(decoded: dict[str, Any]):
    heights = [row["attributes"]["height"] for row in decoded["texts"]
               if row["attributes"].get("height", 0) > 0]
    if not heights:
        return {}, {}
    minimum = 2 * min(heights)
    horizontal, vertical = {}, {}
    for row in decoded["border_line_candidates"]:
        a, b = row["start"], row["end"]
        if len(a) != 3 or len(b) != 3 or a[2] != 0 or b[2] != 0:
            continue
        if abs(a[1] - b[1]) <= TOLERANCE and abs(a[0] - b[0]) >= minimum:
            horizontal.setdefault(round(a[1], 6), []).append(tuple(sorted((a[0], b[0]))))
        elif abs(a[0] - b[0]) <= TOLERANCE and abs(a[1] - b[1]) >= minimum:
            vertical.setdefault(round(a[0], 6), []).append(tuple(sorted((a[1], b[1]))))
    return horizontal, vertical


def closed_cells(point, horizontal, vertical):
    x, y = point[:2]
    cells = []
    for left in vertical:
        if left >= x - TOLERANCE:
            continue
        for right in vertical:
            if right <= x + TOLERANCE:
                continue
            for bottom in horizontal:
                if bottom >= y - TOLERANCE:
                    continue
                for top in horizontal:
                    if top <= y + TOLERANCE:
                        continue
                    if (covered(horizontal[bottom], left, right)
                            and covered(horizontal[top], left, right)
                            and covered(vertical[left], bottom, top)
                            and covered(vertical[right], bottom, top)):
                        cells.append((left, bottom, right, top))
    return sorted(cells, key=lambda cell: ((cell[2] - cell[0]) * (cell[3] - cell[1]), cell))


def inside(point, cell):
    return (cell[0] + TOLERANCE < point[0] < cell[2] - TOLERANCE
            and cell[1] + TOLERANCE < point[1] < cell[3] - TOLERANCE)


def bind_fields(decoded: dict[str, Any]) -> dict[str, Any]:
    limits = decoded["coordinate_limits"]
    if (limits["matrix_commands_present"] or limits["non_z_text_normals"]
            or limits.get("rotated_text_frames")):
        return {"status": "unknown_coordinate_frame", "fields": {},
                "reason": "Preview transform or text orientation is not independently verified"}
    horizontal, vertical = grid(decoded)
    values = [row for row in decoded["texts"]
              if row["raw_text"].strip() and label_role(row["raw_text"]) is None
              and not any(mark in row["raw_text"] for mark in ("\\", "{", "}"))]
    results = {}
    for row in decoded["texts"]:
        role = label_role(row["raw_text"])
        if role is None:
            continue
        cells = closed_cells(row["wcs_insert"], horizontal, vertical)
        if not cells:
            results.setdefault(role, []).append({"status": "unknown", "reason": "label_cell_not_closed"})
            continue
        label_cell = cells[0]
        selected = label_cell
        matches = [value for value in values if inside(value["wcs_insert"], selected)]
        direction = "same_closed_cell"
        if not matches:
            below = [cell for y in horizontal if y < label_cell[1] - TOLERANCE
                     for cell in [(label_cell[0], y, label_cell[2], label_cell[1])]
                     if covered(horizontal[y], cell[0], cell[2])
                     and covered(vertical[cell[0]], cell[1], cell[3])
                     and covered(vertical[cell[2]], cell[1], cell[3])]
            if below:
                selected = max(below, key=lambda cell: cell[1])
                other_labels = [value for value in decoded["texts"]
                                if label_role(value["raw_text"]) is not None
                                and inside(value["wcs_insert"], selected)]
                # A blank MATERIAL or CODE cell must not inherit DRAWING NO or
                # COMMON NAME from the next labelled row.
                if not other_labels:
                    matches = [value for value in values if inside(value["wcs_insert"], selected)]
                    direction = "directly_below_shared_closed_border"
                else:
                    selected = label_cell
        unique = {value["raw_text"] for value in matches}
        evidence = [{"raw_text": value["raw_text"], "raw_text_hex": value["raw_text_hex"],
                     "frame_index": value["frame_index"], "byte_offset": value["byte_offset"],
                     "cad_insert": value["wcs_insert"]} for value in matches]
        results.setdefault(role, []).append({
            "status": "candidate_labelled_cell" if len(unique) == 1 else "unknown",
            "raw_value": next(iter(unique)) if len(unique) == 1 else None,
            "label_raw_text": row["raw_text"], "label_frame_index": row["frame_index"],
            "label_cell_cad_bbox": list(label_cell), "value_cell_cad_bbox": list(selected),
            "binding_basis": direction, "value_evidence": evidence,
            "confidence": None, "physical_owner": "unknown",
            "reason": None if len(unique) == 1 else "missing_or_conflicting_cell_values",
        })
    # No case-specific candidate or drawing filename enters this binding.
    return {"status": "native_cell_candidates", "fields": results,
            "physical_owner_confirmed": False, "custom_object_parameters_decoded": False}


def run(source: Path, output: Path, *, replay: bool = False) -> dict[str, Any]:
    inventory = json.loads(source.read_text(encoding="utf-8"))
    if inventory["source_count"] != 14 or len(inventory["decoded_cases"]) != 14:
        raise ValueError("Only the complete fixed14 preview inventory is supported")
    rows = []
    for record in inventory["decoded_cases"]:
        path = Path(record["decoded_path"])
        if digest(path) != record["decoded_sha256"]:
            raise ValueError("Frozen native preview changed")
        decoded = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"item_code": record["item_code"], "source_sha256": decoded["source"]["source_sha256"],
                     "decoded_sha256": digest(path), "preview_sha256": decoded["preview_sha256"],
                     "title_object_handle": record["handle"], **bind_fields(decoded)})
    result = {"schema_version": "native_title_cell_fields.v1", "source_count": len(rows),
              "source_inventory_sha256": digest(source), "code_sha256": digest(Path(__file__)),
              "cases": rows, "pdf_read": False, "ocr_used": False,
              "route_labels_read": False, "val_test_read": False,
              "route_training_admitted": False}
    if replay:
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise ValueError("Independent native title cell replay differs")
    else:
        write_new_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    result = run(args.source, args.output, replay=args.replay)
    print(json.dumps({"source_count": result["source_count"], "route_training_admitted": False}))


if __name__ == "__main__":
    main()
