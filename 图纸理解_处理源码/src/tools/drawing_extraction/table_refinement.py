"""Repair detected part cells from the current PDF, with bounded rereading."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pymupdf

from src.tools.part_table_recognition.geometry import cell_has_ink, render_source_crop, ruled_grid
from src.tools.part_table_recognition.layout import field_key
from src.tools.part_table_recognition.reader import spotting_tokens
from src.tools.part_table_recognition.structure import structure_part_table

from .document_schema import DrawingDocument, Page
from .region_reader import read_tile, tile_boxes


def intersection_fraction(box: list[float], container: list[float]) -> float:
    area = max(0, min(box[2], container[2]) - max(box[0], container[0])) * max(
        0, min(box[3], container[3]) - max(box[1], container[1]))
    return area / max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))


def observations(document: DrawingDocument, page: Page) -> list[dict[str, Any]]:
    return [{"text": entity.raw_text, "bbox": list(location.bbox),
             "source_id": entity.id, "raw_ref": next(iter(entity.raw_result_refs), ""),
             "pass_id": str(entity.attributes.get("recognition_pass") or "primary"),
             "confidence": entity.confidence}
            for entity in document.raw_ocr if entity.raw_text
            for location in entity.locations
            if location.page_number == page.page_number and location.bbox]


def header_band(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headers = [token for token in tokens if field_key(token["text"])]
    for anchor in headers:
        center = (anchor["bbox"][1] + anchor["bbox"][3]) / 2
        band = [token for token in headers
                if token["bbox"][1] <= center <= token["bbox"][3]]
        keys = {field_key(token["text"]) for token in band}
        if len(keys) >= 3 and {"item_no", "part_code"}.issubset(keys):
            return band
    return []


def _compose(tokens: list[dict[str, Any]]) -> str:
    return " ".join(token["text"] for token in sorted(tokens, key=lambda row: (row["bbox"][1], row["bbox"][0])))


def table_data_direction(tokens: list[dict], band: list[dict]) -> int:
    """Locate serial numbers within their header span, regardless of text alignment."""
    header = next(token for token in band if field_key(token["text"]) == "item_no")
    x0, y0, x1, y1 = header["bbox"]
    padding = (x1 - x0) * .25
    center = (y0 + y1) / 2
    centers = {(token["bbox"][1] + token["bbox"][3]) / 2 for token in tokens
               if token["text"].strip().isascii() and token["text"].strip().isdecimal()
               and x0 - padding <= (token["bbox"][0] + token["bbox"][2]) / 2 <= x1 + padding}
    above = sum(value < center for value in centers)
    below = sum(value > center for value in centers)
    return -1 if above > below else 1 if below > above else 0


def read_source_region(source: Path, page: Page, box: list[float], *, dpi: int,
                       endpoint: str, directory: Path, artifact: Path,
                       rotation: int = 0,
                       tile_reader: Callable = read_tile) -> tuple[list[dict], list[dict]]:
    crop = render_source_crop(source, page=page.page_number, box=box,
                              page_size=[page.width, page.height],
                              pixel_to_pdf=page.pixel_to_pdf_matrix, dpi=dpi, rotation=rotation)
    readings, attempts = [], []
    for index, tile in enumerate(tile_boxes(crop.image.shape[1], crop.image.shape[0])):
        x0, y0, x1, y1 = tile
        output = directory / f"tile-{index:03d}.json"
        result = tile_reader(crop.image[y0:y1, x0:x1], endpoint=endpoint, output=output)
        ref = str(output.relative_to(artifact))
        attempt = {"raw_ref": ref, "image_ref": str(output.with_suffix(".png").relative_to(artifact)),
                   "dpi": dpi, "tile_box": tile, "source_box": crop.page_box(tile),
                   "pixel_to_pdf_matrix": list(page.pixel_to_pdf_matrix),
                   "raster_origin": list(crop.raster_origin), "pdf_derotation": crop.pdf_derotation,
                   "recognition_rotation": rotation,
                   "status": result["status"], "finish_reason": result["finish_reason"],
                   "elapsed_seconds": result["elapsed_seconds"],
                   "review_reason": result.get("review_reason"), "usage": result.get("usage")}
        attempts.append(attempt)
        if result["status"] == "not_attempted":
            break
        if result["status"] != "completed":
            continue
        for token_index, token in enumerate(spotting_tokens(result["content"], (x1 - x0, y1 - y0))):
            local = token["bbox"]
            mapped = crop.page_box([local[0] + x0, local[1] + y0, local[2] + x0, local[3] + y0])
            if any(previous["text"] == token["text"]
                   and intersection_fraction(mapped, previous["bbox"]) >= .7 for previous in readings):
                continue
            readings.append({"text": token["text"], "bbox": mapped,
                             "source_id": f"{ref}#token={token_index}", "raw_ref": ref,
                             "pass_id": f"crop_{dpi}", "confidence": None})
    (directory / "mapping.json").write_text(json.dumps({"attempts": attempts, "observations": readings},
                                                       ensure_ascii=False, indent=2), encoding="utf-8")
    return readings, attempts


def table_rotation(tokens: list[dict], page: Page) -> int:
    if header_band(tokens):
        return 0
    matrix = pymupdf.Matrix(0, -1, 1, 0, 0, page.width)
    rotated = [{**token, "bbox": list(pymupdf.Rect(token["bbox"]) * matrix)} for token in tokens]
    return 90 if header_band(rotated) else 0


def refine_table(document: DrawingDocument, page: Page, region: dict[str, Any], *,
                 endpoint: str, artifact: Path,
                 read_region: Callable | None = None) -> dict[str, Any]:
    tokens = [token for token in observations(document, page)
              if intersection_fraction(token["bbox"], region["source_box"]) >= .8]
    rotation = table_rotation(tokens, page)
    if not rotation:
        return _refine_table(document, page, region, tokens=tokens, endpoint=endpoint,
                             artifact=artifact, read_region=read_region)
    matrix = pymupdf.Matrix(0, -1, 1, 0, 0, page.width)
    oriented_page = page.model_copy(update={
        "width": page.height, "height": page.width,
        "pixel_to_pdf_matrix": list(~matrix * pymupdf.Matrix(*page.pixel_to_pdf_matrix)),
    })
    oriented_tokens = [{**token, "bbox": list(pymupdf.Rect(token["bbox"]) * matrix)} for token in tokens]
    result = _refine_table(document, oriented_page,
        {**region, "source_box": list(pymupdf.Rect(region["source_box"]) * matrix)},
        tokens=oriented_tokens, endpoint=endpoint, artifact=artifact, rotation=rotation,
        read_region=read_region)

    def restore(value: Any, key: str = "") -> Any:
        if key in {"bbox", "source_box", "focus_box", "table_box", "anchor_box"} and isinstance(value, (list, tuple)) and len(value) == 4:
            return list(pymupdf.Rect(value) * ~matrix)
        if key == "field_boxes" and isinstance(value, dict):
            return {name: restore(box, "source_box") for name, box in value.items()}
        if isinstance(value, dict):
            return {name: restore(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [restore(item) for item in value]
        return value

    return restore(result)


def _refine_table(document: DrawingDocument, page: Page, region: dict[str, Any], *,
                  tokens: list[dict[str, Any]],
                  endpoint: str, artifact: Path, rotation: int = 0,
                  read_region: Callable | None = None) -> dict[str, Any]:
    read_region = read_region or read_source_region
    source = Path(document.source_path)
    table_id, box = region["id"], list(region["source_box"])
    band = header_band(tokens)
    result: dict[str, Any] = {"id": table_id, "source_box": box, "page": page.page_number,
                              "rows": [], "attempts": [], "review_reasons": []}
    last_attempt = 1
    if not band:
        for last_attempt, dpi in enumerate((400, 600), start=2):
            readings, attempts = read_region(source, page, box, dpi=dpi, endpoint=endpoint,
                directory=artifact / "region_reads" / table_id / f"header-a{last_attempt}", artifact=artifact,
                rotation=rotation)
            result["attempts"].extend({**item, "attempt": last_attempt, "purpose": "header_recovery"} for item in attempts)
            tokens.extend(readings)
            band = header_band(tokens)
            if band:
                break
    if not band:
        result["review_reasons"] = ["part_table_header_unresolved"]
        return result
    # Layout blocks often include the title block below a bottom-header parts list.
    header_center = sum((token["bbox"][1] + token["bbox"][3]) / 2 for token in band) / len(band)
    direction = table_data_direction(tokens, band)
    if not direction:
        result["review_reasons"] = ["table_data_direction_unresolved"]
        return result
    if direction < 0:
        box[3] = min(box[3], max(token["bbox"][3] for token in band) + 8)
    else:
        box[1] = max(box[1], min(token["bbox"][1] for token in band) - 8)
    box = [max(0, box[0] - 6), max(0, box[1] - 6),
           min(page.width, box[2] + 6), min(page.height, box[3] + 6)]
    crop = render_source_crop(source, page=page.page_number, box=box,
                              page_size=[page.width, page.height],
                              pixel_to_pdf=page.pixel_to_pdf_matrix, dpi=400, rotation=rotation)
    xs, ys = ruled_grid(crop.image)
    if len(xs) < 4 or len(ys) < 3:
        result["review_reasons"] = ["table_ruling_unresolved"]
        return result
    header_row = next(((y0, y1) for y0, y1 in zip(ys, ys[1:])
                       if crop.page_box([xs[0], y0, xs[-1], y1])[1] <= header_center
                       <= crop.page_box([xs[0], y0, xs[-1], y1])[3]), None)
    columns = []
    for index, (x0, x1) in enumerate(zip(xs, xs[1:])):
        column_box = crop.page_box([x0, ys[0], x1, ys[-1]])
        labels = [token for token in band if column_box[0] <= (token["bbox"][0] + token["bbox"][2]) / 2 <= column_box[2]]
        keys = {field_key(token["text"]) for token in labels}
        key = next(iter(keys)) if len(keys) == 1 else None
        if key is None and header_row is not None:
            y0, y1 = header_row
            header_box = crop.page_box([x0, y0, x1, y1])
            if cell_has_ink(crop.image, [x0, y0, x1, y1]):
                # A merged header such as 品牌备注 cannot identify either physical column.
                for attempt, dpi in enumerate((400, 600), start=2):
                    if attempt <= last_attempt:
                        continue
                    readings, attempts = read_region(source, page, header_box, dpi=dpi,
                        endpoint=endpoint, artifact=artifact, rotation=rotation,
                        directory=artifact / "region_reads" / table_id / f"header-c{index}-a{attempt}")
                    result["attempts"].extend({**item, "attempt": attempt,
                        "purpose": "header_column_recovery", "column": index} for item in attempts)
                    if readings:
                        labels = readings
                        key = field_key(_compose(readings))
                        if key:
                            break
                if key is None:
                    result["review_reasons"].append("table_columns_unresolved")
        columns.append({"key": key or f"column_{index}",
                        "label": _compose(labels), "bbox": column_box,
                        "header_evidence": labels})
    known = [column["key"] for column in columns if not column["key"].startswith("column_")]
    if not {"item_no", "part_code"}.issubset(known) or len(known) != len(set(known)):
        result["review_reasons"] = ["table_columns_unresolved"]
        return result
    rows = []
    for row_index, (y0, y1) in enumerate(zip(ys, ys[1:])):
        row_box = crop.page_box([xs[0], y0, xs[-1], y1])
        if row_box[1] <= header_center <= row_box[3]:
            continue
        cells = []
        for column_index, (x0, x1) in enumerate(zip(xs, xs[1:])):
            cell_box = crop.page_box([x0, y0, x1, y1])
            local_tokens = [token for token in tokens if intersection_fraction(token["bbox"], cell_box) >= .8]
            cells.append({"column_index": column_index, "bbox": cell_box, "text": "",
                          "observations": local_tokens, "blank_evidence": not cell_has_ink(crop.image, [x0, y0, x1, y1])})
        rows.append({"row_index": row_index, "bbox": row_box, "cells": cells})
    physical_box = crop.page_box([xs[0], ys[0], xs[-1], ys[-1]])

    def structured(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return structure_part_table(table_id=table_id, page=page.page_number,
            source_sha256=document.source_sha256, image_size=[page.width, page.height],
            table_bbox=physical_box, columns=columns, rows=selected_rows, revision="region_understanding_v1")

    initial = structured(rows)
    for row_index, row in enumerate(rows):
        fields = {**initial[row_index]["field_evidence"], **initial[row_index].get("extra_fields", {})}
        for cell in row["cells"]:
            key = columns[cell["column_index"]]["key"]
            field = fields.get(key)
            if field is not None and field["status"] in {"recognized", "blank"}:
                continue
            for attempt, dpi in enumerate((400, 600), start=2):
                if attempt <= last_attempt:
                    continue
                directory = artifact / "region_reads" / table_id / f"r{row['row_index']}-c{cell['column_index']}-a{attempt}"
                readings, attempts = read_region(source, page, cell["bbox"], dpi=dpi,
                    endpoint=endpoint, directory=directory, artifact=artifact, rotation=rotation)
                result["attempts"].extend({**item, "attempt": attempt, "row": row["row_index"],
                                            "column": key} for item in attempts)
                cell["observations"].extend(readings)
                updated = structured([row])[0]
                state = {**updated["field_evidence"], **updated.get("extra_fields", {})}.get(key, {})
                if state.get("status") == "recognized":
                    break
    result.update(rows=structured(rows), source_box=physical_box,
                  initial_rows=initial, columns=columns)
    for row in result["rows"]:
        for field in [*row["field_evidence"].values(), *row.get("extra_fields", {}).values()]:
            field.update(source_sha256=document.source_sha256, page=page.page_number,
                         relation_confidence=None, scope_status="table_cell")
            if field.get("source_box"):
                field["pdf_box"] = list(pymupdf.Rect(field["source_box"]) * pymupdf.Matrix(*page.pixel_to_pdf_matrix))
            if field.get("value") and not field.get("raw_text"):
                field["raw_text"] = field["value"]
    values = [row.get("item_no") for row in result["rows"]]
    numbers = sorted(int(value) for value in values if value and value.isascii() and value.isdecimal())
    if numbers and any(right - left != 1 for left, right in zip(numbers, numbers[1:])):
        result["review_reasons"].append("item_sequence_requires_review")
    if any(row["needs_review"] for row in result["rows"]):
        result["review_reasons"].append("table_fields_require_review")
    result["review_reasons"] = list(dict.fromkeys(result["review_reasons"]))
    return result
