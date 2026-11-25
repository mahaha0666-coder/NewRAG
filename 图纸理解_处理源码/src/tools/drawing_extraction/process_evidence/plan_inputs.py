"""One DWG input contract: native text/tables and its verified vector drawing."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import fitz

from src.tools.research.artifacts import digest, read_json, write_new_json

from .geometry import render_source
from .native_text import source_records

IMAGE_PIXEL_BUDGET = 2_097_152


def drawing_source(pair: Path, pdf: Path, manifest: dict, source_hash: str) -> dict:
    """Consume a verified pair, never an unrelated file with the same filename."""
    bundle = read_json(pair / "bundle.json")
    if (
        manifest["source"]["sha256"] != source_hash
        or bundle["source"]["sha256"] != source_hash
        or bundle["derived_pdf"]["derived_from_sha256"] != source_hash
        or bundle["derived_pdf"]["sha256"] != digest(pdf)
        or not manifest["automatic_full_extent"]
    ):
        raise ValueError("Drawing does not derive from this DWG")
    return {
        "source_sha256": source_hash,
        "pdf": {"path": str(pdf.resolve()), "sha256": digest(pdf)},
        "bundle": {
            "path": str((pair / "bundle.json").resolve()),
            "sha256": digest(pair / "bundle.json"),
        },
        "manifest": {
            "path": str((pair / "evidence/render_manifest.json").resolve()),
            "sha256": digest(pair / "evidence/render_manifest.json"),
        },
        "page_points": manifest["page_points"],
        "cad_to_pdf_matrix44": manifest["cad_to_pdf_matrix44"],
        "origin": "generated_from_same_dwg",
    }


def verify_drawing(drawing: dict) -> None:
    for key in ("pdf", "bundle", "manifest"):
        item = drawing[key]
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("DWG drawing inputs changed")


def make_packet(facts: dict[str, Any], drawing: dict | None = None) -> dict[str, Any]:
    records = source_records(facts, allow_empty=True)
    source_hash = facts["source"]["sha256"]
    by_id = {row["id"]: row for row in records}
    for dimension in facts.get("dimensions", []):
        record = by_id.get(dimension["id"])
        if record is None and dimension.get("raw_measurement") is not None:
            # An empty override string means "use native measurement", not
            # "no dimension". Do not invent the printed precision or unit.
            record = {
                "id": dimension["id"],
                "kind": "native_measurement",
                "text": json.dumps(
                    {"native_measurement": dimension["raw_measurement"]}
                ),
                "source_ref": dimension["source_ref"],
                "cad_position": dimension.get("cad_coordinates"),
                "scope_status": "unassigned",
            }
            records.append(record)
        if record is not None:
            record["native_context"] = {
                "geometry_ids": dimension.get("geometry_ids", []),
                "part_candidates": dimension.get("part_candidates", []),
                "physical_scope_confirmed": dimension.get("physical_scope_confirmed")
                is True,
                "native_measurement": dimension.get("raw_measurement"),
                "unit": dimension.get("unit"),
                "unit_source": dimension.get("unit_source"),
            }
    if drawing is not None:
        if drawing["source_sha256"] != source_hash:
            raise ValueError("DWG process input mixes original sources")
        verify_drawing(drawing)
        bundle = read_json(Path(drawing["bundle"]["path"]))
        locations = {
            row["source_ref"]["native_handle"]: row["pdf_location"]
            for row in bundle["facts"]
            if row.get("pdf_location", {}).get("status") == "located_in_derived_pdf"
        }
        represented = {row["source_ref"].get("native_handle") for row in records}
        for row in bundle["facts"]:
            # Uninstantiated block templates are not facts about this sheet.
            if row["category"] != "text" or not row.get("scene_admission", {}).get(
                "admitted"
            ):
                continue
            if row["source_ref"].get("native_handle") in represented or not row.get(
                "raw_text"
            ):
                continue
            records.append(
                {
                    "id": row["id"],
                    "kind": "native_text",
                    "text": row["raw_text"],
                    "source_ref": row["source_ref"],
                    "cad_position": row.get("cad_coordinates"),
                    "scope_status": "unassigned",
                }
            )
        for row in bundle["facts"]:
            if (
                row["category"] != "stored_graphics"
                or row["native_kind"] in {"TDMFRAME", "TDMTITLE"}
                or not row.get("scene_admission", {}).get("admitted")
            ):
                continue
            # Preserve the native symbol class alongside its strings. A bare
            # "n=4" loses the fact that it belongs to a welding symbol.
            records.append(
                {
                    "id": row["id"],
                    "kind": "native_symbol",
                    "text": json.dumps(
                        {
                            "native_class": row["native_kind"],
                            "raw_strings": [
                                item["raw_text"]
                                for item in row.get("original_texts", [])
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    "source_ref": row["source_ref"],
                    "cad_position": None,
                    "scope_status": "native_symbol_scope_unresolved",
                }
            )
        for row in records:
            position = locations.get(row["source_ref"].get("native_handle"))
            if position is not None:
                row["pdf_position"] = position
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate process input record")
    if any(row["source_ref"]["source_sha256"] != source_hash for row in records):
        raise ValueError("DWG process input mixes original sources")
    tables = [
        {
            "id": "parts_table",
            "kind": "parts_list",
            "rows": [
                {
                    "evidence_id": row["id"],
                    "fields": row["fields"],
                    "field_candidates": row["field_candidates"],
                    "review_reasons": row.get("review_reasons", []),
                }
                for row in records
                if row["kind"] == "bom_row"
            ],
        }
    ]
    tables = [table for table in tables if table["rows"]]
    return {
        "source_sha256": source_hash,
        "semantic_sha256": facts["semantic_sha256"],
        "records": records,
        "tables": tables,
        "drawing": drawing,
        "images_used": drawing is not None,
        "ocr_used": False,
        "route_labels_used": False,
        "manufacturing_responsibility": "unknown",
        "incoming_stock_state": "unknown",
    }


def render_drawing(packet: dict, output: Path) -> list[dict]:
    """Whole-sheet context plus four 12%-overlap crops from the vector PDF."""
    drawing = packet.get("drawing")
    if drawing is None:
        raise ValueError("同源 DWG 图面尚未生成，请先完成图纸解析。")
    verify_drawing(drawing)
    images = []
    with fitz.open(drawing["pdf"]["path"]) as document:
        if document.needs_pass or len(document) != 1:
            raise ValueError("Expected one unencrypted DWG-derived sheet")
        page = document[0]
        if page.rotation or any(
            abs(a - b) > 0.01
            for a, b in zip(
                [page.rect.width, page.rect.height], drawing["page_points"], strict=True
            )
        ):
            raise ValueError("DWG drawing placement differs")
        w, h = page.rect.width, page.rect.height
        boxes = [
            None,
            [0, 0, 0.56 * w, 0.56 * h],
            [0.44 * w, 0, w, 0.56 * h],
            [0, 0.44 * h, 0.56 * w, h],
            [0.44 * w, 0.44 * h, w, h],
        ]
        for index, box in enumerate(boxes):
            area = fitz.Rect(box or list(page.rect)).get_area()
            dpi = min(400, max(72, int(72 * math.sqrt(IMAGE_PIXEL_BUDGET / area))))
            # Bound raster work before allocating. Original vector PDF is retained.
            if area * (dpi / 72) ** 2 > 16_000_000:
                raise ValueError("Drawing exceeds the current raster work budget")
            image = render_source(
                page,
                output / f"drawing_{index}.png",
                image_id=f"drawing_{index}",
                dpi=dpi,
                clip=box,
                origin="same_dwg_vector_pdf",
            )
            images.append(image)
    write_new_json(output / "images.json", images)
    return images


def model_inputs(packet: dict, images: list[dict]) -> dict:
    """Send structured BOM rows once, not as both JSON strings and a table."""
    texts = []
    for row in packet["records"]:
        if row["kind"] == "bom_row":
            continue
        context = row.get("native_context", {})
        position = row.get("pdf_position")
        # Define metadata keys once. Values and every source record remain
        # present; repeated JSON keys must not crowd out the model's answer.
        texts.append(
            [
                row["id"],
                row["kind"],
                row["text"],
                [round(value, 2) for value in position["pdf_bounds_pt"]]
                if position
                else None,
                context.get("native_measurement"),
                context.get("unit"),
                context.get("unit_source"),
                row["scope_status"],
                context.get("physical_scope_confirmed", False),
                row.get("field"),
            ]
        )
    return {
        "texts_and_dimensions": {
            "columns": [
                "evidence_id",
                "kind",
                "text",
                "bbox_pdf_page1",
                "native_measurement",
                "unit",
                "unit_source",
                "scope_status",
                "physical_scope_confirmed",
                "field",
            ],
            "rows": texts,
        },
        "tables": packet["tables"],
        "images": [
            {
                key: image[key]
                for key in (
                    "image_id",
                    "page_number",
                    "width",
                    "height",
                    "crop_bbox_pdf",
                    "dpi",
                )
            }
            for image in images
        ],
        "manufacturing_responsibility": packet["manufacturing_responsibility"],
        "incoming_stock_state": packet["incoming_stock_state"],
    }


def input_summary(packet: dict, images: list[dict]) -> dict:
    kinds = Counter(row["kind"] for row in packet["records"])
    return {
        "native_records": len(packet["records"]),
        "table_rows": kinds["bom_row"],
        "tables": len(packet["tables"]),
        "images": len(images),
        "native_measurements": kinds["native_measurement"],
        "native_symbols": kinds["native_symbol"],
    }
