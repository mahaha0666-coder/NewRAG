"""Evidence-qualified BOM columns and literal catalog lookup for any drawing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from research.dwg_route_validation_20261015.native_tdm_cells import _near

__all__ = ["load_catalog", "continuation_candidates", "link_rows"]


def load_catalog(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    codes: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for ordinal, row in enumerate(csv.DictReader(stream), start=2):
            code = row["item_code"]
            if not code or not code.isascii() or not code.isdecimal() or code in codes:
                raise ValueError("Catalog requires unique literal numeric material codes")
            codes.add(code)
            if row["deleted"] not in {"True", "False"}:
                raise ValueError("Unrecognized catalog deletion state")
            # Route signatures, labels and operation statistics are not consumed.
            record = {key: row[key] for key in (
                "item_code", "drawing_no", "item_name", "parent_drawing_no", "deleted")}
            record["csv_record_number"] = ordinal
            result.setdefault(row["drawing_no"], []).append(record)
    return result


def continuation_candidates(bom: dict[str, Any]) -> None:
    """Offer a separate hypothesis for translated, identical column geometry.

    Connected-header fields are never overwritten. A remote header remains a
    candidate even when its full column-width signature is uniquely compatible.
    """
    for row in bom["rows"]:
        if row["fields"]:
            continue
        bounds = row["band"]["x_boundaries"]
        signatures: dict[tuple, list[dict]] = {}
        for header in bom["headers"]:
            other = header["band"]["x_boundaries"]
            if (len(bounds) != len(other) or not _near(row["band"]["z"], header["band"]["z"])
                    or not all(_near(a - bounds[0], b - other[0]) for a, b in zip(bounds, other, strict=True))):
                continue
            signatures.setdefault(tuple(header["header_roles"]), []).append(header)
        if len(signatures) != 1:
            row["continuation_status"] = "no_unique_translated_header" if not signatures else "conflicting_translated_headers"
            continue
        roles, headers = next(iter(signatures.items()))
        row["field_candidates"] = {
            role: "\n".join(item["literal"]["raw_text"] for item in cell["literal_candidates"])
            for role, cell in zip(roles, row["cells"], strict=True) if role is not None}
        row["continuation_status"] = "same_sheet_translated_column_signature_candidate"
        row["continuation_evidence"] = [{
            "source_ref": header["source_ref"], "preview_sha256": header["preview_sha256"],
            "x_boundaries": header["band"]["x_boundaries"], "header_roles": header["header_roles"],
            "translation_x": bounds[0] - header["band"]["x_boundaries"][0],
        } for header in headers]


def link_rows(bom: dict[str, Any], catalog: dict[str, list[dict[str, Any]]],
              documents: dict[str, dict[str, dict[str, str]]]) -> list[dict[str, Any]]:
    result = []
    for row in bom["rows"]:
        fields = row["fields"] or row.get("field_candidates", {})
        drawing = fields.get("drawing_number", "")
        basis = row["header_binding_status"] if row["fields"] else row.get("continuation_status", "unknown")
        candidates = [drawing] if drawing else sorted({
            item["literal"]["raw_text"] for cell in row["cells"] for item in cell["literal_candidates"]
            if item["literal"]["raw_text"] in catalog})
        if not drawing and candidates:
            basis = "literal_catalog_hit_without_field_binding"
        matches = [{**record, "matched_drawing_literal": value,
                    "documents": documents.get(record["item_code"], {}),
                    "route_applicability": "unverified"}
                   for value in candidates for record in catalog.get(value, [])]
        result.append({
            "bom_id": row["id"], "source_ref": row["source_ref"],
            "coordinate_status": row["coordinate_status"], "band": row["band"],
            "fields": row["fields"], "field_candidates": row.get("field_candidates", {}),
            "field_binding_status": basis, "candidate_drawing_literals": candidates,
            "raw_cells": [{"cell_id": cell["cell_id"], "x": cell["x"], "y": cell["y"],
                           "literals": [{"raw_text": lit["literal"]["raw_text"], "frame_ref": lit["frame_ref"]}
                                        for lit in cell["literal_candidates"]]} for cell in row["cells"]],
            "catalog_matches": matches, "physical_owner_confirmed": False,
            "relation_confidence": None,
        })
    return result
