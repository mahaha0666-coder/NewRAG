"""Source-scoped child BOMs and notes, without inheriting physical ownership.

Column roles require a literal header in the same connected native table grid.
Values remain strings; native preview cells are not mother-drawing coordinates.
"""

from __future__ import annotations

from copy import deepcopy
import re

from ezdxf.tools.text import plain_mtext

from research.dwg_route_validation_20261015.native_tdm_cells import _near, build_tdm_cells
from research.dwg_route_validation_20261015.native_tdm_previews import build_tdm_previews

HEADER_FIELDS = {"序号": "sequence", "代号": "drawing_number", "名称": "name", "数量": "quantity",
                 "规格": "specification", "材料": "material", "品牌": "brand", "备注": "remark"}
REQUIRED_HEADERS = {"sequence", "drawing_number", "name", "quantity"}


def _same_grid(left: dict, right: dict) -> bool:
    a, b = left["band"], right["band"]
    return (len(a["x_boundaries"]) == len(b["x_boundaries"]) and _near(a["z"], b["z"])
            and all(_near(x, y) for x, y in zip(a["x_boundaries"], b["x_boundaries"], strict=True)))


def _header_roles(row: dict) -> list[str | None] | None:
    roles = [HEADER_FIELDS.get(re.sub(r"\s+", "", "".join(
        candidate["literal"]["raw_text"] for candidate in cell["literal_candidates"])))
        for cell in row["cells"]]
    known = [role for role in roles if role is not None]
    return roles if REQUIRED_HEADERS <= set(known) and len(known) == len(set(known)) else None


def bind_bom_rows(previews: dict, cells: dict) -> dict:
    """Bind qualified rows using their own finite cells and connected header grid."""
    if (cells["input_record_sha256"] != previews["record_sha256"]
            or any(cells[key] != previews[key] for key in ("source_sha256", "response_sha256"))):
        raise ValueError("BOM cells and literal previews describe different native evidence")
    classes = {row["source_ref"]["json_pointer"]: row for row in previews["objects"]}
    bands, unparsed = [], []
    for obj in cells["objects"]:
        declaration = classes[obj["source_ref"]["json_pointer"]]
        if declaration["declared_class"] != "TDMBOMITEM":
            continue
        for preview in obj["previews"]:
            if preview["status"] != "native_rectangle_candidates":
                unparsed.append({"source_ref": obj["source_ref"], "preview": preview})
                continue
            by_id = {cell["cell_id"]: cell for cell in preview["closed_cells"]}
            for band in preview["row_bands"]:
                row = {"source_ref": obj["source_ref"], "preview_sha256": preview["preview_sha256"],
                       "preview_ordinal": preview["occurrence_ordinal"],
                       "coordinate_status": preview["coordinate_status"], "band": band,
                       "cells": [by_id[identifier] for identifier in band["cell_ids"]]}
                row["header_roles"] = _header_roles(row)
                bands.append(row)
    # Grid connectivity uses actual touching row edges, never an inferred row height.
    components, remaining = [], set(range(len(bands)))
    while remaining:
        component, pending = set(), [min(remaining)]
        while pending:
            index = pending.pop()
            if index not in remaining:
                continue
            remaining.remove(index)
            component.add(index)
            a = bands[index]
            pending.extend(other for other in remaining if _same_grid(a, bands[other])
                           and any(_near(x, y) for x, y in ((a["band"]["y"][0], bands[other]["band"]["y"][1]),
                                                          (a["band"]["y"][1], bands[other]["band"]["y"][0]))))
        components.append(sorted(component))
    rows, headers = [], []
    for component in components:
        candidates = [bands[index] for index in component if bands[index]["header_roles"] is not None]
        headers.extend(candidates)
        for index in component:
            original = bands[index]
            if original["header_roles"] is not None:
                continue
            row = deepcopy(original)
            row.pop("header_roles")
            ref = row["source_ref"]
            row["id"] = (f"child-bom:{ref['source_sha256']}:{ref['native_handle']}:"
                         f"{row['preview_ordinal']}:{row['band']['band_id']}")
            row["fields"], row["field_evidence"] = {}, {}
            row["header_binding_status"] = ("connected_literal_header_candidate" if len(candidates) == 1
                                            else "missing_connected_header" if not candidates else "ambiguous_connected_headers")
            if len(candidates) == 1:
                header = candidates[0]
                for ordinal, (cell, role) in enumerate(zip(row["cells"], header["header_roles"], strict=True)):
                    if role is None:
                        continue
                    # No numeric conversion, whitespace correction or guessed missing value.
                    row["fields"][role] = "\n".join(item["literal"]["raw_text"] for item in cell["literal_candidates"])
                    row["field_evidence"][role] = {"column_ordinal": ordinal, "value_cell_id": cell["cell_id"],
                        "header_cell": header["cells"][ordinal], "header_source_ref": header["source_ref"],
                        "header_preview_sha256": header["preview_sha256"]}
            row["physical_owner_confirmed"] = False
            rows.append(row)
    return {"rows": rows, "headers": headers, "unparsed_objects": unparsed,
            "preview_record_sha256": previews["record_sha256"], "cell_record_sha256": cells["record_sha256"],
            "summary": {"rows": len(rows), "mapped_rows": sum(bool(row["fields"]) for row in rows),
                        "connected_header_grids": len(headers), "unparsed_objects": len(unparsed)}}


def technical_notes(texts: list[dict]) -> list[dict]:
    """Keep original MTEXT and each numbered clause; do not fabricate clause boxes."""
    result = []
    for text in texts:
        if "技术要求" not in text["raw_text"]:
            continue
        readable = plain_mtext(text["raw_text"])
        clauses = [{"line_ordinal": index, "text": line.strip()}
                   for index, line in enumerate(readable.splitlines())
                   if re.match(r"\s*\d+[.．]", line)]
        result.append({**text, "plain_text": readable, "clauses": clauses,
                       "coordinate_status": "child_native_MTEXT_origin_only_no_clause_boxes",
                       "scope": "source_child_document_only", "parent_parameters_inherited": False})
    return result


def build_structure(raw: dict, *, source_sha256: str, response_sha256: str, texts: list[dict]) -> dict:
    previews = build_tdm_previews(raw, source_sha256=source_sha256, response_sha256=response_sha256)
    bom = bind_bom_rows(previews, build_tdm_cells(previews))
    notes = technical_notes(texts)
    return {"schema": "native_child_structure.v1", "bom": bom, "technical_notes": notes,
            "summary": {**bom["summary"], "technical_note_blocks": len(notes),
                        "numbered_technical_clauses": sum(len(note["clauses"]) for note in notes)},
            "policy": {"source_scope": "child_document", "parent_parameters_inherited": False,
                       "table_preview_coordinates_are_mother_WCS": False, "numeric_values_cast": False,
                       "physical_ownership_confirmed": False, "complete_route_generated": False}}
