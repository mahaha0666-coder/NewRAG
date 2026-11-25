"""Project native callout chains and literal BOM candidates without part claims.

Only an unchanged source-bound preview/cell pair is consumed. Endpoint topology,
parallelism and finite projection supply candidates, never arrows or ownership.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from typing import Any

from research.dwg_route_validation_20261015.native_attribute_rows import _digest
from research.dwg_route_validation_20261015.native_tdm_cells import (
    ABS_TOLERANCE, NATIVE_COORDINATES, REL_TOLERANCE, _frame_ref, _inside, _near, _point,
)

SCHEMA = "native_tdm_callouts.v1"
PARALLEL_TOLERANCE = 1e-10


def _unchanged(record: dict, schema: str) -> None:
    if (not isinstance(record, dict) or record.get("schema") != schema
            or record.get("record_sha256") != _digest({k: v for k, v in record.items() if k != "record_sha256"})
            or record.get("eligible_for_route_training") is not False):
        raise ValueError(f"An unchanged {schema} source record is required")


def _candidate_class(row: dict, name: str) -> bool:
    return row.get("declared_class") == name or any(
        occurrence.get("raw_value") == name for ref in row.get("class_refs", [])
        for occurrence in ref.get("dxfname_evidence", {}).get("occurrences", []))


def _sub(a: list, b: list) -> list:
    return [x - y for x, y in zip(a, b, strict=True)]


def _dot(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _same(a: list, b: list) -> bool:
    return all(_near(x, y) for x, y in zip(a, b, strict=True))


def _qualify(row: dict, preview: dict, name: str) -> list[str]:
    reasons = list(row.get("reasons", []))
    if row.get("declared_class") != name or row.get("status") != "native_preview_evidence":
        reasons.append("native_class_or_object_not_qualified")
    if (preview.get("status") != "native_preview_evidence"
            or preview.get("coordinate_status") != NATIVE_COORDINATES
            or preview.get("source_binding_status") != "actual_type_class_join"):
        reasons.append("native_preview_coordinates_or_source_not_qualified")
    return reasons


def _inputs(previews: dict, cells: dict) -> dict:
    _unchanged(previews, "native_tdm_previews.v1")
    _unchanged(cells, "native_tdm_cells.v1")
    if cells.get("input_record_sha256") != previews["record_sha256"]:
        raise ValueError("Cell record does not consume this exact preview record")
    for field in ("source_sha256", "response_sha256"):
        value = previews.get(field)
        if (not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                or cells.get(field) != value):
            raise ValueError("Native source and response pins disagree")
    by_pointer = {}
    for row in cells["objects"]:
        pointer = row["source_ref"]["json_pointer"]
        if pointer in by_pointer:
            raise ValueError("Duplicate cell source pointer")
        by_pointer[pointer] = row
    seen = set()
    for row in previews["objects"]:
        ref = row["source_ref"]
        pointer = ref["json_pointer"]
        if (pointer in seen or pointer not in by_pointer or by_pointer[pointer]["source_ref"] != ref
                or any(ref[field] != previews[field] for field in ("source_sha256", "response_sha256"))):
            raise ValueError("Native object source binding disagrees")
        seen.add(pointer)
        cell_previews = by_pointer[pointer]["previews"]
        if len(cell_previews) != len(row["preview_candidates"]):
            raise ValueError("Cell preview occurrence count disagrees")
        for preview, cell_preview in zip(row["preview_candidates"], cell_previews, strict=True):
            if (preview["source_pointer"] != pointer + "/preview" or any(
                    preview[key] != cell_preview[key] for key in (
                        "source_pointer", "occurrence_ordinal", "preview_sha256"))):
                raise ValueError("Native preview source occurrence disagrees")
    if seen != set(by_pointer):
        raise ValueError("Cell record contains another native source object")
    return by_pointer


def _bom_literals(previews: dict, cell_index: dict) -> list[dict]:
    result = []
    for row in previews["objects"]:
        if not _candidate_class(row, "TDMBOMITEM"):
            continue
        cell_previews = cell_index[row["source_ref"]["json_pointer"]]["previews"]
        for preview, grid in zip(row["preview_candidates"], cell_previews, strict=True):
            frames = {frame["index"]: frame for frame in preview["frames"]}
            for text in preview["texts"]:
                frame = frames.get(text["frame_index"])
                reference = _frame_ref(preview, frame) if frame else None
                locations = []
                for cell in grid["closed_cells"]:
                    if not any(candidate["literal"] == text and candidate["frame_ref"] == reference
                               for candidate in cell["literal_candidates"]):
                        continue
                    bands = [{"band_id": band["band_id"], "row_band_y": deepcopy(band["y"]),
                              "column_ordinal": band["cell_ids"].index(cell["cell_id"]),
                              "column_ordinal_origin": 0}
                             for band in grid["row_bands"] if cell["cell_id"] in band["cell_ids"]]
                    locations.append({"cell_id": cell["cell_id"], "x": deepcopy(cell["x"]),
                                      "y": deepcopy(cell["y"]), "z": cell["z"], "row_bands": bands,
                                      "field_role": "unknown"})
                result.append({"source_ref": deepcopy(row["source_ref"]),
                               "class_refs": deepcopy(row["class_refs"]), "frame_ref": reference,
                               "literal": deepcopy(text), "cell_locations": locations,
                               "source_reasons": _qualify(row, preview, "TDMBOMITEM"),
                               "field_role": "unknown", "physical_binding_confirmed": False})
    return result


def _topology(preview: dict, reasons: list[str]) -> dict:
    result = {"status": "unknown", "reasons": reasons, "shared_endpoint_candidates": [],
              "free_endpoints": [], "support_candidates": []}
    geometry, texts = preview["geometry"], preview["texts"]
    frames = {frame["index"]: frame for frame in preview["frames"]}
    if (len(geometry) != 2 or len(texts) != 1 or len(frames) != len(preview["frames"])
            or any(item["kind"] != "POLYLINE" or item.get("vertex_count") != 2
                   or len(item.get("native_vertices", [])) != 2 for item in geometry)):
        reasons.append("requires_exactly_two_two_vertex_POLYLINE_and_one_TEXT2")
        return result
    text = texts[0]
    items = geometry + [text]
    if (len({item["frame_index"] for item in items}) != 3
            or any(item["frame_index"] not in frames or frames[item["frame_index"]]["opcode"] != opcode
                   or item.get("coordinate_status") != NATIVE_COORDINATES
                   for item, opcode in zip(items, (6, 6, 11), strict=True))):
        reasons.append("native_geometry_or_text_frame_not_qualified")
        return result
    vertices = [item["native_vertices"] for item in geometry]
    points = vertices[0] + vertices[1] + [text.get("native_start_point")]
    direction = text.get("native_direction")
    if (not all(_point(point) for point in points) or not _point(direction)
            or text.get("native_normal") != [0., 0., 1.] or not _near(direction[2], 0.)
            or not all(_near(point[2], points[0][2]) for point in points)
            or _dot(direction, direction) <= ABS_TOLERANCE ** 2):
        reasons.append("nonfinite_non_positive_Z_nonplanar_or_zero_direction")
        return result
    if any(_same(*segment) for segment in vertices):
        reasons.append("degenerate_native_segment")
        return result
    shared = [(i, j) for i in (0, 1) for j in (0, 1) if _same(vertices[0][i], vertices[1][j])]
    result["shared_endpoint_candidates"] = [
        {"endpoint_indices": [i, j], "actual_native_points": [deepcopy(vertices[0][i]), deepcopy(vertices[1][j])]}
        for i, j in shared]
    if len(shared) != 1:
        reasons.append("no_unique_shared_native_endpoint")
        return result
    shared_indices = shared[0]
    free = [{"point": deepcopy(vertices[i][1 - joint]), "endpoint_index": 1 - joint,
             "frame_ref": _frame_ref(preview, frames[geometry[i]["frame_index"]])}
            for i, joint in enumerate(shared_indices)]
    result["free_endpoints"] = free
    if _same(free[0]["point"], free[1]["point"]):
        reasons.append("chain_is_closed_or_overlapping_endpoints")
        return result
    if reasons:
        return result
    for index, (a, b) in enumerate(vertices):
        vector = _sub(b, a)
        length2 = _dot(vector, vector)
        cross = [vector[1] * direction[2] - vector[2] * direction[1],
                 vector[2] * direction[0] - vector[0] * direction[2],
                 vector[0] * direction[1] - vector[1] * direction[0]]
        cross2 = _dot(cross, cross)
        scale2 = length2 * _dot(direction, direction)
        if not all(math.isfinite(value) for value in (length2, cross2, scale2)):
            reasons.append("nonfinite_parallelism_intermediate")
            result["support_candidates"] = []
            return result
        parallel = math.sqrt(cross2) <= PARALLEL_TOLERANCE * math.sqrt(scale2)
        parameter = _dot(_sub(text["native_start_point"], a), vector) / length2
        if not math.isfinite(parameter):
            reasons.append("nonfinite_projection_intermediate")
            result["support_candidates"] = []
            return result
        if not parallel or not _inside(parameter, 0., 1.):
            continue
        projection = [x + parameter * v for x, v in zip(a, vector, strict=True)]
        result["support_candidates"].append({"frame_ref": _frame_ref(preview, frames[geometry[index]["frame_index"]]),
            "native_vertices": deepcopy(vertices[index]), "projection_parameter": parameter,
            "native_insertion_projection": projection,
            "insertion_offset_vector": _sub(text["native_start_point"], projection),
            "non_support_free_endpoint": deepcopy(free[1 - index]["point"]),
            "support_free_endpoint": deepcopy(free[index]["point"]),
            "non_support_free_endpoint_ref": deepcopy(free[1 - index]),
            "support_free_endpoint_ref": deepcopy(free[index]),
            "non_text_support_free_endpoint_not_arrowtip": deepcopy(free[1 - index]),
            "status": "parallel_finite_projection_candidate_not_semantic_binding"})
    count = len(result["support_candidates"])
    result["status"] = ("unique_open_chain_support_candidate" if count == 1 else
                        "ambiguous_support_candidates" if count > 1 else "no_support_candidate")
    return result


def build_callout_topology(previews: dict[str, Any], cells: dict[str, Any]) -> dict[str, Any]:
    """Keep every exact-string BOM candidate; never infer sequence/quantity roles."""
    cell_index = _inputs(previews, cells)
    bom_literals = _bom_literals(previews, cell_index)
    callouts, objects = [], []
    for row in previews["objects"]:
        if not _candidate_class(row, "TDMLABEL"):
            continue
        object_result = {"source_ref": deepcopy(row["source_ref"]),
                         "class_refs": deepcopy(row["class_refs"]), "previews": []}
        objects.append(object_result)
        for preview in row["preview_candidates"]:
            frames = {frame["index"]: frame for frame in preview["frames"]}
            result = _topology(preview, _qualify(row, preview, "TDMLABEL"))
            matches = []
            for text in preview["texts"]:
                raw = text.get("raw_text")
                matches.append({"raw_text": raw, "label_frame_ref": _frame_ref(preview, frames[text["frame_index"]])
                                if text["frame_index"] in frames else None,
                                "status": "empty_literal_not_join_evidence" if not isinstance(raw, str) or not raw.strip()
                                else "exact_string_candidates_only",
                                "candidates": [deepcopy(candidate) for candidate in bom_literals
                                               if type(raw) is str and candidate["literal"].get("raw_text") == raw]})
            result.update(source_ref=deepcopy(row["source_ref"]), class_refs=deepcopy(row["class_refs"]),
                          raw_scope=deepcopy(row["raw_scope"]), preview_ref={key: preview[key] for key in (
                              "source_pointer", "occurrence_ordinal", "preview_sha256")},
                          geometry_evidence=deepcopy(preview["geometry"]), literal_evidence=deepcopy(preview["texts"]),
                          frame_refs=[_frame_ref(preview, frame) for frame in preview["frames"]],
                          bom_literal_matches=matches, coordinate_status=NATIVE_COORDINATES if not result["reasons"] else "unknown",
                          WCS_validated=False, arrowtip_confirmed=False, physical_owner="unknown",
                          eligible_for_route_training=False)
            callouts.append(result)
            object_result["previews"].append(result)
        if not row["preview_candidates"]:
            unknown = {"source_ref": deepcopy(row["source_ref"]), "class_refs": deepcopy(row["class_refs"]),
                       "status": "unknown", "reasons": list(row["reasons"]) + ["missing_preview_candidate"],
                       "support_candidates": [], "bom_literal_matches": [], "eligible_for_route_training": False}
            callouts.append(unknown)
            object_result["previews"].append(unknown)
    result = {"schema": SCHEMA, "source_sha256": previews["source_sha256"],
              "response_sha256": previews["response_sha256"], "input_record_sha256": previews["record_sha256"],
              "cells_record_sha256": cells["record_sha256"], "objects": objects,
              "parameters": {"absolute_tolerance": ABS_TOLERANCE, "relative_tolerance": REL_TOLERANCE,
                             "parallel_tolerance": PARALLEL_TOLERANCE, "column_ordinal_origin": 0},
              "summary": {"callout_previews": len(callouts), "status_counts": dict(Counter(c["status"] for c in callouts)),
                          "support_candidates": sum(len(c["support_candidates"]) for c in callouts),
                          "bom_raw_string_candidates": sum(len(m["candidates"]) for c in callouts for m in c["bom_literal_matches"])},
              "WCS_validated": False, "field_roles_validated": False, "physical_ownership_validated": False,
              "eligible_for_route_training": False}
    result["record_sha256"] = _digest(result)
    return result
