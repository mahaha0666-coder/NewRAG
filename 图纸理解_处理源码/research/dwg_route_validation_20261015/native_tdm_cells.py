"""Recover same-preview rectangular boundaries and literal insertion candidates.

Only authenticated native preview coordinates are used. A cell is a rectangle
whose four sides have finite native segment coverage and whose interior has no
other native grid segment. It supplies neither a glyph box nor a field name.
The caller remains responsible for verifying the original source bytes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
import math
from typing import Any

from research.dwg_route_validation_20261015.native_attribute_rows import _digest

SCHEMA = "native_tdm_cells.v1"
NATIVE_COORDINATES = "same_object_native_preview_only_not_WCS"
ABS_TOLERANCE = 1e-8
REL_TOLERANCE = 1e-11


def _near(a: float, b: float) -> bool:
    return abs(a - b) <= ABS_TOLERANCE + REL_TOLERANCE * max(abs(a), abs(b))


def _inside(value: float, lower: float, upper: float) -> bool:
    return lower < value < upper and not _near(value, lower) and not _near(value, upper)


def _point(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 3
            and all(type(x) in (int, float) and math.isfinite(x) for x in value))


def _coordinates(values: list[float]) -> list[float]:
    """Choose actual endpoint values, without averaging or transitive snapping."""
    result = []
    for value in sorted(set(values)):
        if not result or not _near(result[-1], value):
            result.append(value)
    return result


def _frame_ref(preview: dict, frame: dict) -> dict:
    return {"source_pointer": preview["source_pointer"],
            "occurrence_ordinal": preview["occurrence_ordinal"],
            "preview_sha256": preview["preview_sha256"],
            "frame_index": frame["index"], "byte_offset": frame["offset"],
            "byte_size": frame["size"], "opcode": frame["opcode"],
            "frame_sha256": frame["sha256"]}


def _segments(preview: dict, frames: dict[int, dict]) -> tuple[list[dict], list[dict], list[str]]:
    segments, unused, reasons = [], [], []
    for item in preview["geometry"]:
        frame = frames.get(item["frame_index"])
        if frame is None:
            reasons.append("geometry_frame_reference_missing")
            continue
        reference = _frame_ref(preview, frame)
        if item["kind"] not in {"POLYLINE", "POLYLINE_WITH_NORMALS"}:
            unused.append({"frame_ref": reference, "reason": "not_a_rectangular_grid_segment"})
            continue
        expected = 32 if item["kind"] == "POLYLINE_WITH_NORMALS" else 6
        vertices = item.get("native_vertices")
        if (frame["opcode"] != expected
                or item.get("coordinate_status") != NATIVE_COORDINATES
                or not isinstance(vertices, list) or len(vertices) < 2
                or not all(_point(vertex) for vertex in vertices)
                or (expected == 32 and item.get("native_normal") != [0., 0., 1.])):
            reasons.append("unqualified_native_grid_geometry")
            continue
        for index, (a, b) in enumerate(zip(vertices, vertices[1:])):
            ref = dict(reference, segment_index=index)
            if not _near(a[2], b[2]):
                reasons.append("nonplanar_native_grid_segment")
                continue
            dx, dy = _near(a[0], b[0]), _near(a[1], b[1])
            if dx and dy:
                unused.append({"frame_ref": ref, "reason": "degenerate_native_segment"})
            elif not dx and not dy:
                reasons.append("non_axis_aligned_native_grid_segment")
            else:
                axis = 0 if dx else 1  # Fixed coordinate: x for vertical, y for horizontal.
                segments.append({"axis": axis, "fixed": a[axis], "z": a[2],
                                 "lower": min(a[1 - axis], b[1 - axis]),
                                 "upper": max(a[1 - axis], b[1 - axis]),
                                 "frame_ref": ref})
    if segments and any(not _near(segment["z"], segments[0]["z"]) for segment in segments):
        reasons.append("multiple_native_grid_planes")
    return segments, unused, sorted(set(reasons))


def _cover(segments: list[dict], axis: int, fixed: float,
           lower: float, upper: float) -> list[dict] | None:
    candidates = sorted((s for s in segments if s["axis"] == axis and _near(s["fixed"], fixed)
                         and s["upper"] >= lower and s["lower"] <= upper),
                        key=lambda s: (s["lower"], s["upper"], s["frame_ref"]["frame_index"],
                                       s["frame_ref"]["segment_index"]))
    cursor, references = lower, []
    for segment in candidates:
        if segment["lower"] > cursor and not _near(segment["lower"], cursor):
            return None
        if segment["upper"] > cursor:
            references.append(deepcopy(segment["frame_ref"]))
            cursor = segment["upper"]
        if cursor >= upper or _near(cursor, upper):
            return references
    return None


def _interior_segment(segment: dict, x0: float, x1: float, y0: float, y1: float) -> bool:
    fixed0, fixed1, lower, upper = ((x0, x1, y0, y1) if segment["axis"] == 0
                                    else (y0, y1, x0, x1))
    overlap0, overlap1 = max(lower, segment["lower"]), min(upper, segment["upper"])
    return _inside(segment["fixed"], fixed0, fixed1) and overlap1 > overlap0 and not _near(overlap0, overlap1)


def _cells(segments: list[dict]) -> list[dict]:
    # Coordinates include segment ends, so an interrupted side cannot be extended
    # merely because another row has a convenient height or column boundary.
    xs = _coordinates([s["fixed"] for s in segments if s["axis"] == 0]
                      + [p for s in segments if s["axis"] == 1 for p in (s["lower"], s["upper"])])
    ys = _coordinates([s["fixed"] for s in segments if s["axis"] == 1]
                      + [p for s in segments if s["axis"] == 0 for p in (s["lower"], s["upper"])])
    cells = []
    for x0, x1 in combinations(xs, 2):
        for y0, y1 in combinations(ys, 2):
            if any(_interior_segment(segment, x0, x1, y0, y1) for segment in segments):
                continue
            sides = {"left": _cover(segments, 0, x0, y0, y1),
                     "right": _cover(segments, 0, x1, y0, y1),
                     "bottom": _cover(segments, 1, y0, x0, x1),
                     "top": _cover(segments, 1, y1, x0, x1)}
            if any(not references for references in sides.values()):
                continue
            cells.append({"cell_id": f"cell_{len(cells)}", "x": [x0, x1], "y": [y0, y1],
                          "z": segments[0]["z"], "boundary_frame_refs": sides,
                          "status": "closed_native_rectangle_candidate",
                          "literal_candidates": [], "field_role": "unknown",
                          "physical_owner": "unknown"})
    return cells


def _text_candidate(text: dict, preview: dict, frames: dict[int, dict]) -> dict:
    frame = frames.get(text["frame_index"])
    return {"literal": deepcopy(text), "frame_ref": _frame_ref(preview, frame) if frame else None,
            "status": "unassigned", "reason": None}


def _assign_texts(preview: dict, frames: dict[int, dict], cells: list[dict],
                  reasons: list[str]) -> list[dict]:
    unassigned = []
    for text in preview["texts"]:
        candidate = _text_candidate(text, preview, frames)
        point = text.get("native_start_point")
        if reasons:
            candidate["reason"] = "preview_topology_unknown"
        elif (candidate["frame_ref"] is None or candidate["frame_ref"]["opcode"] != 11
              or text.get("coordinate_status") != NATIVE_COORDINATES or not _point(point)):
            candidate["reason"] = "unqualified_native_TEXT2_insertion"
        else:
            hits = [c for c in cells if _near(point[2], c["z"])
                    and _inside(point[0], *c["x"]) and _inside(point[1], *c["y"])]
            if len(hits) == 1:
                candidate.update(status="unique_strict_interior_insertion_candidate", reason=None)
                hits[0]["literal_candidates"].append(candidate)
                continue
            on_boundary = any(_near(point[2], c["z"])
                              and c["x"][0] <= point[0] <= c["x"][1]
                              and c["y"][0] <= point[1] <= c["y"][1]
                              and any(_near(point[axis], edge) for axis, key in ((0, "x"), (1, "y"))
                                      for edge in c[key]) for c in cells)
            candidate["reason"] = ("multiple_cell_interiors" if hits else "on_cell_boundary" if on_boundary
                                   else "no_closed_native_cell" if not cells
                                   else "native_insertion_plane_mismatch" if not any(_near(point[2], c["z"]) for c in cells)
                                   else "outside_closed_cells")
        unassigned.append(candidate)
    return unassigned


def _bands(cells: list[dict]) -> list[dict]:
    candidates = defaultdict(list)
    for cell in cells:
        candidates[(*cell["y"], cell["z"])].append(cell)
    bands = []
    for (y0, y1, z), group in sorted(candidates.items()):
        ordered = sorted(group, key=lambda c: c["x"])
        if any(not _near(a["x"][1], b["x"][0]) for a, b in zip(ordered, ordered[1:])):
            continue
        bands.append({"band_id": f"band_{len(bands)}", "y": [y0, y1], "z": z,
                      "x_boundaries": [ordered[0]["x"][0]] + [c["x"][1] for c in ordered],
                      "cell_ids": [c["cell_id"] for c in ordered],
                      "status": "same_preview_tiled_row_band_candidate"})
    return bands


def _preview_cells(preview: dict, object_reasons: list[str]) -> dict:
    result = {"source_pointer": preview["source_pointer"],
              "sidecar_pointer": preview.get("sidecar_pointer"),
              "occurrence_ordinal": preview["occurrence_ordinal"],
              "preview_sha256": preview["preview_sha256"],
              "coordinate_status": preview["coordinate_status"],
              "closed_cells": [], "row_bands": [], "unassigned_texts": [],
              "unused_geometry": [], "reasons": list(object_reasons), "status": "unknown"}
    frames = {frame["index"]: frame for frame in preview["frames"]}
    if len(frames) != len(preview["frames"]):
        result["reasons"].append("duplicate_preview_frame_index")
    if preview["status"] != "native_preview_evidence" or preview["coordinate_status"] != NATIVE_COORDINATES:
        result["reasons"].append("native_preview_coordinates_not_qualified")
    if result["reasons"]:
        result["reasons"].extend(preview.get("reasons", []))
    else:
        segments, unused, reasons = _segments(preview, frames)
        result["unused_geometry"] = unused
        result["reasons"].extend(reasons)
        if not reasons:
            result["closed_cells"] = _cells(segments) if segments else []
            result["row_bands"] = _bands(result["closed_cells"])
            result["status"] = "native_rectangle_candidates" if result["closed_cells"] else "no_closed_native_cells"
    result["unassigned_texts"] = _assign_texts(preview, frames, result["closed_cells"], result["reasons"])
    result["reasons"] = sorted(set(result["reasons"]))
    return result


def build_tdm_cells(previews: dict[str, Any]) -> dict[str, Any]:
    """Consume an unchanged build_tdm_previews result; never read source files."""
    if (previews.get("schema") != "native_tdm_previews.v1"
            or previews.get("record_sha256") != _digest({k: v for k, v in previews.items() if k != "record_sha256"})
            or previews.get("eligible_for_route_training") is not False):
        raise ValueError("An unchanged native_tdm_previews record is required")
    for field in ("source_sha256", "response_sha256"):
        value = previews.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Original native source and response SHA256 pins are required")
    objects, seen = [], set()
    for row in previews["objects"]:
        ref = row["source_ref"]
        if (ref["source_sha256"] != previews["source_sha256"] or ref["response_sha256"] != previews["response_sha256"]
                or ref["json_pointer"] in seen):
            raise ValueError("Native object source identity disagrees or repeats")
        seen.add(ref["json_pointer"])
        candidates = []
        for preview in row["preview_candidates"]:
            if preview["source_pointer"] != ref["json_pointer"] + "/preview":
                raise ValueError("Preview source pointer disagrees with its native object")
            candidates.append(_preview_cells(preview, row["reasons"]))
        objects.append({"source_ref": deepcopy(ref), "previews": candidates,
                        "physical_owner": "unknown", "eligible_for_route_training": False})
    rows = [p for obj in objects for p in obj["previews"]]
    cells = [c for p in rows for c in p["closed_cells"]]
    result = {"schema": SCHEMA, "source_sha256": previews["source_sha256"],
              "response_sha256": previews["response_sha256"], "input_record_sha256": previews["record_sha256"],
              "parameters": {"absolute_tolerance": ABS_TOLERANCE, "relative_tolerance": REL_TOLERANCE,
                             "coordinate_representative": "actual_endpoint_no_average",
                             "text_test": "strict_interior_native_insertion_only"},
              "objects": objects,
              "summary": {"objects": len(objects), "preview_candidates": len(rows), "closed_cells": len(cells),
                          "assigned_literal_candidates": sum(len(c["literal_candidates"]) for c in cells),
                          "unassigned_literals": sum(len(p["unassigned_texts"]) for p in rows),
                          "preview_status_counts": dict(Counter(p["status"] for p in rows))},
              "WCS_validated": False, "glyph_bounds_validated": False, "field_roles_validated": False,
              "physical_ownership_validated": False, "eligible_for_route_training": False}
    result["record_sha256"] = _digest(result)
    return result
