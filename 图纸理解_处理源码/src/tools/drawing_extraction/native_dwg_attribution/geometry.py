"""Verified native occurrences and conservative finite contacts, without proximity joins."""

from __future__ import annotations

import math

from research.dwg_pdf_visualization_20261031.geometry_targets import IDENTITY, build_targets, native_loops
from research.dwg_route_validation_20261015.native_balloon_probe import line_catalog
from research.dwg_route_validation_20261015.native_insert_transform_probe import compare_case
from research.dwg_route_validation_20261015.native_insert_transforms import build_instance_transforms, transform_point
from research.dwg_route_validation_20261015.native_tdm_contact_probe import diagnose_case as line_contacts
from research.dwg_route_validation_20261015.native_tdm_curve_probe import diagnose_case as curve_contacts
from src.tools.drawing_extraction.native_dwg_batch.artifacts import fingerprint

from .contours import classify_layers, complete_contour_candidates

__all__ = ["geometry_id", "collect_geometry"]


def geometry_id(kind: str, ref: dict, path: list) -> str:
    return "geometry:" + fingerprint([kind, ref, path])[:24]


def _curve_points(curve: dict) -> list[list[float]]:
    start = curve.get("start_angle", 0.) if curve["native_kind"] == "ARC" else 0.
    sweep = ((curve["end_angle"] - start) % math.tau) if curve["native_kind"] == "ARC" else math.tau
    count = max(12, math.ceil(sweep / (math.pi / 64)))
    center, radius = curve["native_center"], curve["native_radius"]
    return [transform_point(curve["matrix"], [center[0] + radius * math.cos(start + sweep * i / count),
                                             center[1] + radius * math.sin(start + sweep * i / count), center[2]])
            for i in range(count + 1)]


def collect_geometry(raw: dict, graph: dict, tables: dict, parts: list[dict], callouts: list[dict]) -> tuple[list[dict], dict]:
    """Use the frozen LINE/ARC solvers, preserving instance path identity."""
    if graph["truncation"]["truncated"]:
        raise ValueError("Native instance graph truncated; geometry attribution refused")
    transforms = build_instance_transforms(graph)
    comparison = compare_case(graph, transforms)
    lines = line_catalog(graph, {"comparison": comparison})
    geometry = []
    for line in lines:
        if line["start"] != line["end"]:
            geometry.append({"id": geometry_id("LINE", line["source_ref"], line["path_edge_ids"]),
                             "kind": "LINE", **line, "points": [line["start"], line["end"]],
                             "coordinate_status": "native_modelspace_or_verified_INSERT_affine"})
    if tables.get("callouts", {}).get("objects"):
        _, line_detail = line_contacts({"evidence": tables["previews"], "cells": tables["cells"]}, graph)
        curve_summary, curve_detail = curve_contacts({"evidence": line_detail}, graph)
        curves = curve_detail["curve_catalog"]
        contact_checks = curve_detail["endpoint_checks"]
    else:
        from research.dwg_route_validation_20261015.native_curve_contacts import curve_catalog
        from research.dwg_route_validation_20261015.native_tdm_curve_probe import independent_samples

        curves = curve_catalog(graph, transforms)
        reference = independent_samples(graph, curves)
        curve_summary = {"curve_catalog": curves["summary"], "independent_curve_samples": len(reference["samples"]),
                         "max_independent_sample_delta": reference["max_absolute_delta"]}
        contact_checks = []
    for curve in curves["objects"]:
        if curve["status"] == "accepted":
            geometry.append({"id": geometry_id(curve["native_kind"], curve["source_ref"], curve["path_edge_ids"]),
                             "kind": curve["native_kind"], **curve, "points": _curve_points(curve),
                             "coordinate_status": "native_modelspace_or_verified_INSERT_affine",
                             "display_tessellation_is_contact_evidence": False})
    by_ref = {item["source_ref"]["json_pointer"]: item for item in callouts}
    candidates = {part["id"]: {"id": part["id"], "callouts": []} for part in parts}
    for check in contact_checks:
        callout = by_ref[check["callout_source_ref"]["json_pointer"]]
        callout.setdefault("endpoint_checks", []).append(check)
        if check["endpoint_role"] != "non_support_free_endpoint":
            continue
        ids = [geometry_id("LINE", row["line"]["source_ref"], row["line"]["path_edge_ids"])
               for row in check["baseline_line_contacts"]]
        ids += [geometry_id(curves["objects"][row["curve_ordinal"]]["native_kind"], row["source_ref"], row["path_edge_ids"])
                for row in check["curve_contacts"]]
        callout["geometry_ids"] = sorted(set(callout["geometry_ids"] + ids))
        # Multiple topology hypotheses must not be silently merged into one tip.
        unique_topology = len(callout["previews"]) == 1 and len(callout["previews"][0]["support_candidates"]) == 1
        if len(callout["part_candidates"]) != 1 or not unique_topology:
            continue
        part_id = callout["part_candidates"][0]["part_id"]
        candidates[part_id]["callouts"].append({"position_candidate": check["endpoint"],
                                                "geometric_contacts": check})
    targets = build_targets(list(candidates.values()), raw, graph)
    matrices = {tuple(row["path_edge_ids"]): row["matrix"] for row in transforms["instances"] if row["status"] == "verified_affine"}
    matrices[()] = IDENTITY
    for part in parts:
        part["geometry_candidates"] = targets["targets"][part["id"]]
        part["review_reasons"] += part["geometry_candidates"]["review_reasons"]
        for callout in callouts:
            if (len(callout["part_candidates"]) == 1 and callout["part_candidates"][0]["part_id"] == part["id"]
                    and len(callout["previews"]) == 1 and len(callout["previews"][0]["support_candidates"]) == 1):
                part["geometry_ids"].extend(callout["geometry_ids"])
        part["geometry_ids"] = sorted(set(part["geometry_ids"]))
        for group in part["geometry_candidates"]["region_groups"]:
            refs = [member["source_ref"] for member in group["members"]]
            identifier = geometry_id("HATCH_REGION", refs[0], group["members"][0]["path_edge_ids"])
            # A HATCH can contain several distinct islands. Include the native
            # boundary indices to avoid overwriting a second material region.
            identifier += ":" + fingerprint(group["members"][0]["native_boundary_indices"])[:12]
            if not any(item["id"] == identifier for item in geometry):
                straight = []
                for member in group["members"]:
                    original = raw["OBJECTS"][int(member["source_ref"]["json_pointer"].rsplit("/", 1)[1])]
                    matrix = matrices[tuple(member["path_edge_ids"])]
                    for loop in native_loops(original):
                        if loop["index"] not in member["native_boundary_indices"]:
                            continue
                        for a, b in loop["straight"]:
                            if a != b:
                                straight.append({"source_ref": member["source_ref"], "path_edge_ids": member["path_edge_ids"],
                                                 "native_boundary_index": loop["index"],
                                                 "start": transform_point(matrix, [*a, original["elevation"]]),
                                                 "end": transform_point(matrix, [*b, original["elevation"]])})
                geometry.append({"id": identifier, "kind": "HATCH_REGION", "source_refs": refs,
                                 "members": group["members"], "rings": group["rings"],
                                 "native_straight_segments": straight,
                                 "points": group["rings"][0], "physical_boundary_complete": False,
                                 "coordinate_status": "native_modelspace_or_verified_INSERT_affine"})
            part["geometry_ids"].append(identifier)
    classify_layers(raw, graph, geometry)
    contours = complete_contour_candidates(graph, geometry, parts, callouts)
    return geometry, {"native_transform_summary": transforms["summary"],
                      "native_transform_record_sha256": transforms["record_sha256"],
                      "line_occurrences": len(lines), "curve_checks": curve_summary,
                      "hatch_catalog": targets["hatch_catalog"],
                      "contacts": targets["summary"], "contours": contours, "preview_to_WCS_validated": False}
