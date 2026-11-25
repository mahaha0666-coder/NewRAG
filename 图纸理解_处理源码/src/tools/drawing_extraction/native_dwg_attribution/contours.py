"""Whole closed native contour candidates, with finite boundaries and no flooding."""

from __future__ import annotations

from collections import Counter, defaultdict
import math

from ezdxf.math import Vec2, is_point_in_polygon_2d

from research.dwg_pdf_visualization_20261031.block_contours import closed_contour
from research.dwg_pdf_visualization_20261031.geometry_relations import TOL
from research.dwg_route_validation_20261015.native_balloon_probe import finite_line_distance
from research.dwg_route_validation_20261015.parser_probe import handle_value
from src.tools.drawing_extraction.native_dwg_batch.artifacts import fingerprint

__all__ = ["complete_contour_candidates", "classify_layers"]


def classify_layers(raw: dict, graph: dict, geometry: list[dict]) -> None:
    layers = {handle_value(row.get("handle")): row.get("name", "") for row in raw["OBJECTS"]
              if row.get("object") == "LAYER"}
    nodes = {node["source_ref"]["json_pointer"]: node for node in graph["objects"]}
    edges = {edge["id"]: edge for edge in graph["reference_edges"]}

    def layer_of(ref: dict) -> str | None:
        field = nodes[ref["json_pointer"]]["field_evidence"].get("layer", {})
        return None if field.get("conflicting") else layers.get(handle_value(field.get("raw_value")))

    for row in geometry:
        if "source_ref" not in row:
            continue
        layer = layer_of(row["source_ref"])
        if layer == "0":
            for edge_id in reversed(row["path_edge_ids"]):
                inherited = layer_of(edges[edge_id]["source_ref"])
                if inherited and inherited != "0":
                    layer = inherited
                    break
        row["effective_layer"] = layer
        row["line_role"] = ("contour_solid_declared" if layer and "轮廓" in layer
                            else "centerline_declared" if layer and "中心" in layer
                            else "annotation_declared" if layer and ("标注" in layer or "尺寸" in layer)
                            else "unclassified")


def complete_contour_candidates(graph: dict, geometry: list[dict], parts: list[dict], callouts: list[dict]) -> dict:
    """All contour members of one verified INSERT must form one simple cycle.

    A point inside a closed ring proves a geometric candidate only. It cannot
    prove that the CAD block contains exactly one physical manufacturing part.
    """
    groups, skipped, closed = defaultdict(list), Counter(), []
    for row in geometry:
        if row.get("line_role") != "contour_solid_declared" or not row.get("path_edge_ids"):
            continue
        for length in range(1, len(row["path_edge_ids"]) + 1):
            groups[tuple(row["path_edge_ids"][:length])].append(row)
    nodes = []
    curves = {}
    for row in geometry:
        if row["kind"] not in {"LINE", "ARC"}:
            continue
        nodes.append({"id": row["id"], "source_ref": row["source_ref"], "native_kind": row["kind"],
                      "geometry": {"type": "line", "start": row["points"][0], "end": row["points"][-1]}
                      if row["kind"] == "LINE" else {"type": "arc"}})
        if row["kind"] == "ARC":
            curves[row["id"]] = row
    by_id = {node["id"]: node for node in nodes}
    for path, members in sorted(groups.items()):
        if len(members) > 256:
            skipped["contour_solver_member_budget_exceeded"] += 1
            continue
        try:
            proof = closed_contour([by_id[row["id"]] for row in members], curves)
        except (KeyError, ValueError) as error:
            skipped[str(error)] += 1
            continue
        closed.append({"id": "contour:" + fingerprint([path, [row["id"] for row in members]])[:24],
                       "path_edge_ids": list(path), "geometry_ids": [row["id"] for row in members], "proof": proof,
                       "physical_boundary_complete": False, "part_candidates": []})
    by_part = {part["id"]: part for part in parts}
    proposals = []
    for callout in callouts:
        if (len(callout["part_candidates"]) != 1 or len(callout["previews"]) != 1
                or len(callout["previews"][0]["support_candidates"]) != 1):
            continue
        part = by_part[callout["part_candidates"][0]["part_id"]]
        if part["geometry_ids"]:
            continue
        point = callout["previews"][0]["support_candidates"][0]["non_support_free_endpoint"]
        hits = []
        for contour in closed:
            proof = contour["proof"]
            box, ring = proof["bounds"], proof["ring"]
            if not box[0] <= point[0] <= box[2] or not box[1] <= point[1] <= box[3] or abs(point[2] - proof["plane_z"]) > TOL:
                continue
            distance = min(finite_line_distance(point, a, b)["distance_native_units"]
                           for a, b in zip(ring, ring[1:]) if math.dist(a, b) > TOL)
            if (distance <= max(2 * proof["display_error_bound"], 2 * TOL)
                    or is_point_in_polygon_2d(Vec2(point), [Vec2(p) for p in ring], abs_tol=TOL) != 1):
                continue
            hits.append(contour)
            contour["part_candidates"].append(part["id"])
        proposals.append((part, callout, hits))
    for part, callout, hits in proposals:
        part["closed_contour_candidates"] = [{"contour_id": row["id"], "geometry_ids": row["geometry_ids"]} for row in hits]
        if len(hits) == 1 and len(set(hits[0]["part_candidates"])) == 1:
            # Highlight the whole proven cycle. Its physical meaning and the
            # coordinate frame of the callout still require independent review.
            part["geometry_ids"] = sorted(set(hits[0]["geometry_ids"]))
            callout["geometry_ids"] = sorted(set(callout["geometry_ids"] + part["geometry_ids"]))
            part["review_reasons"] = [reason for reason in part["review_reasons"]
                                       if reason != "no_exact_line_or_accepted_hatch_region_contact"]
            part["review_reasons"].append("closed_CAD_contour_does_not_certify_complete_physical_part")
        elif len(hits) > 1:
            part["review_reasons"].append("multiple_closed_native_contour_candidates")
        elif hits:
            part["review_reasons"].append("same_closed_contour_used_by_multiple_BOM_items")
    return {"closed_contours": closed, "skipped": dict(skipped),
            "summary": {"native_block_paths_inspected": len(groups), "closed_native_contours": len(closed),
                        "parts_with_unique_closed_contour_candidate": sum(len(hits) == 1 and len(set(hits[0]["part_candidates"])) == 1
                                                                         for _, _, hits in proposals),
                        "physical_owners_confirmed": 0}}
