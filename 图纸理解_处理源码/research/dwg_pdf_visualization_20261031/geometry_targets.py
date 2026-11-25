"""Localize source-bound line and hatch evidence without assigning part owners.

Coincident occurrences retain every source reference. Only explicit, closed
native boundaries are used; no nearest-line snapping or connectivity flooding.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

from ezdxf.math import Vec2, bulge_to_arc, is_point_in_polygon_2d

from research.dwg_pdf_visualization_20261031.hatch_curves import flatten_cubic_spline
from research.dwg_pdf_visualization_20261031.hatch_ellipses import flatten_ellipse_edge
from research.dwg_route_validation_20261015.native_balloon_probe import finite_line_distance
from research.dwg_route_validation_20261015.native_insert_transforms import (
    build_instance_transforms, transform_point,
)
from research.dwg_route_validation_20261015.native_layer_roles import _field_evidence

__all__ = ["build_targets", "native_loops", "same_ring", "same_segment"]

TOL = 1e-7
FLATTEN_ERROR = 0.002  # Native units; only for display and guarded containment.
IDENTITY = [[1., 0., 0., 0.], [0., 1., 0., 0.],
            [0., 0., 1., 0.], [0., 0., 0., 1.]]


def _near(left: list, right: list) -> bool:
    return len(left) == len(right) and math.dist(left, right) <= TOL


def same_segment(left: dict, right: dict) -> bool:
    return (_near(left["start"], right["start"]) and _near(left["end"], right["end"])) or (
        _near(left["start"], right["end"]) and _near(left["end"], right["start"]))


def same_ring(left: list, right: list) -> bool:
    """Compare closed rings with different starts/directions, without rounding."""
    if len(left) != len(right):
        return False
    a, b = left[:-1], right[:-1]
    for slot, point in enumerate(b):
        if _near(a[0], point):
            for direction in (1, -1):
                if all(_near(p, b[(slot + direction * i) % len(b)]) for i, p in enumerate(a)):
                    return True
    return False


def _arc(center: list, radius: float, start: float, sweep: float) -> list[list[float]]:
    if (not all(math.isfinite(v) for v in [*center, radius, start, sweep])
            or radius <= 0 or not 0 < abs(sweep) <= math.tau):
        raise ValueError("invalid_native_arc")
    step = 2 * math.acos(max(-1., 1. - FLATTEN_ERROR / radius))
    if step == 0:
        raise ValueError("native_arc_exceeds_display_budget")
    count = max(2, math.ceil(abs(sweep) / step))
    if count > 20000:
        raise ValueError("native_arc_exceeds_display_budget")
    return [[center[0] + radius * math.cos(start + sweep * i / count),
             center[1] + radius * math.sin(start + sweep * i / count)] for i in range(count + 1)]


def _reject_geometry_conflicts(value: Any) -> None:
    if isinstance(value, dict):
        fields = {"extrusion", "style", "paths", "elevation", "closed", "polyline_paths", "point", "bulge",
                  "segs", "curve_type", "first_endpoint", "second_endpoint", "center", "radius",
                  "start_angle", "end_angle", "is_ccw", "degree", "is_rational", "is_periodic",
                  "knots", "control_points", "weights", "weight", "endpoint", "minor_major_ratio"}
        repeated = value.get("__native_duplicate_occurrences__", {})
        for key in fields & repeated.keys():
            if _field_evidence(value, key, "#")["conflicting"]:
                raise ValueError(f"conflicting_native_hatch_field:{key}")
        for key, child in value.items():
            if key != "__native_duplicate_occurrences__":
                _reject_geometry_conflicts(child)
    elif isinstance(value, list):
        for child in value:
            _reject_geometry_conflicts(child)


def native_loops(raw: dict, *, elliptic_edges: bool = False, canonical_closure: bool = False) -> list[dict]:
    """Read native hatch loops, retaining gaps and unsupported forms as errors.

    Native clockwise HATCH angles follow the DWG/DXF clockwise convention;
    their geometric angles are negated. Cubic spline display is bounded and
    recorded separately; sampled chords never become native straight edges.
    """
    _reject_geometry_conflicts(raw)
    if raw.get("extrusion") != [0., 0., 1.] or raw.get("style") != 0:
        raise ValueError("unsupported_hatch_plane_or_fill_style")
    result = []
    for index, boundary in enumerate(raw["paths"]):
        edges = []
        approximations = []
        if "polyline_paths" in boundary:
            if boundary.get("closed") != 1:
                raise ValueError("native_boundary_not_declared_closed")
            vertices = boundary["polyline_paths"]
            if len(vertices) < 3:
                raise ValueError("degenerate_native_boundary")
            for p, q in zip(vertices, vertices[1:] + vertices[:1], strict=True):
                start, end = p["point"], q["point"]
                bulge = p.get("bulge", 0.)
                if bulge:
                    center, a, b, radius = bulge_to_arc(Vec2(start), Vec2(end), bulge)
                    points = _arc(list(center), radius, a, (b - a) % math.tau)
                    if bulge < 0:
                        points.reverse()
                    edges.append({"points": points, "kind": "arc"})
                else:
                    edges.append({"points": [start, end], "kind": "line"})
        else:
            for ordinal, segment in enumerate(boundary["segs"]):
                if segment["curve_type"] == 1:
                    edges.append({"points": [segment["first_endpoint"], segment["second_endpoint"]],
                                  "kind": "line"})
                elif segment["curve_type"] == 2:
                    a, b = segment["start_angle"], segment["end_angle"]
                    if segment["is_ccw"] == 1:
                        sweep = (b - a) % math.tau
                    elif segment["is_ccw"] == 0:
                        a, sweep = -a, -((b - a) % math.tau)
                    else:
                        raise ValueError("unknown_native_arc_direction")
                    edges.append({"points": _arc(segment["center"], segment["radius"], a, sweep),
                                  "kind": "arc"})
                elif segment["curve_type"] == 3 and elliptic_edges:
                    points = flatten_ellipse_edge(segment, FLATTEN_ERROR)
                    edges.append({"points": points, "kind": "ellipse"})
                    approximations.append({"source_pointer_suffix": f"/paths/{index}/segs/{ordinal}",
                                           "native_kind": "elliptical_arc",
                                           "method": "bounded_second_derivative_chords",
                                           "error_bound_native_units": FLATTEN_ERROR,
                                           "point_count": len(points),
                                           "sampled_chords_are_native_line_evidence": False})
                elif segment["curve_type"] == 4:
                    points = flatten_cubic_spline(segment, FLATTEN_ERROR)
                    edges.append({"points": points, "kind": "spline"})
                    approximations.append({"source_pointer_suffix": f"/paths/{index}/segs/{ordinal}",
                                           "native_kind": "nonrational_clamped_cubic_spline",
                                           "method": "bezier_control_hull_subdivision",
                                           "error_bound_native_units": FLATTEN_ERROR,
                                           "point_count": len(points),
                                           "sampled_chords_are_native_line_evidence": False})
                else:
                    raise ValueError("unsupported_native_hatch_edge")
        points: list = []
        straight = []
        for edge in edges:
            samples = edge["points"]
            if points and not _near(points[-1], samples[0]):
                if _near(points[-1], samples[-1]):
                    samples = list(reversed(samples))
                else:
                    raise ValueError("gap_in_native_boundary")
            if edge["kind"] == "line":
                straight.append([samples[0], samples[-1]])
            points.extend(samples if not points else samples[1:])
        if len(points) < 4 or not _near(points[0], points[-1]):
            raise ValueError("open_native_boundary")
        if not all(math.isfinite(v) for point in points for v in point):
            raise ValueError("nonfinite_native_boundary")
        # Native contact was already verified above. Canonicalizing this one
        # display endpoint prevents Polygon from adding a spurious microscopic
        # closing edge. Original segments and native straight evidence stay as
        # supplied; the extra display error is explicit and propagated.
        closure = None
        if canonical_closure and points[-1] != points[0]:
            closure = {"original_display_end": list(points[-1]), "display_start": list(points[0]),
                       "displacement_native": math.dist(points[-1], points[0]),
                       "maximum_displacement_native": TOL, "native_coordinates_changed": False}
            points[-1] = list(points[0])
        area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:]))) / 2
        if area <= TOL:
            raise ValueError("zero_area_native_boundary")
        loop = {"index": index, "points": points, "straight": straight, "area": area}
        if approximations:
            loop["curve_approximations"] = approximations
        if closure:
            loop["closure_display_adjustment"] = closure
        result.append(loop)
    return result


def _contains(point: list, loops: list[dict]) -> tuple[str, int | None]:
    states = [is_point_in_polygon_2d(Vec2(point), list(map(Vec2, loop["points"])), abs_tol=TOL)
              for loop in loops]
    exact = [i for i, loop in enumerate(loops) if any(
        finite_line_distance([*point, 0.], [*a, 0.], [*b, 0.])["numeric_contact"]
        for a, b in loop["straight"] if not _near(a, b))]
    if exact:
        regions = []
        for index in exact:
            ancestors = [i for i, loop in enumerate(loops) if i != index and
                         is_point_in_polygon_2d(Vec2(loops[index]["points"][0]),
                                                list(map(Vec2, loop["points"])), abs_tol=TOL) == 1]
            # A hole edge belongs to its surrounding material region. Selecting
            # the hole ring as a filled face would invent material in the hole.
            regions.append(min(ancestors, key=lambda i: loops[i]["area"])
                           if len(ancestors) % 2 else index)
        return "on_native_straight_boundary", min(regions, key=lambda i: loops[i]["area"])
    # Tessellated arcs are never promoted to exact native incidence. The guard
    # also rejects uncertain points at hole boundaries or near curve chords.
    if any(finite_line_distance([*point, 0.], [*a, 0.], [*b, 0.])["distance_native_units"]
           <= 2 * FLATTEN_ERROR for loop in loops for a, b in zip(loop["points"], loop["points"][1:])
           if not _near(a, b)):
        return "near_boundary_unresolved", None
    inside = [i for i, state in enumerate(states) if state == 1]
    if len(inside) % 2:
        return "inside_native_hatch_candidate", min(inside, key=lambda i: loops[i]["area"])
    return "outside", None


def _regions(raw: dict, graph: dict) -> tuple[list[dict], dict]:
    transforms = build_instance_transforms(graph)
    by_pointer = {node["source_ref"]["json_pointer"]: node for node in graph["objects"]}
    transform_by_path = {tuple(row["path_edge_ids"]): row for row in transforms["instances"]}
    occurrences = []
    for node in graph["objects"]:
        if node["native_kind"] == "HATCH" and node["cad_context"]["cad_space"] == "modelspace":
            occurrences.append((node, [], {"matrix": IDENTITY, "inverse": IDENTITY}))
    for instance in graph["modelspace_instances"]:
        transform = transform_by_path[tuple(instance["path_edge_ids"])]
        for ref in instance["member_source_refs"]:
            node = by_pointer[ref["json_pointer"]]
            if node["native_kind"] == "HATCH":
                occurrences.append((node, instance["path_edge_ids"], transform))
    regions, skipped, unresolved = [], Counter(), []
    for node, path, transform in occurrences:
        reason = None
        if (graph["truncation"]["truncated"] or node["cad_context"]["reasons"]
                or path and transform.get("status") != "verified_affine"):
            reason = "unverified_native_context_or_transform"
        original = raw["OBJECTS"][int(node["source_ref"]["json_pointer"].rsplit("/", 1)[1])]
        if reason is None:
            try:
                loops = native_loops(original)
            except (KeyError, ValueError) as error:
                reason = str(error)
        if reason is not None:
            skipped[reason] += 1
            unresolved.append({"source_ref": node["source_ref"], "path_edge_ids": path, "reason": reason})
            continue
        elevation = original["elevation"]
        regions.append({"source_ref": node["source_ref"], "path_edge_ids": path,
                        "matrix": transform["matrix"], "inverse": transform["inverse"],
                        "elevation": elevation, "loops": loops})
    return regions, {"native_hatch_occurrences": len(occurrences), "accepted": len(regions),
                     "skipped": dict(skipped), "unresolved": unresolved,
                     "flatten_error_native_units": FLATTEN_ERROR,
                     "coincident_geometry_tolerance_native_units": TOL}


def build_targets(parts: list[dict], raw: dict, graph: dict) -> dict[str, Any]:
    """Build local evidence; a unique line or face never certifies ownership."""
    regions, catalog_summary = _regions(raw, graph)
    nodes = {node["source_ref"]["native_handle"]: node for node in graph["objects"]}
    edges = {edge["id"]: edge for edge in graph["reference_edges"]}
    layers = {f"{obj['handle'][-1]:X}": obj["name"] for obj in raw["OBJECTS"] if obj.get("object") == "LAYER"}
    targets, contacts, collapsed, region_rows = {}, 0, 0, 0
    def layer_of(ref: dict) -> str:
        field = nodes[ref["native_handle"]]["field_evidence"].get("layer", {})
        value = field.get("raw_value")
        return layers.get(f"{value[-1]:X}", "") if value and not field.get("conflicting") else ""

    for part in parts:
        line_groups, face_groups = [], []
        for callout in part["callouts"]:
            for contact in callout["geometric_contacts"]["baseline_line_contacts"]:
                line = contact["line"]
                layer = layer_of(line["source_ref"])
                if layer == "0":
                    for edge_id in reversed(line["path_edge_ids"]):
                        candidate = layer_of(edges[edge_id]["source_ref"])
                        if candidate and candidate != "0":
                            layer = candidate
                            break
                role = ("contour" if "轮廓" in layer else "center" if "中心" in layer
                        else "annotation" if "标注" in layer else "unclassified")
                member = {"source_ref": line["source_ref"], "path_edge_ids": line["path_edge_ids"],
                          "effective_layer": layer, "role": role}
                match = next((group for group in line_groups if same_segment(group, line)), None)
                if match is None:
                    match = {"start": line["start"], "end": line["end"], "members": [], "shared_with": []}
                    line_groups.append(match)
                match["members"].append(member)
                contacts += 1
            for region in regions:
                local = transform_point(region["inverse"], callout["position_candidate"])
                if abs(local[2] - region["elevation"]) > TOL:
                    continue
                status, index = _contains(local[:2], region["loops"])
                if index is None:
                    continue
                selected = region["loops"][index]
                contained = [loop for loop in region["loops"] if loop is selected or
                             is_point_in_polygon_2d(Vec2(loop["points"][0]),
                                                    list(map(Vec2, selected["points"])), abs_tol=TOL) == 1]
                rings = [[transform_point(region["matrix"], [*p, region["elevation"]])
                          for p in loop["points"]] for loop in contained]
                member = {"source_ref": region["source_ref"], "path_edge_ids": region["path_edge_ids"],
                          "native_boundary_indices": [loop["index"] for loop in contained], "contact": status}
                match = next((group for group in face_groups if len(group["rings"]) == len(rings)
                              and all(same_ring(a, b) for a, b in zip(group["rings"], rings, strict=True))), None)
                if match is None:
                    match = {"rings": rings, "members": [], "shared_with": []}
                    face_groups.append(match)
                match["members"].append(member)
        collapsed += sum(len(group["members"]) - 1 for group in line_groups)
        region_rows += bool(face_groups)
        for group in line_groups:
            roles = {member["role"] for member in group["members"]}
            group["role"] = "contour" if roles == {"contour"} else "unclassified"
            if roles & {"center", "annotation"}:
                group["role"] = "excluded_annotation"
        targets[part["id"]] = {"line_groups": line_groups, "region_groups": face_groups,
                               "physical_owner_confirmed": False,
                               "coordinate_status": "native_preview_identity_assumption_not_WCS_certified"}
    for part in parts:
        for other in parts:
            if part["id"] == other["id"]:
                continue
            for group in targets[part["id"]]["line_groups"]:
                if any(same_segment(group, candidate) for candidate in targets[other["id"]]["line_groups"]):
                    group["shared_with"].append(other["id"])
    for target in targets.values():
        reasons = ["physical_part_boundary_not_verified", "preview_to_CAD_not_independently_WCS_certified"]
        if not target["line_groups"] and not target["region_groups"]:
            reasons.append("no_exact_line_or_accepted_hatch_region_contact")
        if any(group["shared_with"] for group in target["line_groups"]):
            reasons.append("same_contact_segment_used_by_multiple_BOM_items")
        if any(group["role"] != "contour" for group in target["line_groups"]):
            reasons.append("unclassified_or_annotation_line_requires_review")
        target["review_reasons"] = reasons
    return {"targets": targets, "hatch_catalog": catalog_summary,
            "summary": {"original_line_contact_occurrences": contacts,
                        "coincident_line_occurrences_collapsed": collapsed,
                        "unique_contact_segments_per_callout": contacts - collapsed,
                        "parts_with_hatch_region_candidates": region_rows,
                        "parts_with_shared_contact_segments": sum(any(g["shared_with"] for g in t["line_groups"])
                                                                  for t in targets.values()),
                        "physical_owners_confirmed": 0},
            "policy": {"nearest_line_snapping": False, "connectivity_flooding": False,
                       "regions_are_complete_part_boundaries": False, "confidence_generated": False}}
