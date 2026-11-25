"""Conservative closed-contour candidates inside verified native CAD blocks.

Only original LINE/ARC endpoints establish closure. Display chords are used
for guarded containment, never for connectivity, contact or physical ownership.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import combinations
import math

from ezdxf.math import ConstructionCircle, Vec2, is_point_in_polygon_2d

from research.dwg_pdf_visualization_20261031.geometry_relations import TOL, bounds, line_relation
from research.dwg_pdf_visualization_20261031.geometry_targets import FLATTEN_ERROR, _arc
from research.dwg_pdf_visualization_20261031.line_faces import finite_faces
from research.dwg_route_validation_20261015.native_balloon_probe import finite_line_distance
from research.dwg_route_validation_20261015.native_curve_contacts import finite_curve_contact
from research.dwg_route_validation_20261015.native_insert_transforms import transform_point


def _primitive(node: dict, curves: dict[str, dict]) -> dict:
    geometry = node.get("geometry") or {}
    if node["native_kind"] == "LINE" and geometry.get("type") == "line":
        points = [geometry["start"], geometry["end"]]
        if math.dist(*points) <= TOL:
            raise ValueError("degenerate_contour_line")
        return {"id": node["id"], "source_ref": node["source_ref"], "kind": "LINE",
                "endpoints": points, "points": points, "error_bound": 0.}
    curve = curves.get(node["id"])
    if node["native_kind"] != "ARC" or not curve or curve["status"] != "accepted":
        raise ValueError("unsupported_contour_primitive")
    matrix, center = curve["matrix"], curve["native_center"]
    if any(abs(matrix[2][axis]) > TOL for axis in (0, 1)):
        raise ValueError("contour_plane_not_parallel_to_display")
    start, sweep = curve["start_angle"], (curve["end_angle"] - curve["start_angle"]) % math.tau
    points = [transform_point(matrix, [*point, center[2]])
              for point in _arc(center[:2], curve["native_radius"], start, sweep)]
    return {"id": node["id"], "source_ref": node["source_ref"], "kind": "ARC", "curve": curve,
            "endpoints": [points[0], points[-1]], "points": points,
            "error_bound": FLATTEN_ERROR * curve["uniform_scale"]}


def _cycle(primitives: list[dict]) -> list[tuple[int, int]]:
    """Require one unique endpoint partner, no snapping or transitive clustering."""
    if len(primitives) < 3:
        raise ValueError("fewer_than_three_contour_primitives")
    endpoints = [(i, slot, point) for i, row in enumerate(primitives) for slot, point in enumerate(row["endpoints"])]
    partners = {}
    for i, slot, point in endpoints:
        matches = [(j, other) for j, other, candidate in endpoints if j != i and math.dist(point, candidate) <= TOL]
        if len(matches) != 1:
            raise ValueError("open_or_branching_native_endpoints")
        partners[i, slot] = matches[0]
    ordered, visited, current = [], set(), (0, 0)
    while current[0] not in visited:
        i, entry = current
        visited.add(i)
        ordered.append(current)
        current = partners[i, 1 - entry]
    if current != (0, 0) or len(visited) != len(primitives):
        raise ValueError("multiple_cycles_or_incomplete_contour")
    return ordered


def _circle(primitive: dict) -> ConstructionCircle:
    curve = primitive["curve"]
    center = transform_point(curve["matrix"], curve["native_center"])
    return ConstructionCircle(Vec2(center), curve["native_radius"] * curve["uniform_scale"])


def _finite_circle_line(circle: ConstructionCircle, endpoints: list[list[float]]) -> list[Vec2]:
    """Intersect the actual segment; exact discriminant, high precision roots.

    The generic infinite-line solver can return points off the source line near
    shallow native arcs. A positive exact discriminant is never rounded into a
    tangent. Roots outside [0, 1] cannot become a contour self-intersection.
    """
    start, end = [[Fraction(value) for value in point[:2]] for point in endpoints]
    center = [Fraction(value) for value in circle.center]
    radius = Fraction(circle.radius)
    delta = [b - a for a, b in zip(start, end, strict=True)]
    offset = [a - b for a, b in zip(start, center, strict=True)]
    a = sum(value * value for value in delta)
    b = 2 * sum(x * y for x, y in zip(offset, delta, strict=True))
    c = sum(value * value for value in offset) - radius * radius
    if a <= 0:
        raise ValueError("degenerate_contour_line")
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return []
    with localcontext() as context:
        context.prec = 80

        def decimal(value: Fraction) -> Decimal:
            return Decimal(value.numerator) / Decimal(value.denominator)

        root = decimal(discriminant).sqrt()
        parameters = [(-decimal(b) - root) / (2 * decimal(a))]
        if discriminant:
            parameters.append((-decimal(b) + root) / (2 * decimal(a)))
        points = []
        for parameter in parameters:
            if not 0 <= parameter <= 1:
                continue
            point = Vec2(*(float(decimal(x) + parameter * decimal(d))
                           for x, d in zip(start, delta, strict=True)))
            check = finite_line_distance([point.x, point.y, endpoints[0][2]], *endpoints)
            if check["distance_native_units"] > TOL:
                raise ValueError("finite_circle_line_numeric_residual_unresolved")
            points.append(point)
        return points


def _simple(primitives: list[dict]) -> None:
    """Reject crossings, overlapping lines and unsupported co-circular arcs."""
    for left, right in combinations(primitives, 2):
        joins = [a for a in left["endpoints"] for b in right["endpoints"] if math.dist(a, b) <= TOL]
        if left["kind"] == right["kind"] == "LINE":
            relation = line_relation(dict(zip(("start", "end"), left["endpoints"])),
                                     dict(zip(("start", "end"), right["endpoints"])))
            if relation and (relation[0] != "finite_line_contact" or not joins):
                raise ValueError("self_intersection_or_overlapping_contour")
            continue
        if left["kind"] == right["kind"] == "ARC":
            a, b = _circle(left), _circle(right)
            if a.center.distance(b.center) <= TOL and abs(a.radius - b.radius) <= TOL:
                raise ValueError("co_circular_contour_arcs_unresolved")
            points = a.intersect_circle(b, abs_tol=TOL)
        else:
            arc, line = (left, right) if left["kind"] == "ARC" else (right, left)
            points = _finite_circle_line(_circle(arc), line["endpoints"])
        for xy in points:
            point = [xy.x, xy.y, left["endpoints"][0][2]]
            contacts = [finite_curve_contact(point, row["curve"])["numeric_contact"]
                        for row in (left, right) if row["kind"] != "LINE"]
            if any(contact is None for contact in contacts):
                raise ValueError("native_curve_self_intersection_unresolved")
            if not all(contacts):
                continue
            if not any(math.dist(point, joint) <= TOL for joint in joins):
                raise ValueError("self_intersection_or_overlapping_contour")


def closed_contour(nodes: list[dict], curves: dict[str, dict]) -> dict:
    """Prove a single simple cycle using every declared contour member."""
    primitives = [_primitive(node, curves) for node in sorted(nodes, key=lambda row: row["id"])]
    endpoints = [point for row in primitives for point in row["endpoints"]]
    if endpoints and max(point[2] for point in endpoints) - min(point[2] for point in endpoints) > TOL:
        raise ValueError("nonplanar_contour")
    order = _cycle(primitives)
    _simple(primitives)
    ring, ordered, joins = [], [], []
    for ordinal, (i, entry) in enumerate(order):
        row = primitives[i]
        points = row["points"] if entry == 0 else list(reversed(row["points"]))
        ring.extend(points)
        ordered.append({key: value for key, value in row.items() if key != "points"} | {"entry_endpoint": entry})
        j, other_entry = order[(ordinal + 1) % len(order)]
        left, right = row["endpoints"][1 - entry], primitives[j]["endpoints"][other_entry]
        joins.append({"source": row["id"], "target": primitives[j]["id"],
                      "source_endpoint": left, "target_endpoint": right, "distance": math.dist(left, right)})
    ring.append(ring[0])
    return {"status": "single_simple_native_contour", "primitives": ordered, "endpoint_joins": joins,
            "ring": ring, "bounds": bounds(ring), "plane_z": endpoints[0][2],
            "display_error_bound": max(row["error_bound"] for row in primitives),
            "join_tolerance_native_units": TOL, "all_contour_members_used": True,
            "sampled_chords_used_for_connectivity": False, "physical_owner_implied": False}


def _inside_box(point: list, box: list) -> bool:
    return box[0] - TOL <= point[0] <= box[2] + TOL and box[1] - TOL <= point[1] <= box[3] + TOL


def _finite_face_candidates(blocks: list[dict], nodes: dict[str, dict], callouts: list[dict], linked: set[str]) -> None:
    """Supplement open/T-junction contours without changing the strict proof."""
    unavailable = linked | {row["query"]["callout"] for row in blocks if row["status"] == "eligible_callout_candidate"}
    candidates = defaultdict(list)
    for block in blocks:
        if block.get("reason") != "open_or_branching_native_endpoints":
            continue
        contour = [nodes[key] for key in block["contour_ids"]]
        if not all(node["native_kind"] == "LINE" and (node.get("geometry") or {}).get("type") == "line" for node in contour):
            continue
        lines = [{"id": node["id"], "source_ref": node["source_ref"], **node["geometry"]} for node in contour]
        box = bounds([point for line in lines for point in (line["start"], line["end"])])
        enclosed = [callout for callout in callouts if _inside_box(callout["position"], box)]
        decision = {"status": "unresolved", "callouts_in_source_bounds": [node["id"] for node in enclosed]}
        block["line_face_candidate"] = decision
        if len(enclosed) != 1:
            decision["reason"] = "multiple_or_no_callouts_in_source_bounds"
            continue
        callout = enclosed[0]
        if callout["id"] in unavailable:
            decision["reason"] = "existing_stronger_graphic_evidence_preserved"
            continue
        try:
            result = finite_faces(lines)
        except ValueError as error:
            decision["reason"] = str(error)
            continue
        block["line_faces"] = result
        point = callout["position"]
        matches = [face for face in result["faces"] if face["status"] == "bounded_finite_line_face" and
                   abs(point[2] - face["plane_z"]) <= TOL and
                   is_point_in_polygon_2d(Vec2(point), [Vec2(p) for p in face["ring"]], abs_tol=TOL) == 1]
        if len(matches) != 1:
            decision["reason"] = "no_unique_bounded_finite_face"
            continue
        face = matches[0]
        distance = min(finite_line_distance(point, row["start"], row["end"])["distance_native_units"] for row in face["intervals"])
        query = {"callout": callout["id"], "point": point, "boundary_distance": distance, "guard": 2 * TOL,
                 "preview_WCS_certified": False, "endpoint_frame_ref": callout["endpoint_frame_ref"]}
        decision.update(face_id=face["id"], query=query)
        if distance <= query["guard"]:
            decision["reason"] = "near_boundary_unresolved"
            continue
        decision["status"] = "eligible_callout_candidate"
        candidates[callout["id"]].append(decision)
    for rows in candidates.values():
        if len(rows) > 1:
            for row in rows:
                row.update(status="unresolved", reason="multiple_native_block_candidates")


def collect(nodes: list[dict], edges: list[dict], curves: list[dict]) -> dict:
    """Inspect native memberships and propose only unique unresolved callouts."""
    by_id = {node["id"]: node for node in nodes}
    children = defaultdict(list)
    for edge in edges:
        if (edge["kind"] == "placed_block_member" and edge["status"] == "native_reference"
                and edge["evidence"].get("coordinate_status") == "verified_affine"):
            children[edge["source"]].append(edge["target"])
    callouts = [node for node in nodes if node["kind"] == "callout"]
    already_linked = {edge["source"] for edge in edges if edge["kind"] in
                      {"pointer_on_finite_line", "pointer_on_finite_curve", "callout_hatch_scope"}}
    curves_by_id = {row["id"]: row for row in curves}
    blocks = []
    for root in sorted(children):
        members, frontier = set(), [root]
        while frontier:
            for identifier in children.get(frontier.pop(), []):
                if identifier not in members:
                    members.add(identifier)
                    frontier.append(identifier)
        contour = [by_id[key] for key in sorted(members) if by_id[key].get("line_role") == "contour_solid_declared"]
        if not contour:
            continue
        row = {"root": root, "members": sorted(members), "contour_ids": [node["id"] for node in contour],
               "status": "unresolved", "physical_owner_implied": False}
        try:
            proof = closed_contour(contour, curves_by_id)
        except ValueError as error:
            row["reason"] = str(error)
            blocks.append(row)
            continue
        row.update(status="closed_contour_candidate", proof=proof)
        enclosed = [node["id"] for node in callouts if _inside_box(node["position"], proof["bounds"])]
        row["callouts_in_contour_bounds"] = enclosed
        if len(enclosed) != 1:
            row["reason"] = "multiple_or_no_callouts_in_contour_bounds"
        else:
            callout = by_id[enclosed[0]]
            point, ring = callout["position"], proof["ring"]
            distance = min(finite_line_distance(point, a, b)["distance_native_units"] for a, b in zip(ring, ring[1:])
                           if math.dist(a, b) > TOL)
            row["query"] = {"callout": callout["id"], "point": point, "boundary_distance": distance,
                            "guard": max(2 * proof["display_error_bound"], 2 * TOL),
                            "preview_WCS_certified": False, "endpoint_frame_ref": callout["endpoint_frame_ref"]}
            if callout["id"] in already_linked:
                row["reason"] = "existing_native_contact_or_hatch_scope_preserved"
            elif abs(point[2] - proof["plane_z"]) > TOL:
                row["reason"] = "callout_not_in_contour_plane"
            elif distance <= row["query"]["guard"]:
                row["reason"] = "near_boundary_unresolved"
            elif is_point_in_polygon_2d(Vec2(point), [Vec2(p) for p in ring], abs_tol=TOL) != 1:
                row["reason"] = "outside_native_contour"
            else:
                row["status"] = "eligible_callout_candidate"
        blocks.append(row)
    candidates = defaultdict(list)
    for row in blocks:
        if row["status"] == "eligible_callout_candidate":
            candidates[row["query"]["callout"]].append(row)
    for rows in candidates.values():
        if len(rows) > 1:
            for row in rows:
                row.update(status="unresolved", reason="multiple_native_block_candidates")
    _finite_face_candidates(blocks, by_id, callouts, already_linked)
    return {"blocks": blocks, "summary": {"inspected_blocks": len(blocks),
            "single_closed_contours": sum("proof" in row for row in blocks),
            "callout_candidates": sum(row["status"] == "eligible_callout_candidate" for row in blocks),
            "finite_line_face_candidates": sum(row.get("line_face_candidate", {}).get("status") == "eligible_callout_candidate" for row in blocks),
            "reasons": dict(Counter(row["reason"] for row in blocks if "reason" in row)),
            "physical_part_owners_confirmed": 0}}
