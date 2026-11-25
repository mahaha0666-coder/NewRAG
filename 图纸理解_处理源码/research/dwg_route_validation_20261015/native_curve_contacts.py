"""Finite native ARC/CIRCLE incidence; storage evidence never proves a part.

Only explicit +Z native OCS circles and verified similarity INSERT paths are
admitted. Native angles are radians, increasing counterclockwise before INSERT.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from typing import Any

from research.dwg_route_validation_20261015.native_insert_transforms import (
    SCHEMA as TRANSFORM_SCHEMA,
    _digest, _field, _inverse, _matrix, _number, _point, _validate_graph,
    transform_point,
)
from research.dwg_route_validation_20261015.native_relation_features import _guards

SCHEMA = "native_curve_contacts.v1"
ABS_TOLERANCE = 1e-8
REL_TOLERANCE = 1e-11
SIMILARITY_TOLERANCE = 1e-10
ANGLE_AMBIGUITY_TOLERANCE = 1e-12
IDENTITY = [[1., 0., 0., 0.], [0., 1., 0., 0.],
            [0., 0., 1., 0.], [0., 0., 0., 1.]]
PARAMETERS = {
    "absolute_tolerance_native_units": ABS_TOLERANCE,
    "relative_tolerance": REL_TOLERANCE,
    "scale_definition": "max(abs(query XYZ), abs(modelspace full-circle AABB extrema XYZ))",
    "aabb_definition": "world center +/- radius * hypot(matrix[row][0], matrix[row][1])",
    "similarity_relative_tolerance": SIMILARITY_TOLERANCE,
    "zero_full_turn_angle_ambiguity_radians": ANGLE_AMBIGUITY_TOLERANCE,
    "native_angle_unit": "radian",
    "arc_direction": "CCW in native +Z OCS before INSERT; never sort endpoints",
    "arc_aabb_policy": "full parent circle only for numerical scale, not finite arc extent",
}


def _arc_domain(start: Any, end: Any) -> tuple[float, float, float]:
    start, end = _number(start), _number(end)
    difference = _number(end - start)
    if (abs(difference) >= math.tau or abs(difference) <= ANGLE_AMBIGUITY_TOLERANCE
            or math.tau - abs(difference) <= ANGLE_AMBIGUITY_TOLERANCE):
        raise ValueError("zero_full_turn_or_multiturn_arc_ambiguous")
    return start, end, difference % math.tau


def _similarity(matrix: Any, inverse: Any) -> tuple[list, list, float]:
    matrix, inverse = _matrix(matrix), _matrix(inverse)
    expected_inverse, _ = _inverse(matrix)
    if any(not math.isclose(inverse[row][col], expected_inverse[row][col],
                            rel_tol=SIMILARITY_TOLERANCE, abs_tol=1e-12)
           for row in range(4) for col in range(4)):
        raise ValueError("inverse_does_not_match_matrix")
    columns = [[matrix[row][col] for row in range(3)] for col in range(3)]
    lengths = [math.hypot(*column) for column in columns]
    if any(not math.isfinite(length) or length <= 0 for length in lengths):
        raise ValueError("singular_or_nonfinite_similarity")
    if any(not math.isclose(length, lengths[0], rel_tol=SIMILARITY_TOLERANCE, abs_tol=0.)
           for length in lengths[1:]):
        raise ValueError("nonuniform_scale_not_supported")
    normalized = [[value / length for value in column] for column, length in zip(columns, lengths)]
    if any(abs(sum(left * right for left, right in zip(normalized[a], normalized[b])))
           > SIMILARITY_TOLERANCE for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("shear_not_supported")
    return matrix, inverse, lengths[0]


def _circle_aabb(center: list, radius: float, matrix: list) -> dict:
    world_center = transform_point(matrix, center)
    extents = [_number(radius * math.hypot(matrix[row][0], matrix[row][1])) for row in range(3)]
    return {"min": [_number(value - extent) for value, extent in zip(world_center, extents)],
            "max": [_number(value + extent) for value, extent in zip(world_center, extents)]}


def _native_curve(node: dict, path: list, transform: dict | None, reasons: list) -> dict:
    fields = node["field_evidence"]
    result = {"source_ref": deepcopy(node["source_ref"]), "path_edge_ids": list(path),
              "native_kind": node["native_kind"], "status": "unknown", "reasons": list(reasons),
              "native_field_evidence": {key: deepcopy(fields[key]) for key in
                                        ("center", "radius", "extrusion", "start_angle", "end_angle")
                                        if key in fields},
              "native_center": None, "native_radius": None, "start_angle": None, "end_angle": None,
              "matrix": None, "inverse": None, "uniform_scale": None, "world_circle_aabb": None,
              "physical_owner": "unknown", "eligible_for_route_training": False}
    if node.get("opaque_payload") is not False:
        result["reasons"].append("opaque_native_curve_payload")
    try:
        center, radius = _point(_field(fields, "center")), _number(_field(fields, "radius"))
        normal = _point(_field(fields, "extrusion"))
        result.update(native_center=center, native_radius=radius)
        if normal != [0., 0., 1.]:
            raise ValueError("explicit_positive_Z_normal_required")
        if radius <= 0:
            raise ValueError("positive_radius_required")
        if node["native_kind"] == "ARC":
            start, end, _ = _arc_domain(_field(fields, "start_angle"), _field(fields, "end_angle"))
            result.update(start_angle=start, end_angle=end)
        if transform is None:
            raise ValueError("modelspace_transform_unverified")
        matrix, inverse, scale = _similarity(transform["matrix"], transform["inverse"])
        aabb = _circle_aabb(center, radius, matrix)
        result.update(matrix=matrix, inverse=inverse, uniform_scale=scale, world_circle_aabb=aabb)
        if not result["reasons"]:
            result["status"] = "accepted"
    except (ValueError, OverflowError, ZeroDivisionError, KeyError) as error:
        result["reasons"].append(f"curve_unresolved:{error}")
    return result


def curve_catalog(graph: dict, transforms: dict) -> dict:
    """Project source-qualified curve occurrences without reading any files."""
    _validate_graph(graph)
    _guards(transforms)
    if (transforms.get("schema_version") != TRANSFORM_SCHEMA
            or transforms.get("source_kind") != "native_dwg"
            or transforms.get("record_sha256") != _digest({key: value for key, value in transforms.items()
                                                           if key != "record_sha256"})
            or transforms.get("ownership_record_sha256") != graph["record_sha256"]
            or any(transforms.get(key) != graph[key] for key in ("source_sha256", "response_sha256"))):
        raise ValueError("Source/ownership/transform record binding mismatch")
    nodes = {node["source_ref"]["json_pointer"]: node for node in graph["objects"]}
    by_path = {}
    for transform in transforms["instances"]:
        path = tuple(transform["path_edge_ids"])
        if path in by_path:
            raise ValueError("Ambiguous transform instance path")
        by_path[path] = transform
    objects, represented = [], set()
    catalog_reasons = ["modelspace_reachability_truncated"] if any(
        value.get("truncation", {}).get("truncated") for value in (graph, transforms)) else []
    for occurrence in graph["modelspace_instances"]:
        path = occurrence["path_edge_ids"]
        transform = by_path.get(tuple(path))
        reasons = list(catalog_reasons)
        if (transform is None or transform.get("status") != "verified_affine"
                or transform.get("reasons") or transform.get("source_ref") != occurrence["source_ref"]
                or transform.get("definition_source_ref") != occurrence["definition_source_ref"]):
            reasons.append("instance_path_transform_unknown_or_mismatched")
            transform = None
        for ref in occurrence["member_source_refs"]:
            node = nodes.get(ref.get("json_pointer"))
            if node is None or node["source_ref"] != ref:
                raise ValueError("Instance member reference does not match source graph")
            if node["native_kind"] in {"ARC", "CIRCLE"}:
                represented.add(ref["json_pointer"])
                objects.append(_native_curve(node, path, transform, reasons))
    for node in graph["objects"]:
        if node["native_kind"] not in {"ARC", "CIRCLE"}:
            continue
        context = node["cad_context"]
        if context["cad_space"] == "modelspace":
            reasons = list(catalog_reasons)
            if context.get("reasons") or context["status"] == "unknown":
                reasons.append("direct_modelspace_context_unverified")
            objects.append(_native_curve(node, [], {"matrix": IDENTITY, "inverse": IDENTITY}, reasons))
        elif node["source_ref"]["json_pointer"] not in represented:
            objects.append(_native_curve(node, [], None, ["no_verified_modelspace_occurrence"]))
    counts = Counter(row["status"] for row in objects)
    output = {"schema_version": SCHEMA, "source_kind": "native_dwg",
              "source_sha256": graph["source_sha256"], "response_sha256": graph["response_sha256"],
              "ownership_record_sha256": graph["record_sha256"],
              "transform_record_sha256": transforms["record_sha256"],
              "parameters": deepcopy(PARAMETERS), "objects": objects,
              "summary": {"curve_occurrences": len(objects), "accepted": counts["accepted"],
                          "unknown": counts["unknown"], "accepted_by_kind": dict(Counter(
                              row["native_kind"] for row in objects if row["status"] == "accepted"))},
              "eligible_for_route_training": False}
    output["record_sha256"] = _digest(output)
    return output


def finite_curve_contact(point: list, curve: dict) -> dict:
    """Check finite curve incidence in native modelspace units, with no fallback."""
    result = {"status": "unknown", "numeric_contact": None, "parameter": None,
              "parameter_kind": None, "evaluated_modelspace_point": None,
              "residual_native_units": None, "tolerance_native_units": None,
              "parameters": deepcopy(PARAMETERS), "reasons": []}
    if curve.get("status") != "accepted" or curve.get("reasons"):
        result["reasons"].append("curve_not_accepted")
        return result
    try:
        query, center = _point(point), _point(curve["native_center"])
        radius = _number(curve["native_radius"])
        if radius <= 0:
            raise ValueError("positive_radius_required")
        matrix, inverse, scale = _similarity(curve["matrix"], curve["inverse"])
        aabb = _circle_aabb(center, radius, matrix)
        magnitude = max(abs(value) for value in query + aabb["min"] + aabb["max"])
        tolerance = _number(ABS_TOLERANCE + REL_TOLERANCE * magnitude)
        result["tolerance_native_units"] = tolerance
        local = transform_point(inverse, query)
        delta = [_number(local[axis] - center[axis]) for axis in range(3)]
        rho = _number(math.hypot(delta[0], delta[1]))
        if rho == 0:
            result.update(status="no_contact", numeric_contact=False)
            result["reasons"].append("circle_center_has_no_boundary_direction")
            return result
        angle = math.atan2(delta[1], delta[0]) % math.tau
        kind = curve["native_kind"]
        if kind == "ARC":
            start, end, sweep = _arc_domain(curve["start_angle"], curve["end_angle"])
            offset = (angle - start % math.tau) % math.tau
            if offset > sweep:
                # Only actual endpoint incidence closes floating-point wraparound.
                endpoints = [transform_point(matrix, [
                    _number(center[0] + radius * math.cos(endpoint)),
                    _number(center[1] + radius * math.sin(endpoint)), center[2]]) for endpoint in (start, end)]
                residuals = [_number(math.dist(query, endpoint)) for endpoint in endpoints]
                if min(residuals) > tolerance:
                    result.update(status="no_contact", numeric_contact=False)
                    result["reasons"].append("outside_finite_counterclockwise_arc")
                    return result
                endpoint_index = min(range(2), key=residuals.__getitem__)
                angle, offset = (start, 0.) if endpoint_index == 0 else (end, sweep)
            result.update(parameter=offset / sweep, parameter_kind="arc_CCW_fraction")
        elif kind == "CIRCLE":
            result.update(parameter=angle, parameter_kind="circle_native_radian")
        else:
            raise ValueError("unsupported_native_curve_kind")
        evaluated = transform_point(matrix, [_number(center[0] + radius * math.cos(angle)),
                                              _number(center[1] + radius * math.sin(angle)), center[2]])
        residual = _number(math.dist(query, evaluated))
        plane_residual = _number(abs(delta[2]) * scale)
        contact = residual <= tolerance and plane_residual <= tolerance
        result.update(status="contact" if contact else "no_contact", numeric_contact=contact,
                      evaluated_modelspace_point=evaluated, residual_native_units=residual)
        if not contact:
            result["reasons"].append("different_curve_plane" if plane_residual > tolerance else "outside_curve_boundary")
    except (ValueError, OverflowError, ZeroDivisionError, KeyError, TypeError) as error:
        result.update(status="unknown", numeric_contact=None)
        result["reasons"].append(f"contact_unresolved:{error}")
    return result
