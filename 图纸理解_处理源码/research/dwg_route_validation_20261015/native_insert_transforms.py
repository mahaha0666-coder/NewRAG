"""Source-bound INSERT affine transforms; no rendering or physical-part claims.

Native JSON INSERT.rotation is radians: the pinned LibreDWG out_dxf.c converts
group 50 with rad2deg, whereas out_json.c emits the raw value. The formula follows
ezdxf Insert.matrix44/OCS, expressed here with column vectors instead of rows.
Dimension display blocks deliberately have no transform implementation here.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from research.dwg_route_validation_20261015.native_cad_ownership import (
    SCHEMA as OWNERSHIP_SCHEMA,
)
from research.dwg_route_validation_20261015.native_relation_features import _guards

SCHEMA = "native_insert_transforms.v1"
CONVENTION = "column vectors; native block XYZ -> modelspace native WCS; root_matrix @ child_matrix"
_REQUIRED = ("ins_pt", "scale", "rotation", "extrusion")
_ARRAY = ("num_rows", "num_cols", "row_spacing", "col_spacing")
Matrix = list[list[float]]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _number(value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("non_numeric_value")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("unrepresentable_number") from error
    if not math.isfinite(result):
        raise ValueError("nonfinite_number")
    return result


def _point(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("explicit_three_component_point_required")
    return [_number(item) for item in value]


def _matrix(value: Any) -> Matrix:
    if not isinstance(value, list) or len(value) != 4 or any(
            not isinstance(row, list) or len(row) != 4 for row in value):
        raise ValueError("explicit_affine_4x4_required")
    matrix = [[_number(item) for item in row] for row in value]
    if matrix[3] != [0., 0., 0., 1.]:
        raise ValueError("non_affine_last_row")
    return matrix


def transform_point(matrix: Matrix, point: list[float]) -> list[float]:
    """Apply a finite affine 4x4 to an explicit native XYZ point, without units."""
    matrix, point = _matrix(matrix), _point(point)
    return [_number(sum(matrix[row][col] * point[col] for col in range(3)) + matrix[row][3])
            for row in range(3)]


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    return _matrix([[sum(left[row][slot] * right[slot][col] for slot in range(4))
                     for col in range(4)] for row in range(4)])


def _inverse(matrix: Matrix) -> tuple[Matrix, float]:
    """Pivoted 3x3 elimination retains general nested affine shear."""
    rows = [matrix[row][:3] + [float(row == col) for col in range(3)] for row in range(3)]
    determinant = 1.
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(rows[row][col]))
        if rows[pivot][col] == 0:
            raise ValueError("singular_transform")
        if pivot != col:
            rows[pivot], rows[col] = rows[col], rows[pivot]
            determinant = -determinant
        value = rows[col][col]
        determinant *= value
        rows[col] = [item / value for item in rows[col]]
        for row in range(3):
            if row != col:
                factor = rows[row][col]
                rows[row] = [item - factor * other for item, other in zip(rows[row], rows[col])]
    determinant = _number(determinant)
    if determinant == 0:
        raise ValueError("singular_or_unrepresentable_determinant")
    inverse = [row[3:] + [-sum(row[3 + col] * matrix[col][3] for col in range(3))] for row in rows]
    return _matrix([*inverse, [0., 0., 0., 1.]]), determinant


def _field(fields: dict, key: str) -> Any:
    evidence = fields.get(key)
    if not isinstance(evidence, dict) or not evidence.get("occurrence_count"):
        raise ValueError(f"{key}:missing")
    occurrences = evidence.get("occurrences")
    if (evidence.get("conflicting") is not False or evidence.get("present_in_projection") is not True
            or type(evidence.get("occurrence_count")) is not int or not isinstance(occurrences, list)
            or len(occurrences) != evidence["occurrence_count"] or not evidence.get("source_pointer")):
        raise ValueError(f"{key}:conflicting_or_malformed_evidence")
    if any(not isinstance(row, dict) or _canonical(row.get("raw_value")) != _canonical(evidence.get("raw_value"))
           for row in occurrences):
        raise ValueError(f"{key}:ambiguous_occurrences")
    return evidence["raw_value"]


def _cross(left: list[float], right: list[float]) -> list[float]:
    return [left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0]]


def _normalize(vector: list[float]) -> list[float]:
    length = math.hypot(*vector)
    if not math.isfinite(length) or length == 0:
        raise ValueError("invalid_or_zero_extrusion")
    return [item / length for item in vector]


def _local_matrix(fields: dict, block_fields: dict) -> Matrix:
    insertion, scale = _point(_field(fields, "ins_pt")), _point(_field(fields, "scale"))
    angle, normal = _number(_field(fields, "rotation")), _normalize(_point(_field(fields, "extrusion")))
    base = _point(_field(block_fields, "base_pt"))
    if any(item == 0 for item in scale):
        raise ValueError("singular_zero_scale")
    if "scale_flag" in fields:
        flag = _field(fields, "scale_flag")
        if (type(flag) is not int or flag not in {0, 1, 2, 3}
                or flag == 3 and scale != [1., 1., 1.]
                or flag == 1 and scale[0] != 1.
                or flag == 2 and not scale[0] == scale[1] == scale[2]):
            raise ValueError("scale_flag_inconsistent_or_unsupported")
    # The arbitrary axis algorithm uses NORMALIZED extrusion and 1/64 threshold.
    axis = [0., 1., 0.] if abs(normal[0]) < 1 / 64 and abs(normal[1]) < 1 / 64 else [0., 0., 1.]
    x_axis = _normalize(_cross(axis, normal))
    y_axis = _normalize(_cross(normal, x_axis))
    cosine, sine = math.cos(angle), math.sin(angle)
    columns = [[scale[0] * (cosine * x_axis[row] + sine * y_axis[row]) for row in range(3)],
               [scale[1] * (-sine * x_axis[row] + cosine * y_axis[row]) for row in range(3)],
               [scale[2] * normal[row] for row in range(3)]]
    translation = [sum(axis[row] * value for axis, value in zip((x_axis, y_axis, normal), insertion))
                   - sum(columns[col][row] * base[col] for col in range(3)) for row in range(3)]
    return _matrix([[columns[col][row] for col in range(3)] + [translation[row]] for row in range(3)]
                   + [[0., 0., 0., 1.]])


def _validate_graph(graph: dict) -> tuple[dict, dict, dict]:
    _guards(graph)
    if graph.get("schema_version") != OWNERSHIP_SCHEMA or graph.get("source_kind") != "native_dwg":
        raise ValueError("A frozen native ownership graph is required")
    if graph.get("record_sha256") != _digest({key: value for key, value in graph.items() if key != "record_sha256"}):
        raise ValueError("Ownership graph no longer matches its record digest")
    for key in ("source_sha256", "response_sha256"):
        value = graph.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("Source/response SHA256 required")
    nodes = {}
    for node in graph["objects"]:
        ref = node["source_ref"]
        if any(ref.get(key) != graph[key] for key in ("source_sha256", "response_sha256")):
            raise ValueError("Cross-source object reference")
        handle = ref["native_handle"]
        if handle is not None:
            if handle in nodes:
                raise ValueError("Ambiguous native graph identity")
            nodes[handle] = node
    edges = {}
    for edge in graph["reference_edges"]:
        if type(edge.get("id")) is not int or edge["id"] in edges:
            raise ValueError("Ambiguous native reference edge")
        edges[edge["id"]] = edge
    definitions = {row["source_ref"]["native_handle"]: row for row in graph["definitions"]}
    return nodes, edges, definitions


def build_instance_transforms(ownership_graph: dict[str, Any]) -> dict[str, Any]:
    """Compute INSERT occurrences only; unknown evidence never supplies defaults."""
    nodes, edges, definitions = _validate_graph(ownership_graph)
    graph = ownership_graph
    rows = deepcopy(graph["modelspace_instances"])
    paths = [tuple(row["path_edge_ids"]) for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("Ambiguous native instance path")
    by_path, results = dict(zip(paths, rows)), {}
    for path in sorted(by_path, key=lambda item: (len(item), item)):
        occurrence = by_path[path]
        reasons, local, inverse, determinant = [], None, None, None
        result = {"path_edge_ids": list(path), "source_ref": occurrence["source_ref"],
                  "definition_source_ref": occurrence["definition_source_ref"], "status": "unknown",
                  "matrix": None, "inverse": None, "determinant": None,
                  "local_matrix": None, "local_inverse": None, "local_determinant": None,
                  "transform_field_evidence": {}, "definition_base_point_evidence": None,
                  "contents_completeness": deepcopy(occurrence.get("contents_completeness")),
                  "coordinate_convention": CONVENTION, "units": "unknown", "physical_owner": "unknown",
                  "physical_part_confirmed": False, "final_visibility": "unknown", "reasons": reasons}
        results[path] = result
        edge = edges.get(path[-1]) if path and all(type(slot) is int for slot in path) else None
        if edge is None or edge.get("relation") != "insert_definition" or edge.get("status") != "resolved_reference":
            reasons.append("insert_path_reference_unresolved_or_unsupported")
            continue
        node = nodes.get(edge["source_ref"]["native_handle"])
        definition_handle = edge["target_ref"]["native_handle"] if edge.get("target_ref") else None
        block, definition = nodes.get(definition_handle), definitions.get(definition_handle)
        if (node is None or block is None or definition is None or node["native_kind"] != "INSERT"
                or block["native_kind"] != "BLOCK_HEADER" or node["opaque_payload"] or block["opaque_payload"]
                or node["cad_context"]["status"] == "unknown"
                or definition.get("xref_status") != "internal_declared"):
            reasons.append("native_instance_or_definition_context_unverified")
            continue
        if (node["source_ref"] != edge["source_ref"] or block["source_ref"] != edge["target_ref"]
                or occurrence["source_ref"] != edge["source_ref"]
                or occurrence["definition_source_ref"] != edge["target_ref"]):
            reasons.append("path_source_evidence_mismatch")
            continue
        fields, block_fields = node["field_evidence"], block["field_evidence"]
        result["transform_field_evidence"] = {key: deepcopy(fields[key]) for key in (*_REQUIRED, "scale_flag", *_ARRAY)
                                              if key in fields}
        result["definition_base_point_evidence"] = deepcopy(block_fields.get("base_pt"))
        if any(key in fields for key in _ARRAY):
            reasons.append("array_expansion_not_supported")
        if len(path) == 1:
            if node["cad_context"]["cad_space"] != "modelspace":
                reasons.append("root_not_verified_modelspace")
        else:
            parent = results.get(path[:-1])
            parent_occurrence = by_path.get(path[:-1])
            if parent is None or parent["status"] != "verified_affine" or parent_occurrence is None:
                reasons.append("parent_transform_unresolved")
            elif (node["cad_context"]["definition_handle"] != parent_occurrence["definition_source_ref"]["native_handle"]
                  or node["source_ref"] not in parent_occurrence["member_source_refs"]):
                reasons.append("nested_parent_membership_unverified")
        try:
            local = _local_matrix(fields, block_fields)
            inverse, determinant = _inverse(local)
            result.update(local_matrix=local, local_inverse=inverse, local_determinant=determinant)
            if not reasons:
                matrix = local if len(path) == 1 else _multiply(results[path[:-1]]["matrix"], local)
                inverse, determinant = _inverse(matrix)
                result.update(status="verified_affine", matrix=matrix, inverse=inverse, determinant=determinant)
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            reasons.append(f"transform_unresolved:{error}")
    displays = [{"path_edge_ids": deepcopy(row["path_edge_ids"]), "source_ref": deepcopy(row["source_ref"]),
                 "definition_source_ref": deepcopy(row["definition_source_ref"]), "status": "unknown",
                 "matrix": None, "inverse": None, "determinant": None,
                 "reasons": ["dimension_display_transform_formula_not_implemented"],
                 "physical_owner": "unknown", "physical_part_confirmed": False}
                for row in graph["display_paths"]]
    output = {"schema_version": SCHEMA, "source_kind": "native_dwg",
              "source_sha256": graph["source_sha256"], "response_sha256": graph["response_sha256"],
              "ownership_record_sha256": graph["record_sha256"], "instances": list(results.values()),
              "display_paths": displays, "truncation": deepcopy(graph["truncation"]),
              "unresolved_source_paths": deepcopy(graph["reachability_unresolved"]),
              "unresolved_source_reference_ids": deepcopy(graph["unresolved_references"]),
              "eligible_for_route_training": False,
              "policy": {"coordinate_convention": CONVENTION, "native_rotation_unit": "radian",
                         "formula": "T(OCS*ins_pt) @ OCS @ Rz(rotation) @ S(scale) @ T(-block.base_pt)",
                         "all_transform_fields_explicit": True, "dimension_display_formula_supported": False,
                         "no_entity_coordinate_or_unit_or_physical_part_inference": True},
              "summary": {"instance_paths": len(results), "verified_affine": sum(
                  row["status"] == "verified_affine" for row in results.values()),
                  "unknown": sum(row["status"] == "unknown" for row in results.values()),
                  "dimension_display_unknown": len(displays), "truncated": graph["truncation"]["truncated"]}}
    output["record_sha256"] = _digest(output)
    return output
