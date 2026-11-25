"""Project exact declared native layer names without manufacturing inference.

The caller must pin and verify original DWG/response bytes. This pure projection
checks the supplied lossless OBJECTS against an independently rebuilt native IR;
it cannot recompute either byte hash from an already parsed dictionary.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from research.dwg_route_validation_20261015.duplicate_audit import SIDECAR
from research.dwg_route_validation_20261015.native_ir import build_ir
from research.dwg_route_validation_20261015.parser_probe import (
    handle_value, hex_handle, object_kind,
)


DECLARED_LAYER_ROLES = {
    "01 轮廓实线层": "contour_solid_declared",
    "02 细实线层": "thin_solid_declared",
    "03 中心线层": "centerline_declared",
    "04 虚线层": "hidden_line_declared",
    "05 剖面线层": "hatch_declared",
    "06 文字层": "text_declared",
    "07 标注层": "annotation_declared",
}
IDENTITY_FIELDS = ("handle", "entity", "object", "dxfname", "_source_class_dxfname", "entmode")


def _encoding(value: Any) -> str:
    """Compare JSON values without conflating bool/int or int/float."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _field_evidence(obj: dict[str, Any], field: str, pointer: str) -> dict[str, Any]:
    repeated = obj.get(SIDECAR, {})
    if not isinstance(repeated, dict):
        raise ValueError("Malformed native duplicate occurrence sidecar")
    in_sidecar = field in repeated
    if in_sidecar:
        values = repeated[field]
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError("Malformed native field occurrence list")
        conflict = len({_encoding(value) for value in values}) != 1
        if (conflict and field in obj) or (not conflict and (
                field not in obj or _encoding(obj[field]) != _encoding(values[0]))):
            raise ValueError("Native field disagrees with lossless duplicate sidecar")
    else:
        values = [obj[field]] if field in obj else []
        conflict = False
    field_pointer = f"{pointer}/{_token(field)}" if values else None
    return {
        "source_pointer": field_pointer,
        "present_in_projection": field in obj,
        "raw_value": deepcopy(obj.get(field)),
        "occurrence_count": len(values),
        "duplicate_sidecar_present": in_sidecar,
        "conflicting": conflict,
        "occurrences": [
            {"occurrence_ordinal": slot, "raw_value": deepcopy(value),
             "source_pointer": field_pointer,
             "sidecar_pointer": (f"{pointer}/{SIDECAR}/{_token(field)}/{slot}"
                                 if in_sidecar else None)}
            for slot, value in enumerate(values)
        ],
    }


def _validate_lossless_structure(value: Any, pointer: str = "#") -> None:
    """Reject restored conflicting keys, including native relationship fields."""
    if isinstance(value, dict):
        repeated = value.get(SIDECAR, {})
        if not isinstance(repeated, dict):
            raise ValueError("Malformed native duplicate occurrence sidecar")
        for field in repeated:
            if not isinstance(field, str) or field == SIDECAR:
                raise ValueError("Malformed native duplicate occurrence field")
            _field_evidence(value, field, pointer)
        for field, child in value.items():
            if not isinstance(field, str):
                raise ValueError("Native JSON field names must be strings")
            _validate_lossless_structure(child, f"{pointer}/{_token(field)}")
    elif isinstance(value, list):
        for slot, child in enumerate(value):
            _validate_lossless_structure(child, f"{pointer}/{slot}")


def _validated_objects(parsed: dict[str, Any], ir: dict[str, Any]) -> dict[int, tuple[int, dict[str, Any]]]:
    if not isinstance(parsed, dict) or not isinstance(ir, dict):
        raise ValueError("A lossless native projection and native IR are required")
    _validate_lossless_structure(parsed)
    objects = parsed.get("OBJECTS")
    if not isinstance(objects, list) or any(not isinstance(obj, dict) for obj in objects):
        raise ValueError("Native OBJECTS must be an unambiguous object list")
    if (ir.get("schema_version") != "native_dwg_evidence_ir.v1"
            or ir.get("source_kind") != "native_dwg"):
        raise ValueError("The native build_ir schema and source kind are required")
    for field in ("source_sha256", "response_sha256"):
        value = ir.get(field)
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError("Original source and response SHA256 are required")
        evidence = _field_evidence(parsed, field, "#")
        if evidence["conflicting"] or (field in parsed and parsed[field] != value):
            raise ValueError("Parsed source metadata and IR hashes disagree")
    index: dict[int, tuple[int, dict[str, Any]]] = {}
    for position, obj in enumerate(objects):
        pointer = f"#/OBJECTS/{position}"
        for field in IDENTITY_FIELDS:
            if _field_evidence(obj, field, pointer)["conflicting"]:
                raise ValueError(f"Conflicting native identity field: {pointer}/{field}")
        kind = object_kind(obj)
        if kind is not None and not isinstance(kind, str):
            raise ValueError("Native object kind must remain a string")
        handle = handle_value(obj.get("handle"))
        if handle is not None:
            if handle in index:
                raise ValueError(f"Duplicate native object handle {handle:X}")
            index[handle] = (position, obj)
    if not isinstance(ir.get("item_code"), str):
        raise ValueError("Native IR item_code must remain a string")
    rebuilt = build_ir(parsed, item_code=ir["item_code"],
                       source_sha256=ir["source_sha256"], response_sha256=ir["response_sha256"])
    if _encoding(ir) != _encoding(rebuilt):
        raise ValueError("Native IR does not match the supplied lossless source projection")
    for collection in ("dimensions", "geometry_nodes"):
        for node in ir[collection]:
            obj = index[int(node["id"], 16)][1]
            if type(obj.get("entmode")) is not int or obj["entmode"] != 2:
                raise ValueError("Projected native modelspace entmode must be integer 2")
    return index


def _layer_record(node: dict[str, Any], collection: str,
                  index: dict[int, tuple[int, dict[str, Any]]],
                  source_sha256: str, response_sha256: str) -> dict[str, Any]:
    position, obj = index[int(node["id"], 16)]
    pointer = f"#/OBJECTS/{position}"
    reference = _field_evidence(obj, "layer", pointer)
    target_handle = handle_value(reference["raw_value"])
    row = {
        "node_id": node["id"], "node_collection": collection,
        "native_kind": node["native_kind"], "native_entmode": node["native_entmode"],
        "source_sha256": source_sha256, "response_sha256": response_sha256,
        "source_ref": deepcopy(node["source_ref"]),
        "cad_coordinates": deepcopy(node["cad_coordinates"]),
        "raw_layer_reference": reference["raw_value"],
        "layer_reference_source_pointer": reference["source_pointer"],
        "layer_reference_evidence": reference,
        "layer_reference_handle": hex_handle(target_handle),
        "layer_target_source_ref": None, "layer_target_native_kind": None,
        "layer_object_handle": None, "raw_layer_object_handle": None,
        "layer_object_handle_evidence": None, "raw_layer_name": None,
        "layer_name_source_pointer": None, "layer_name_evidence": None,
        "declared_layer_role": "unknown", "status": "unknown", "reasons": [],
        "physical_owner": "unknown", "unit": "unknown", "relationship_confidence": None,
    }
    if reference["conflicting"]:
        row["reasons"].append("conflicting_layer_reference_field")
        return row
    if reference["occurrence_count"] == 0:
        row["reasons"].append("missing_layer_reference")
        return row
    if target_handle is None:
        row["reasons"].append("null_or_invalid_layer_reference")
        return row
    target = index.get(target_handle)
    if target is None:
        row["reasons"].append("unresolved_layer_handle")
        return row
    target_position, layer = target
    layer_pointer = f"#/OBJECTS/{target_position}"
    row["layer_target_source_ref"] = {
        "source_sha256": source_sha256, "response_sha256": response_sha256,
        "json_pointer": layer_pointer, "native_handle": hex_handle(target_handle),
    }
    row["layer_target_native_kind"] = object_kind(layer)
    if object_kind(layer) != "LAYER":
        row["reasons"].append("layer_reference_target_is_not_LAYER")
        return row
    name = _field_evidence(layer, "name", layer_pointer)
    layer_identity = _field_evidence(layer, "handle", layer_pointer)
    row.update(layer_object_handle=hex_handle(target_handle),
               raw_layer_object_handle=layer_identity["raw_value"],
               layer_object_handle_evidence=layer_identity, raw_layer_name=name["raw_value"],
               layer_name_source_pointer=name["source_pointer"], layer_name_evidence=name)
    if name["conflicting"]:
        row["reasons"].append("conflicting_layer_name_field")
    elif not isinstance(name["raw_value"], str) or name["raw_value"] == "":
        row["reasons"].append("missing_empty_or_nonstring_layer_name")
    elif name["raw_value"] in DECLARED_LAYER_ROLES:
        row["declared_layer_role"] = DECLARED_LAYER_ROLES[name["raw_value"]]
        row["status"] = "resolved_known_layer"
    else:
        row["status"] = "resolved_unknown_layer_name"
        row["reasons"].append("layer_name_outside_exact_whitelist")
    return row


def project_layers(parsed: dict[str, Any], ir: dict[str, Any]) -> dict[str, Any]:
    """Return independent layer evidence for every IR node and existing direct edge.

    Duplicate positions are field pointers plus ordinals and sidecar pointers,
    never guessed original byte offsets. Coordinates are copied verbatim.
    """
    index = _validated_objects(parsed, ir)
    dimensions = [_layer_record(node, "dimensions", index, ir["source_sha256"], ir["response_sha256"])
                  for node in ir["dimensions"]]
    geometry = [_layer_record(node, "geometry_nodes", index, ir["source_sha256"], ir["response_sha256"])
                for node in ir["geometry_nodes"]]
    dimensions_by_id = {node["node_id"]: node for node in dimensions}
    geometry_by_id = {node["node_id"]: node for node in geometry}
    relations = [
        dict(deepcopy(edge), dimension_layer=deepcopy(dimensions_by_id[edge["dimension_id"]]),
             geometry_layer=deepcopy(geometry_by_id[edge["geometry_id"]]))
        for edge in ir["relations"] if edge["status"] == "verified_direct_modelspace"
    ]
    return {
        "schema_version": "native_declared_layer_roles.v1", "item_code": ir["item_code"],
        "source_sha256": ir["source_sha256"], "response_sha256": ir["response_sha256"],
        "dimensions": dimensions, "geometry_nodes": geometry, "relations": relations,
        "layer_name_match_policy": "exact_native_name_only",
        "source_validation": "native_IR_rebuilt_from_supplied_lossless_projection",
        "original_byte_hashes_verified_by_projection": False,
        "coordinate_transform_performed": False, "eligible_for_route_training": False,
        "capability_limits": ["not_physical_part_ownership", "not_current_part_identity",
                              "not_hole_detection", "not_process_evidence", "not_dimension_units"],
    }
