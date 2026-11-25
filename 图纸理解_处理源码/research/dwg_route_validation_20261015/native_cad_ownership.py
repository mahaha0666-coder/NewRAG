"""Pure native CAD ownership/reachability evidence, never physical-part inference.

The caller verifies original DWG/response bytes. Definitions occur once; explicit
modelspace INSERT and dimension-display paths retain separate occurrences. No
coordinates are transformed and no source text becomes a manufacturing fact.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from typing import Any

from research.dwg_route_validation_20261015.duplicate_audit import SIDECAR
from research.dwg_route_validation_20261015.native_dimension_display import strict_handle
from research.dwg_route_validation_20261015.native_layer_roles import (
    IDENTITY_FIELDS, _field_evidence, _validate_lossless_structure,
)
from research.dwg_route_validation_20261015.native_relation_features import _guards
from research.dwg_route_validation_20261015.parser_probe import hex_handle, object_kind

SCHEMA = "native_cad_ownership.v1"
MAX_DEPTH = 16
MAX_PATHS = 10_000
_SPACES = {"*Model_Space": "modelspace", "*Paper_Space": "paperspace"}
_BOUNDARIES = {"BLOCK", "ENDBLK"}
_TEXT_FIELDS = ("text_value", "text", "user_text", "tag", "prompt")
_TRANSFORMS = ("ins_pt", "scale", "scale_flag", "rotation", "extrusion", "base_pt",
               "ins_scale", "ins_rotation", "elevation", "num_rows", "num_cols",
               "row_spacing", "col_spacing")
_FIELDS = ("ownerhandle", "entmode", "entities", "block_header", "block", "name",
           "invisible", "layer", "groups", "has_attribs", "attribs", "num_owned",
           "vertex", "seqend", "first_vertex", "last_vertex", "first_attrib", "last_attrib",
           "first_entity", "last_entity", "blkisxref", "xrefoverlaid", "is_xref_dep",
           "is_xref_ref", "xref", "xref_pname", "anonymous", "start", "end",
           "center", "radius", "start_angle", "end_angle") + _TEXT_FIELDS + _TRANSFORMS


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _category(kind: str | None) -> str:
    if kind in _BOUNDARIES:
        return "block_boundary"
    if kind in {"TEXT", "MTEXT"}:
        return "text"
    if kind in {"ATTRIB", "ATTDEF"}:
        return "attribute_value" if kind == "ATTRIB" else "attribute_definition"
    if kind in {"INSERT", "MINSERT"}:
        return "insert"
    if (kind or "").startswith("DIMENSION_"):
        return "dimension"
    if kind in {"LINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE", "LWPOLYLINE",
                "POLYLINE_2D", "POLYLINE_3D", "POLYLINE_MESH", "POLYLINE_PFACE",
                "SOLID", "TRACE", "POINT", "HATCH", "3DFACE", "REGION", "BODY"}:
        return "geometry"
    return "other"


def _text_role(kind: str | None, field: str) -> str:
    if field in {"tag", "prompt"}:
        return "template_metadata_not_body"
    if kind == "ATTDEF":
        return "definition_default_not_instance_value"
    if kind == "ATTRIB":
        return "attribute_instance_literal"
    if (kind, field) in {("TEXT", "text_value"), ("MTEXT", "text")}:
        return "original_text_literal"
    if (kind or "").startswith("DIMENSION_") and field == "user_text":
        return "dimension_override_literal"
    return "nonbody_metadata"


def _parent_allowed(child: str | None, parent: str | None) -> bool:
    if child == "ATTRIB":
        return parent in {"INSERT", "MINSERT"}
    if (child or "").startswith("VERTEX"):
        return (parent or "").startswith("POLYLINE")
    return child == "SEQEND" and (parent in {"INSERT", "MINSERT"} or (parent or "").startswith("POLYLINE"))


def _xref_status(fields: dict[str, dict]) -> str:
    flags = [fields.get(key, {}) for key in ("blkisxref", "xrefoverlaid", "is_xref_dep")]
    path = fields.get("xref_pname", {})
    if any(item.get("conflicting") for item in [*flags, path]):
        return "unknown_or_conflicting"
    if any(type(item.get("raw_value")) is int and item["raw_value"] != 0 for item in flags):
        return "external_or_dependent_declared"
    if isinstance(path.get("raw_value"), str) and path["raw_value"]:
        return "external_or_dependent_declared"
    if (all(type(item.get("raw_value")) is int and item["raw_value"] == 0 for item in flags)
            and path.get("raw_value") == ""):
        return "internal_declared"
    return "unknown_or_conflicting"


def build_ownership_graph(parsed: dict[str, Any], *, source_sha256: str,
                          response_sha256: str) -> dict[str, Any]:
    """Describe source-declared CAD ownership and bounded instance reachability.

    Invalid handles remain unindexed evidence; ambiguous identities reject the
    projection. Missing/older block member arrays are unknown, never empty or a
    handle range. Final visibility, units and physical ownership remain unknown.
    """
    if not isinstance(parsed, dict):
        raise ValueError("A lossless native projection is required")
    _guards(parsed)
    _validate_lossless_structure(parsed)
    for field, expected in (("source_sha256", source_sha256), ("response_sha256", response_sha256)):
        if (not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError("Original source and response SHA256 are required")
        evidence = _field_evidence(parsed, field, "#")
        if evidence["conflicting"] or (field in parsed and parsed[field] != expected):
            raise ValueError("Native source metadata hash mismatch")
    projection_sha = _digest(parsed)
    source = deepcopy(parsed)
    objects = source.get("OBJECTS")
    if not isinstance(objects, list) or any(not isinstance(obj, dict) for obj in objects):
        raise ValueError("Native OBJECTS must be an object list")
    index, nodes, source_objects = {}, [], {}
    for position, obj in enumerate(objects):
        pointer = f"#/OBJECTS/{position}"
        if any(_field_evidence(obj, key, pointer)["conflicting"] for key in IDENTITY_FIELDS):
            raise ValueError("Conflicting native object identity")
        kind = object_kind(obj)
        if kind is not None and not isinstance(kind, str):
            raise ValueError("Native object kind must remain a string")
        handle = strict_handle(obj.get("handle"), reference=False)
        if handle is not None and handle in index:
            raise ValueError("Duplicate native object handle")
        ref = {"source_sha256": source_sha256, "response_sha256": response_sha256,
               "json_pointer": pointer, "native_handle": hex_handle(handle),
               "original_handle": deepcopy(obj.get("handle"))}
        fields = {key: _field_evidence(obj, key, pointer) for key in dict.fromkeys(_FIELDS)
                  if key in obj or key in obj.get(SIDECAR, {})}
        # _field_evidence also preserves missing fields when they are required below.
        for key in ("ownerhandle", "entmode"):
            fields.setdefault(key, _field_evidence(obj, key, pointer))
        node = {"source_ref": ref, "native_kind": kind, "is_entity": "entity" in obj or "entmode" in obj,
                "category": _category(kind), "field_evidence": fields,
                "identity_status": "valid" if handle is not None else "invalid_or_missing_handle",
                "opaque_payload": obj.get("entity") == "UNKNOWN_ENT" or obj.get("object") == "UNKNOWN_OBJ"
                or bool(obj.get("unknown_bits")), "physical_owner": "unknown",
                "physical_part_confirmed": False, "wcs_coordinates": "unknown_not_transformed",
                "final_visibility": "unknown", "text_fields": [
                    {"field": key, "role": _text_role(kind, key), "evidence": fields[key]}
                    for key in _TEXT_FIELDS if key in fields]}
        if kind == "GROUP":
            ev = _field_evidence(obj, "groups", pointer)
            node["group_membership"] = {
                "status": "decoded_native_group_only" if isinstance(ev["raw_value"], list)
                and not ev["conflicting"] else "not_decoded_or_conflicting_unknown",
                "evidence": ev, "establishes_ownership": False}
        nodes.append(node)
        source_objects[pointer] = obj
        if handle is not None:
            index[handle] = node

    edges, edge_by_field = [], {}

    def reference(node: dict, field: str, relation: str, slot: int | None = None) -> dict:
        ref = node["source_ref"]
        obj = source_objects[ref["json_pointer"]]
        evidence = _field_evidence(obj, field, ref["json_pointer"])
        if slot is not None:
            evidence = {"source_pointer": f"{evidence['source_pointer']}/{slot}",
                        "raw_value": deepcopy(obj[field][slot]), "conflicting": False,
                        "container_occurrence_count": evidence["occurrence_count"],
                        "occurrences": [{**row, "raw_value": deepcopy(row["raw_value"][slot]),
                                         "source_pointer": f"{row['source_pointer']}/{slot}",
                                         "sidecar_pointer": (f"{row['sidecar_pointer']}/{slot}"
                                                             if row['sidecar_pointer'] else None)}
                                        for row in evidence["occurrences"]]}
        origin = strict_handle(ref["original_handle"], reference=False)
        target_handle = strict_handle(evidence["raw_value"], reference=True, origin=origin)
        target = index.get(target_handle)
        reasons = []
        if origin is None:
            reasons.append("invalid_source_identity")
        if evidence["conflicting"]:
            reasons.append("conflicting_reference")
        elif target_handle is None:
            reasons.append("missing_null_or_invalid_reference")
        elif target is None:
            reasons.append("unresolved_handle")
        edge = {"id": len(edges), "relation": relation, "reference_field": field, "source_ref": ref,
                "target_ref": target["source_ref"] if target else None,
                "target_kind": target["native_kind"] if target else None,
                "field_evidence": evidence, "status": "unresolved" if reasons else "resolved_reference",
                "validation_reasons": reasons}
        edges.append(edge)
        if slot is None:
            edge_by_field[(ref["native_handle"], field)] = edge
        return edge

    definitions, members_by_block = [], {}
    for node in nodes:
        kind, fields = node["native_kind"], node["field_evidence"]
        if node["is_entity"] and (fields["ownerhandle"]["occurrence_count"]
                                  or fields["entmode"]["raw_value"] == 0):
            reference(node, "ownerhandle", "cad_owner")
        if kind in {"INSERT", "MINSERT"}:
            reference(node, "block_header", "insert_definition")
        if (kind or "").startswith("DIMENSION_"):
            reference(node, "block", "dimension_display_definition")
        child_array = "attribs" if kind in {"INSERT", "MINSERT"} else (
            "vertex" if (kind or "").startswith("POLYLINE") else None)
        if child_array is not None:
            for field in (child_array, "seqend"):
                evidence = _field_evidence(source_objects[node["source_ref"]["json_pointer"]],
                                           field, node["source_ref"]["json_pointer"])
                if not evidence["occurrence_count"]:
                    continue
                if field == child_array and isinstance(evidence["raw_value"], list) and not evidence["conflicting"]:
                    for slot in range(len(evidence["raw_value"])):
                        reference(node, field, "parent_subentity", slot)
                else:
                    reference(node, field, "parent_subentity")
        if kind in {"BLOCK_HEADER", "GROUP"}:
            field = "entities" if kind == "BLOCK_HEADER" else "groups"
            evidence = _field_evidence(source_objects[node["source_ref"]["json_pointer"]],
                                       field, node["source_ref"]["json_pointer"])
            refs = evidence["raw_value"]
            member_edges = []
            if evidence["occurrence_count"]:
                if isinstance(refs, list) and not evidence["conflicting"]:
                    member_edges = [reference(node, field, "block_member" if kind == "BLOCK_HEADER"
                                              else "native_group_member", slot) for slot in range(len(refs))]
                else:
                    member_edges = [reference(node, field, "block_member" if kind == "BLOCK_HEADER"
                                              else "native_group_member")]
            if kind == "BLOCK_HEADER":
                handle = node["source_ref"]["native_handle"]
                name = fields.get("name", {})
                definitions.append({"source_ref": node["source_ref"], "name_evidence": name,
                                    "declared_native_space": _SPACES.get(name.get("raw_value"), "unknown")
                                    if not name.get("conflicting") and isinstance(name.get("raw_value"), str)
                                    else "unknown", "xref_status": _xref_status(fields),
                                    "members_evidence": evidence,
                                    "member_edge_ids": [edge["id"] for edge in member_edges],
                                    "physical_part_confirmed": False})
                members_by_block[handle] = member_edges
    definition_by_handle = {row["source_ref"]["native_handle"]: row for row in definitions
                            if row["source_ref"]["native_handle"] is not None}

    for definition in definitions:
        handle = definition["source_ref"]["native_handle"]
        member_edges = members_by_block[handle]
        frequencies = Counter(edge["target_ref"]["native_handle"] for edge in member_edges if edge["target_ref"])
        for edge in member_edges:
            if edge["status"] != "resolved_reference":
                continue
            target = index[int(edge["target_ref"]["native_handle"], 16)]
            entmode = target["field_evidence"]["entmode"]["raw_value"]
            owner = edge_by_field.get((target["source_ref"]["native_handle"], "ownerhandle"))
            reciprocal = owner is not None and owner["status"] == "resolved_reference" and (
                owner["target_ref"]["native_handle"] == handle)
            implicit = (type(entmode) is int and (
                (entmode == 2 and definition["declared_native_space"] == "modelspace")
                or (entmode == 1 and definition["declared_native_space"] == "paperspace"))
                and (not target["field_evidence"]["ownerhandle"]["occurrence_count"] or reciprocal))
            if (not target["is_entity"] or frequencies[edge["target_ref"]["native_handle"]] != 1
                    or not (type(entmode) is int and entmode == 0 and reciprocal or implicit)):
                edge["status"] = "unresolved"
                edge["validation_reasons"].append("block_member_owner_or_multiplicity_mismatch")
            else:
                edge["status"] = "verified_membership"

    parent_frequencies = Counter((edge["source_ref"]["native_handle"], edge["reference_field"],
                                  edge["target_ref"]["native_handle"]) for edge in edges
                                 if edge["relation"] == "parent_subentity" and edge["target_ref"])
    for edge in edges:
        if edge["relation"] != "parent_subentity" or edge["status"] != "resolved_reference":
            continue
        parent_handle = edge["source_ref"]["native_handle"]
        child_handle = edge["target_ref"]["native_handle"]
        child = index[int(child_handle, 16)]
        parent = index[int(parent_handle, 16)]
        owner = edge_by_field.get((child_handle, "ownerhandle"))
        reciprocal = owner is not None and owner["status"] == "resolved_reference" and (
            owner["target_ref"]["native_handle"] == parent_handle)
        field = edge["reference_field"]
        expected_field = "attribs" if child["native_kind"] == "ATTRIB" else (
            "seqend" if child["native_kind"] == "SEQEND" else "vertex")
        mode = child["field_evidence"]["entmode"]["raw_value"]
        if (not reciprocal or type(mode) is not int or mode != 0 or field != expected_field
                or not _parent_allowed(child["native_kind"], parent["native_kind"])
                or parent_frequencies[(parent_handle, field, child_handle)] != 1):
            edge["status"] = "unresolved"
            edge["validation_reasons"].append("parent_subentity_owner_kind_or_multiplicity_mismatch")
        else:
            edge["status"] = "verified_parent_membership"

    incoming_memberships = defaultdict(list)
    for edge in edges:
        if edge["relation"] in {"block_member", "parent_subentity"} and edge["target_ref"] is not None:
            incoming_memberships[edge["target_ref"]["native_handle"]].append(edge)

    def parent_check(child: dict, parent: dict) -> dict:
        kind, fields = child["native_kind"], parent["field_evidence"]
        field = "attribs" if kind == "ATTRIB" else ("seqend" if kind == "SEQEND" else "vertex")
        evidence = fields.get(field)
        reasons = []
        if kind != "SEQEND":
            for count_field in (("num_owned", "has_attribs") if kind == "ATTRIB" else ("num_owned",)):
                count = fields.get(count_field, {})
                value = count.get("raw_value")
                if count and (count.get("conflicting") or type(value) is not int or value <= 0):
                    reasons.append("parent_declares_no_or_conflicting_subentities")
        if evidence is not None:
            origin = int(parent["source_ref"]["native_handle"], 16)
            child_handle = int(child["source_ref"]["native_handle"], 16)
            raw = evidence["raw_value"]
            if evidence["conflicting"]:
                reasons.append("parent_subentity_reference_conflicting")
            elif field == "seqend":
                if strict_handle(raw, reference=True, origin=origin) != child_handle:
                    reasons.append("parent_seqend_not_reciprocal")
            elif (not isinstance(raw, list) or sum(strict_handle(ref, reference=True, origin=origin)
                                                 == child_handle for ref in raw) != 1):
                reasons.append("parent_subentity_membership_not_reciprocal")
            if field != "seqend" and isinstance(raw, list) and "num_owned" in fields:
                count = fields["num_owned"]
                if count["conflicting"] or type(count["raw_value"]) is not int or count["raw_value"] != len(raw):
                    reasons.append("parent_subentity_count_mismatch_or_invalid")
        return {"parent_source_ref": parent["source_ref"], "field": field,
                "status": "contradictory" if reasons else (
                    "reciprocal_verified" if evidence is not None else "owner_declared_only"),
                "reasons": reasons}

    def context(node: dict) -> dict:
        chain, reasons, current, seen = [], [], node, set()
        parent_checks = []
        definition_handle, space, status = None, "unknown", "unknown"
        while current is not None:
            handle = current["source_ref"]["native_handle"]
            if handle is None:
                reasons.append("invalid_or_missing_identity")
                break
            if handle in seen:
                reasons.append("owner_cycle")
                break
            seen.add(handle)
            chain.append(handle)
            if any(edge["status"] not in {"verified_membership", "verified_parent_membership"}
                   for edge in incoming_memberships[handle]):
                reasons.append("contradictory_declared_membership")
            if len(chain) > MAX_DEPTH:
                reasons.append("owner_depth_limit")
                break
            mode = current["field_evidence"]["entmode"]["raw_value"]
            if type(mode) is not int or mode not in {0, 1, 2}:
                reasons.append("entmode_missing_or_unsupported")
                break
            if mode in {1, 2}:
                space = "modelspace" if mode == 2 else "paperspace"
                explicit_owner = current["field_evidence"]["ownerhandle"]
                if explicit_owner["occurrence_count"]:
                    edge = edge_by_field.get((handle, "ownerhandle"))
                    declared = (definition_by_handle.get(edge["target_ref"]["native_handle"], {})
                                if edge and edge["target_ref"] else {})
                    if (edge is None or edge["status"] != "resolved_reference"
                            or declared.get("declared_native_space") != space):
                        reasons.append("implicit_space_explicit_owner_contradiction")
                status = space + ("_implicit" if current is node else "_parent_owned")
                break
            edge = edge_by_field.get((handle, "ownerhandle"))
            if edge is None or edge["status"] != "resolved_reference":
                reasons.append("owner_missing_conflicting_or_unresolved")
                break
            owner_handle = edge["target_ref"]["native_handle"]
            owner = index[int(owner_handle, 16)]
            if owner["native_kind"] == "BLOCK_HEADER":
                definition = definition_by_handle[owner_handle]
                membership = definition["members_evidence"]
                chain.append(owner_handle)
                if current["native_kind"] not in _BOUNDARIES and membership["occurrence_count"]:
                    matches = [member for member in members_by_block[owner_handle]
                               if member["target_ref"] and member["target_ref"]["native_handle"] == handle]
                    if len(matches) != 1 or matches[0]["status"] != "verified_membership":
                        reasons.append("owner_block_membership_not_reciprocal")
                        break
                definition_handle = owner_handle
                space = definition["declared_native_space"]
                status = (space + "_declared_owner" if space != "unknown" else "block_definition_owned")
                break
            if not owner["is_entity"]:
                reasons.append("owner_target_not_entity_or_block")
                break
            if not _parent_allowed(current["native_kind"], owner["native_kind"]):
                reasons.append("unsupported_parent_entity_ownership")
            else:
                check = parent_check(current, owner)
                parent_checks.append(check)
                reasons.extend(check["reasons"])
            current = owner
        if reasons:
            status, space, definition_handle = "unknown", "unknown", None
        return {"status": status, "cad_space": space, "definition_handle": definition_handle,
                "owner_chain": chain, "parent_membership_checks": parent_checks,
                "reasons": sorted(set(reasons)), "physical_owner": "unknown"}

    owned = defaultdict(list)
    for node in nodes:
        node["cad_context"] = context(node) if node["is_entity"] else {
            "status": "not_entity", "cad_space": "unknown", "definition_handle": None,
            "owner_chain": [], "parent_membership_checks": [], "reasons": [], "physical_owner": "unknown"}
        cad = node["cad_context"]
        if cad["status"] == "block_definition_owned" and node["category"] != "block_boundary":
            owned[cad["definition_handle"]].append(node)

    for definition in definitions:
        membership = definition["members_evidence"]
        handle = definition["source_ref"]["native_handle"]
        reasons = []
        if handle is None:
            reasons.append("invalid_definition_identity")
        if not membership["occurrence_count"]:
            reasons.append("member_list_not_supplied")
        elif membership["conflicting"] or not isinstance(membership["raw_value"], list):
            reasons.append("member_list_conflicting_or_invalid")
        if any(edge["status"] != "verified_membership" for edge in members_by_block[handle]):
            reasons.append("declared_member_mismatch_or_unresolved")
        count = index[int(handle, 16)]["field_evidence"].get("num_owned") if handle else None
        if count is not None and (count["conflicting"] or type(count["raw_value"]) is not int
                                  or count["raw_value"] < 0 or not isinstance(membership["raw_value"], list)
                                  or count["raw_value"] != len(membership["raw_value"])):
            reasons.append("member_count_mismatch_or_invalid")
        if any(node["cad_context"]["status"] == "unknown" and handle in node["cad_context"]["owner_chain"]
               for node in nodes if node["is_entity"]):
            reasons.append("owned_entity_context_unresolved")
        definition["contents_completeness"] = {
            "status": "unknown" if reasons else "declared_list_consistent",
            "scope": "returned_native_object_membership_only", "reasons": reasons}

    instances, displays, reachability_unresolved = [], [], []
    owner_limits = [{"reason": "owner_depth_limit", "source_ref": node["source_ref"]}
                    for node in nodes if "owner_depth_limit" in node["cad_context"]["reasons"]]
    truncation = {"truncated": bool(owner_limits), "max_depth": MAX_DEPTH, "max_paths": MAX_PATHS,
                  "events": owner_limits[:100], "events_omitted": max(0, len(owner_limits) - 100)}

    def truncate(reason: str, path: list[int], edge: dict) -> None:
        truncation["truncated"] = True
        event = {"reason": reason, "edge_id": edge["id"], "parent_path_edge_ids": path}
        if len(truncation["events"]) < 100:
            truncation["events"].append(event)
        else:
            truncation["events_omitted"] += 1

    def visit(edge: dict, path: list[int], stack: tuple[str, ...], display: bool) -> None:
        if len(path) >= MAX_DEPTH:
            truncate("instance_depth_limit", path, edge)
            return
        if len(instances) + len(displays) >= MAX_PATHS:
            truncate("instance_path_limit", path, edge)
            return
        definition_handle = edge["target_ref"]["native_handle"] if edge["target_ref"] else None
        definition = definition_by_handle.get(definition_handle)
        origin = index.get(int(edge["source_ref"]["native_handle"], 16))
        reasons = list(edge["validation_reasons"])
        if definition is None:
            reasons.append("target_not_resolved_block_definition")
        elif definition["xref_status"] != "internal_declared":
            reasons.append("xref_or_internal_definition_unverified")
        elif definition["declared_native_space"] != "unknown":
            reasons.append("space_definition_not_insert_content")
        if definition_handle in stack:
            reasons.append("block_definition_cycle")
        if origin is not None and (origin["native_kind"] == "MINSERT" or any(
                field in origin["field_evidence"] for field in ("num_rows", "num_cols", "row_spacing", "col_spacing"))):
            reasons.append("array_instance_expansion_unknown")
        if origin is None or origin["opaque_payload"]:
            reasons.append("opaque_or_invalid_instance_source")
        if reasons:
            reachability_unresolved.append({"edge_id": edge["id"], "parent_path_edge_ids": path,
                                           "reasons": sorted(set(reasons))})
            return
        display = display or edge["relation"] == "dimension_display_definition"
        next_path = [*path, edge["id"]]
        members = owned[definition_handle]
        counts = Counter(member["category"] for member in members)
        if display and "text" in counts:
            counts["display_block_text"] = counts.pop("text")
        record = {"path_edge_ids": next_path, "source_ref": edge["source_ref"],
                  "definition_source_ref": definition["source_ref"],
                  "role": "dimension_display" if display else "insert_instance",
                  "member_source_refs": [member["source_ref"] for member in members],
                  "member_categories": dict(sorted(counts.items())),
                  "contents_completeness": definition["contents_completeness"],
                  "raw_transform_fields": {key: origin["field_evidence"][key]
                                           for key in _TRANSFORMS if key in origin["field_evidence"]},
                  "definition_base_point_evidence": index[int(definition_handle, 16)]["field_evidence"].get("base_pt"),
                  "wcs_transform_status": "unknown_not_applied", "final_visibility": "unknown",
                  "physical_owner": "unknown", "physical_part_confirmed": False}
        (displays if display else instances).append(record)
        for member in members:
            field = "block_header" if member["native_kind"] in {"INSERT", "MINSERT"} else (
                "block" if member["category"] == "dimension" else None)
            child = edge_by_field.get((member["source_ref"]["native_handle"], field))
            if child is not None:
                visit(child, next_path, (*stack, definition_handle), display)

    for node in nodes:
        if node["cad_context"]["cad_space"] != "modelspace":
            continue
        field = "block_header" if node["native_kind"] in {"INSERT", "MINSERT"} else (
            "block" if node["category"] == "dimension" else None)
        edge = edge_by_field.get((node["source_ref"]["native_handle"], field))
        if edge is not None:
            visit(edge, [], (), False)
    unresolved = [edge["id"] for edge in edges if edge["status"] == "unresolved"]
    result = {"schema_version": SCHEMA, "source_kind": "native_dwg",
              "source_sha256": source_sha256, "response_sha256": response_sha256,
              "projection_sha256": projection_sha, "objects": nodes, "definitions": definitions,
              "reference_edges": edges, "modelspace_instances": instances, "display_paths": displays,
              "unresolved_references": unresolved, "reachability_unresolved": reachability_unresolved,
              "truncation": truncation, "processing_status": "truncated" if truncation["truncated"]
              else "graph_built_with_unresolved_evidence", "units": "unknown", "physical_owner": "unknown",
              "eligible_for_route_training": False, "original_byte_hashes_verified_by_consumer": False,
              "summary": {"entity_contexts": dict(sorted(Counter(node["cad_context"]["status"] for node in nodes
                                                               if node["is_entity"]).items())),
                          "entity_categories": dict(sorted(Counter(node["category"] for node in nodes
                                                                   if node["is_entity"]).items())),
                          "definitions": len(definitions), "verified_owner_memberships": sum(
                              edge["status"] == "verified_membership" for edge in edges),
                          "modelspace_instances": len(instances), "display_paths": len(displays),
                          "reachable_instance_categories": dict(sum((Counter(row["member_categories"])
                                                                      for row in instances), Counter())),
                          "display_categories": dict(sum((Counter(row["member_categories"])
                                                           for row in displays), Counter())),
                          "unresolved_references": len(unresolved),
                          "truncated": truncation["truncated"]}}
    result["record_sha256"] = _digest(result)
    return result
