"""Project source-bound INSERT attribute rows, without physical-part inference.

An accepted row means that original ATTRIB literals have a complete, reciprocal
CAD parent relation. It does not identify a physical part or validate its code.
The caller still verifies original DWG/response bytes independently.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from typing import Any

from research.dwg_route_validation_20261015.native_dimension_display import strict_handle

SCHEMA = "native_attribute_rows.v1"
TAGS = ("代号", "序号", "名称", "材料", "数量", "备注", "品牌", "规格")
CORE_TAGS = frozenset({"代号", "序号", "名称", "数量"})
_CONTEXTS = {"modelspace_implicit", "modelspace_parent_owned", "modelspace_declared_owner",
             "paperspace_implicit", "paperspace_parent_owned", "paperspace_declared_owner",
             "block_definition_owned"}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _evidence(node: dict, field: str) -> dict:
    return node["field_evidence"].get(field, {
        "source_pointer": None, "present_in_projection": False, "raw_value": None,
        "occurrence_count": 0, "duplicate_sidecar_present": False,
        "conflicting": False, "occurrences": [],
    })


def _values(evidence: dict) -> list[Any]:
    return [entry["raw_value"] for entry in evidence["occurrences"]]


def _context_valid(node: dict) -> bool:
    context = node["cad_context"]
    return context["status"] in _CONTEXTS and not context["reasons"]


def _index(graph: dict) -> tuple[dict[str, dict], dict[str, dict], dict[str, list], dict[str, list], dict[str, list]]:
    if (graph.get("schema_version") != "native_cad_ownership.v1"
            or graph.get("source_kind") != "native_dwg" or graph.get("units") != "unknown"
            or graph.get("physical_owner") != "unknown" or graph.get("eligible_for_route_training") is not False
            or graph.get("record_sha256") != _digest({key: value for key, value in graph.items()
                                                      if key != "record_sha256"})):
        raise ValueError("An unchanged native ownership graph is required")
    for field in ("source_sha256", "response_sha256", "projection_sha256"):
        value = graph.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Original native source/projection SHA256 pins are required")
    nodes, pointers = {}, {}
    for position, node in enumerate(graph["objects"]):
        ref = node["source_ref"]
        handle = strict_handle(ref["original_handle"], reference=False)
        if (ref["source_sha256"] != graph["source_sha256"]
                or ref["response_sha256"] != graph["response_sha256"]
                or ref["json_pointer"] != f"#/OBJECTS/{position}"
                or ref["native_handle"] != (f"{handle:X}" if handle is not None else None)):
            raise ValueError("Native object source identity changed")
        pointers[ref["json_pointer"]] = node
        if handle is not None:
            if ref["native_handle"] in nodes:
                raise ValueError("Duplicate native object identity")
            nodes[ref["native_handle"]] = node
    parents, owners, definitions = defaultdict(list), defaultdict(list), defaultdict(list)
    for edge in graph["reference_edges"]:
        for ref in (edge["source_ref"], edge["target_ref"]):
            if ref is not None and (ref["json_pointer"] not in pointers
                                    or pointers[ref["json_pointer"]]["source_ref"] != ref):
                raise ValueError("Reference edge source identity disagrees with its object")
        pointer = edge["source_ref"]["json_pointer"]
        if edge["relation"] == "parent_subentity" and edge["reference_field"] == "attribs":
            parents[pointer].append(edge)
        elif edge["relation"] == "cad_owner":
            owners[pointer].append(edge)
        elif edge["relation"] == "insert_definition":
            definitions[pointer].append(edge)
    return nodes, pointers, parents, owners, definitions


def project_attribute_rows(ownership_graph: dict[str, Any]) -> dict[str, Any]:
    """Keep exact tagged instance literals and unknown/incomplete row candidates.

    Core tag presence selects a row candidate, including declared empty strings.
    Complete nonrow groups remain separate; incomplete groups cannot establish
    that a missing core tag was never declared. No numeric or name normalization
    is performed, and ATTDEF defaults never supply instance values.
    """
    graph = ownership_graph
    nodes, pointers, parents, owners, definitions = _index(graph)
    owned = defaultdict(set)
    for pointer, node in pointers.items():
        if node["native_kind"] != "ATTRIB":
            continue
        origin = strict_handle(node["source_ref"]["original_handle"], reference=False)
        for raw in _values(_evidence(node, "ownerhandle")):
            target = strict_handle(raw, reference=True, origin=origin)
            if target is not None:
                owned[f"{target:X}"].add(pointer)

    rows, nonrows, used_attributes = [], [], set()
    insert_count = 0
    for pointer, parent in pointers.items():
        if parent["native_kind"] != "INSERT":
            continue
        insert_count += 1
        handle = parent["source_ref"]["native_handle"]
        origin = strict_handle(parent["source_ref"]["original_handle"], reference=False)
        array = _evidence(parent, "attribs")
        flag = _evidence(parent, "has_attribs")
        count = _evidence(parent, "num_owned")
        if (origin is not None and not array["occurrence_count"] and not parents[pointer] and not owned[handle]
                and not flag["conflicting"] and type(flag["raw_value"]) is int and flag["raw_value"] == 0
                and (not count["occurrence_count"] or not count["conflicting"]
                     and type(count["raw_value"]) is int and count["raw_value"] == 0)):
            continue
        reasons = []
        if origin is None:
            reasons.append("invalid_parent_identity")
        if not _context_valid(parent):
            reasons.append("parent_cad_context_unknown")
        if graph["truncation"]["truncated"]:
            reasons.append("ownership_graph_processing_truncated")
        if not array["occurrence_count"] or array["conflicting"] or not isinstance(array["raw_value"], list):
            reasons.append("attribute_array_missing_conflicting_or_invalid")
            raw_refs = []
        else:
            raw_refs = array["raw_value"]
        if flag["occurrence_count"] and (flag["conflicting"] or type(flag["raw_value"]) is not int
                                         or flag["raw_value"] != bool(raw_refs)):
            reasons.append("has_attribs_disagrees_with_array")
        if count["occurrence_count"] and (count["conflicting"] or type(count["raw_value"]) is not int
                                          or count["raw_value"] != len(raw_refs)):
            reasons.append("attribute_count_mismatch_or_invalid")
        definition_edges = definitions[pointer]
        definition_ref = None
        if (len(definition_edges) == 1 and definition_edges[0]["status"] == "resolved_reference"
                and definition_edges[0]["target_kind"] == "BLOCK_HEADER"):
            definition_ref = definition_edges[0]["target_ref"]
        else:
            reasons.append("insert_definition_unresolved_or_ambiguous")
        declared = []
        for raw in raw_refs:
            target = strict_handle(raw, reference=True, origin=origin)
            target_handle = f"{target:X}" if target is not None else None
            target_node = nodes.get(target_handle)
            if target_node is None or target_node["native_kind"] != "ATTRIB":
                reasons.append("attribute_target_missing_invalid_or_not_ATTRIB")
            else:
                declared.append(target_node["source_ref"]["json_pointer"])
        if len(set(declared)) != len(declared):
            reasons.append("duplicate_attribute_reference")
        if set(declared) != owned[handle]:
            reasons.append("attribute_array_owner_claims_incomplete_or_contradictory")
        candidates = set(declared) | owned[handle]
        candidates.update(edge["target_ref"]["json_pointer"] for edge in parents[pointer]
                          if edge["target_ref"] is not None
                          and edge["target_kind"] == "ATTRIB")
        attributes = []
        for child_pointer in sorted(candidates, key=lambda value: int(value.rsplit("/", 1)[-1])):
            child = pointers[child_pointer]
            child_origin = strict_handle(child["source_ref"]["original_handle"], reference=False)
            child_reasons = []
            if child_origin is None:
                child_reasons.append("invalid_attribute_identity")
            matches = [edge for edge in parents[pointer] if edge["target_ref"] == child["source_ref"]]
            reverse = owners[child_pointer]
            if (declared.count(child_pointer) != 1 or len(matches) != 1
                    or matches[0]["status"] != "verified_parent_membership"):
                child_reasons.append("parent_attribute_membership_unverified")
            else:
                slots = [slot for slot, raw in enumerate(raw_refs)
                         if strict_handle(raw, reference=True, origin=origin) == child_origin]
                if (matches[0]["field_evidence"]["raw_value"] != raw_refs[slots[0]]
                        or matches[0]["field_evidence"]["source_pointer"] != f"{pointer}/attribs/{slots[0]}"):
                    child_reasons.append("parent_attribute_reference_evidence_disagrees")
            if (len(reverse) != 1 or reverse[0]["status"] != "resolved_reference"
                    or reverse[0]["target_ref"] != parent["source_ref"]):
                child_reasons.append("attribute_reverse_owner_unverified")
            owner = _evidence(child, "ownerhandle")
            if (owner["conflicting"] or strict_handle(owner["raw_value"], reference=True,
                                                     origin=child_origin) != origin):
                child_reasons.append("attribute_raw_owner_disagrees")
            mode = _evidence(child, "entmode")
            if mode["conflicting"] or type(mode["raw_value"]) is not int or mode["raw_value"] != 0:
                child_reasons.append("attribute_entmode_not_parent_owned")
            if not _context_valid(child):
                child_reasons.append("attribute_cad_context_unknown")
            if child_reasons:
                reasons.append("attribute_membership_or_context_unverified")
            attributes.append({"source_ref": deepcopy(child["source_ref"]),
                               "tag_evidence": deepcopy(_evidence(child, "tag")),
                               "text_evidence": deepcopy(_evidence(child, "text_value")),
                               "binding_status": "unknown" if child_reasons else "reciprocal_verified",
                               "binding_reasons": child_reasons,
                               "parent_edge_ids": [edge["id"] for edge in matches],
                               "owner_edge_ids": [edge["id"] for edge in reverse]})
        used_attributes.update(candidates)
        by_tag = defaultdict(list)
        for attr in attributes:
            for tag in {value for value in _values(attr["tag_evidence"]) if isinstance(value, str)}:
                by_tag[tag].append(attr)
            tag = attr["tag_evidence"]
            if tag["conflicting"] or tag["occurrence_count"] != 1 or not isinstance(tag["raw_value"], str):
                reasons.append("attribute_tag_missing_duplicate_conflicting_or_invalid")
        if any(len(attributes_for_tag) != 1 for attributes_for_tag in by_tag.values()):
            reasons.append("duplicate_attribute_tags")
        fields = {}
        for tag in TAGS:
            matches = by_tag.get(tag, [])
            field_reasons = []
            value = None
            if not matches:
                status = "unknown" if reasons else "not_declared"
            elif len(matches) != 1:
                status = "unknown"
                field_reasons.append("duplicate_tag_attributes")
            else:
                attr = matches[0]
                for role in ("tag", "text"):
                    evidence = attr[role + "_evidence"]
                    if (evidence["conflicting"] or evidence["occurrence_count"] != 1
                            or not isinstance(evidence["raw_value"], str)):
                        field_reasons.append(role + "_missing_duplicate_conflicting_or_invalid")
                if attr["binding_status"] != "reciprocal_verified":
                    field_reasons.append("attribute_binding_unverified")
                status = "unknown" if field_reasons else "declared"
                if status == "declared":
                    value = attr["text_evidence"]["raw_value"]
                    status = "declared_empty" if value == "" else "declared"
            fields[tag] = {"status": status, "value": value,
                           "attribute_source_refs": [attr["source_ref"] for attr in matches],
                           "reasons": field_reasons}
        if any(value["status"] == "unknown" for value in fields.values() if value["attribute_source_refs"]):
            reasons.append("row_field_ambiguous_or_invalid")
        is_candidate = CORE_TAGS.issubset(by_tag)
        classification = "row_candidate" if is_candidate else (
            "possible_row_unknown" if reasons else "nonrow_attribute_group")
        record = {"parent_source_ref": deepcopy(parent["source_ref"]),
                  "source_definition_ref": deepcopy(definition_ref),
                  "definition_edge_ids": [edge["id"] for edge in definition_edges],
                  "parent_cad_context": deepcopy(parent["cad_context"]),
                  "attribute_array_evidence": deepcopy(array),
                  "has_attribs_evidence": deepcopy(flag), "num_owned_evidence": deepcopy(count),
                  "classification": classification,
                  "status": "unknown" if reasons else ("source_bound_attribute_row" if is_candidate
                                                          else "source_bound_nonrow_group"),
                  "observed_exact_tags": sorted(by_tag), "missing_core_tags": sorted(CORE_TAGS - by_tag.keys()),
                  "fields": fields, "attributes": attributes,
                  "unresolved_reasons": sorted(set(reasons)), "physical_target": "unknown",
                  "physical_part_confirmed": False, "units": "unknown"}
        (nonrows if classification == "nonrow_attribute_group" else rows).append(record)
    orphans = [{"source_ref": deepcopy(node["source_ref"]),
                "tag_evidence": deepcopy(_evidence(node, "tag")),
                "text_evidence": deepcopy(_evidence(node, "text_value")),
                "reason": "no_resolved_INSERT_group", "physical_target": "unknown"}
               for pointer, node in pointers.items()
               if node["native_kind"] == "ATTRIB" and pointer not in used_attributes]
    result = {"schema_version": SCHEMA, "source_kind": "native_dwg",
              "source_sha256": graph["source_sha256"], "response_sha256": graph["response_sha256"],
              "projection_sha256": graph["projection_sha256"], "ownership_record_sha256": graph["record_sha256"],
              "rows": rows, "nonrow_groups": nonrows, "orphan_attributes": orphans,
              "physical_target": "unknown", "eligible_for_route_training": False,
              "original_byte_hashes_verified_by_consumer": False,
              "summary": {"inserts_considered": insert_count, "attribute_groups": len(rows) + len(nonrows),
                          "row_candidates": sum(row["classification"] == "row_candidate" for row in rows),
                          "source_bound_rows": sum(row["status"] == "source_bound_attribute_row" for row in rows),
                          "possible_rows_unknown": sum(row["classification"] == "possible_row_unknown" for row in rows),
                          "unknown_groups": sum(row["status"] == "unknown" for row in rows),
                          "nonrow_groups": len(nonrows), "orphan_attributes": len(orphans),
                          "row_field_statuses": dict(sorted(Counter(field["status"] for row in rows
                                                                      for field in row["fields"].values()).items()))}}
    result["record_sha256"] = _digest(result)
    return result
