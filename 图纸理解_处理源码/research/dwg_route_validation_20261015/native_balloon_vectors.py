"""Recover same-object native ZWMBALLOON preview vectors, never physical owners.

The caller verifies original byte hashes. Plot-style opcode 26 remains opaque;
custom plugin payloads, circle targets and reactor semantics are not decoded.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import math
import struct
from typing import Any

from ezdxf.tools.binarydata import EndOfBufferError

from research.dwg_route_validation_20261015.native_attribute_rows import (
    _digest, _index, project_attribute_rows,
)
from research.dwg_route_validation_20261015.native_dimension_display import strict_handle
from research.dwg_route_validation_20261015.native_layer_roles import (
    _field_evidence, _validate_lossless_structure,
)
from research.dwg_route_validation_20261015.native_relation_features import _guards
from research.dwg_route_validation_20261015.title_proxy_probe import (
    CODEPAGE_ENCODINGS, decode_hex_preview, decode_preview, validate_frames,
)

SCHEMA = "native_balloon_vectors.v1"
_STYLES = {14: 4, 16: 4, 18: 4, 19: 4, 20: 4, 22: 4, 23: 4, 24: 8, 25: 8, 26: 8}
_FIELDS = ("type", "preview_exists", "preview_size", "preview", "preview_is_proxy",
           "entmode", "invisible", "num_unknown_bits")


def _usable(ev: dict) -> bool:
    return ev["occurrence_count"] > 0 and not ev["conflicting"]


def _values(ev: dict) -> list[Any]:
    return [item["raw_value"] for item in ev["occurrences"]]


def _finite(values: tuple) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite_native_coordinates_or_scalars")


def _string_end(body: bytes, start: int, encoding: str) -> int:
    end = body.find(b"\0", start)
    if end < 0:
        raise ValueError("truncated_TEXT2_string")
    body[start:end].decode(encoding, errors="strict")
    return (end + 4) // 4 * 4


def _check_payload(opcode: int, body: bytes, encoding: str) -> None:
    """Close supported frame extents before the permissive library decoder."""
    if opcode in _STYLES:
        if len(body) != _STYLES[opcode]:
            raise ValueError("invalid_style_frame_extent")
        if opcode in (24, 25):
            _finite(struct.unpack("<d", body))
    elif opcode == 6:
        count = struct.unpack_from("<I", body)[0]
        if count < 2 or len(body) != 4 + count * 24:
            raise ValueError("invalid_POLYLINE_vertex_extent")
        _finite(struct.unpack_from(f"<{count * 3}d", body, 4))
    elif opcode == 2:
        if len(body) != 56:
            raise ValueError("invalid_CIRCLE_extent")
        values = struct.unpack("<7d", body)
        _finite(values)
        if values[3] <= 0 or values[4:] != (0., 0., 1.):
            raise ValueError("unsupported_circle_radius_or_normal")
    elif opcode == 11:
        values = struct.unpack_from("<9d", body)
        _finite(values)
        if (values[3:6] != (0., 0., 1.) or values[6:8] == (0., 0.) or values[8] != 0):
            raise ValueError("unsupported_text_normal_or_direction")
        offset = _string_end(body, 72, encoding)
        _finite(struct.unpack_from("<4d", body, offset + 8))
        offset = _string_end(body, offset + 8 + 32 + 20, encoding)
        if _string_end(body, offset, encoding) != len(body):
            raise ValueError("unaccounted_TEXT2_payload")
    else:
        raise ValueError("matrix_or_unknown_geometry_command")


def _decode(row: dict, encoding: str | None, version: Any) -> None:
    fields, reasons = row["field_evidence"], row["reasons"]
    if any(ev["conflicting"] for ev in fields.values()):
        reasons.append("conflicting_native_field")
    if (not _usable(fields["preview_exists"]) or type(fields["preview_exists"]["raw_value"]) is not int
            or fields["preview_exists"]["raw_value"] != 1):
        reasons.append("preview_not_declared_present")
    size = fields["preview_size"]["raw_value"]
    if not _usable(fields["preview_size"]) or type(size) is not int or size < 0:
        reasons.append("invalid_preview_size")
    if encoding is None or not isinstance(version, str):
        reasons.append("unsupported_or_conflicting_FILEHEADER")
    if reasons:
        return
    try:
        data = decode_hex_preview(fields["preview"]["raw_value"], size)
        row["preview_sha256"] = hashlib.sha256(data).hexdigest()
        frames = validate_frames(data)
        # Frame evidence survives unknown opcodes and malformed individual payloads.
        row["frames"] = [{"index": f.index, "offset": f.offset, "size": f.size,
                          "opcode": f.opcode, "name": f.name,
                          "sha256": hashlib.sha256(data[f.offset:f.offset + f.size]).hexdigest(),
                          "raw_hex": data[f.offset:f.offset + f.size].hex().upper(),
                          "payload_authentication": "not_validated"} for f in frames]
        for frame in frames:
            _check_payload(frame.opcode, data[frame.offset + 8:frame.offset + frame.size], encoding)
            row["frames"][frame.index]["payload_authentication"] = (
                "unverified_style_only" if frame.opcode == 26 else "supported_frame_layout")
        decoded = decode_preview(data, encoding=encoding, version=version)
        if any(not text["library_text_identical_to_raw"] for text in decoded["texts"]):
            raise ValueError("library_rewrote_native_text")
        row["geometry"] = [entry for entry in decoded["entities"] if entry["type"] != "TEXT"]
        row["texts"] = decoded["texts"]
        row["style_payloads_unverified"] = [f.index for f in frames if f.opcode == 26]
        row["coordinate_status"] = "same_object_native_preview_coordinates_only"
        row["status"] = "native_vector_evidence"
    except (ValueError, TypeError, IndexError, OverflowError, struct.error, EndOfBufferError) as error:
        reasons.append(f"preview_unknown:{type(error).__name__}:{error}")


def _incoming(parsed: dict, nodes: list[dict], balloons: list[dict]) -> list[dict]:
    selected = {row["source_ref"]["native_handle"]: row for row in balloons
                if row["source_ref"]["native_handle"] is not None}
    indexed = {node["source_ref"]["native_handle"]: node for node in nodes
               if node["source_ref"]["native_handle"] is not None}
    unresolved = []
    for obj, node in zip(parsed["OBJECTS"], nodes, strict=True):
        source = node["source_ref"]
        evidence = _field_evidence(obj, "reactors", source["json_pointer"])
        count = _field_evidence(obj, "num_reactors", source["json_pointer"])
        origin = strict_handle(source["original_handle"], reference=False)
        if not evidence["occurrence_count"] and (count["conflicting"] or
                (count["occurrence_count"] and (type(count["raw_value"]) is not int or count["raw_value"] != 0))):
            unresolved.append({"source_ref": deepcopy(source), "field_evidence": evidence,
                               "count_evidence": count, "reason": "missing_reactor_array"})
        for occurrence in evidence["occurrences"]:
            refs = occurrence["raw_value"]
            if not isinstance(refs, list):
                unresolved.append({"source_ref": deepcopy(source), "field_evidence": evidence,
                                   "reason": "invalid_reactor_array"})
                continue
            inconsistent = evidence["conflicting"] or count["conflicting"] or (count["occurrence_count"]
                and (type(count["raw_value"]) is not int or count["raw_value"] != len(refs)))
            if inconsistent:
                unresolved.append({"source_ref": deepcopy(source), "field_evidence": deepcopy(evidence),
                    "count_evidence": deepcopy(count), "occurrence_ordinal": occurrence["occurrence_ordinal"],
                    "reason": "conflicting_or_inconsistent_reactor_array"})
            handles = [strict_handle(value, reference=True, origin=origin) for value in refs]
            for slot, (raw, handle) in enumerate(zip(refs, handles, strict=True)):
                key = f"{handle:X}" if handle is not None else None
                if key not in indexed:
                    unresolved.append({"source_ref": deepcopy(source), "field_evidence": deepcopy(evidence),
                        "count_evidence": deepcopy(count), "slot": slot,
                        "occurrence_ordinal": occurrence["occurrence_ordinal"],
                        "original_reference": deepcopy(raw), "reason": "invalid_or_unresolved_reactor_reference"})
                if key not in selected:
                    continue
                target = selected[key]
                reasons = []
                if inconsistent:
                    reasons.append("conflicting_or_inconsistent_reactor_array")
                if handles.count(handle) != 1 or source["native_handle"] not in indexed:
                    reasons.append("duplicate_reference_or_invalid_source_identity")
                edge = {"source_ref": deepcopy(source), "target_ref": deepcopy(target["source_ref"]),
                        "field_evidence": deepcopy(evidence), "count_evidence": deepcopy(count),
                        "slot": slot, "occurrence_ordinal": occurrence["occurrence_ordinal"],
                        "source_pointer": f"{source['json_pointer']}/reactors/{slot}",
                        "original_reference": deepcopy(raw), "status": "unknown" if reasons else
                        "declared_native_reactor_reference", "reasons": reasons,
                        "semantics": "unknown_dependency_not_physical_ownership"}
                target["incoming_reactors"].append(edge)
    return unresolved


def build_balloon_vectors(parsed: dict[str, Any], ownership_graph: dict[str, Any],
                          attribute_rows: dict[str, Any]) -> dict[str, Any]:
    """Recover only CLASSES-linked ZWMBALLOON previews and exact row candidates."""
    _guards(parsed)
    _validate_lossless_structure(parsed)
    _index(ownership_graph)  # Self digest, all original handles and source pointers.
    if _digest(parsed) != ownership_graph["projection_sha256"]:
        raise ValueError("Ownership graph does not match the supplied native projection")
    for field in ("source_sha256", "response_sha256"):
        ev = _field_evidence(parsed, field, "#")
        if ev["conflicting"] or (ev["occurrence_count"] and ev["raw_value"] != ownership_graph[field]):
            raise ValueError("Native source metadata disagrees")
    if attribute_rows != project_attribute_rows(ownership_graph):
        raise ValueError("Attribute rows do not match the unchanged source ownership graph")
    classes, header = parsed.get("CLASSES"), parsed.get("FILEHEADER", {})
    if not isinstance(classes, list) or any(not isinstance(row, dict) for row in classes):
        raise ValueError("An original native CLASSES array is required")
    if not isinstance(header, dict):
        header = {}
    codepage = _field_evidence(header, "codepage", "#/FILEHEADER")
    version = _field_evidence(header, "version", "#/FILEHEADER")
    encoding = CODEPAGE_ENCODINGS.get(codepage["raw_value"]) if (
        _usable(codepage) and type(codepage["raw_value"]) is int) else None
    version_value = version["raw_value"] if _usable(version) else None
    class_index = defaultdict(list)
    for position, cls in enumerate(classes):
        number = _field_evidence(cls, "number", f"#/CLASSES/{position}")
        for value in _values(number):
            if type(value) is int and position not in [entry[0] for entry in class_index[value]]:
                class_index[value].append((position, cls, number))
    balloons = []
    for obj, node in zip(parsed["OBJECTS"], ownership_graph["objects"], strict=True):
        pointer = node["source_ref"]["json_pointer"]
        fields = {key: _field_evidence(obj, key, pointer) for key in _FIELDS}
        candidates = [entry for value in _values(fields["type"]) if type(value) is int
                      for entry in class_index[value]]
        if not any("ZWMBALLOON" in _values(_field_evidence(cls, "dxfname", f"#/CLASSES/{pos}"))
                   for pos, cls, _ in candidates):
            continue
        reasons, class_refs = [], []
        for pos, cls, number in candidates:
            name = _field_evidence(cls, "dxfname", f"#/CLASSES/{pos}")
            class_refs.append({"json_pointer": f"#/CLASSES/{pos}", "number_evidence": number,
                               "dxfname_evidence": name})
            if number["conflicting"] or name["conflicting"] or name["raw_value"] != "ZWMBALLOON":
                reasons.append("conflicting_or_wrong_native_class")
        if len(candidates) != 1 or not _usable(fields["type"]):
            reasons.append("ambiguous_native_class_reference")
        if node["identity_status"] != "valid" or node["cad_context"]["reasons"]:
            reasons.append("unresolved_native_identity_or_CAD_context")
        row = {"source_ref": deepcopy(node["source_ref"]), "class_refs": class_refs,
               "cad_context": deepcopy(node["cad_context"]), "field_evidence": fields,
               "status": "unknown", "reasons": reasons, "frames": [], "geometry": [], "texts": [],
               "preview_sha256": None, "coordinate_status": "unknown", "incoming_reactors": [],
               "physical_owner": "unknown", "physical_part_confirmed": False,
               "unknown_bits_decoded": False, "style_payloads_unverified": []}
        _decode(row, encoding, version_value)
        balloons.append(row)
    unresolved = _incoming(parsed, ownership_graph["objects"], balloons)
    labels = Counter(text["raw_text"] for row in balloons for text in row["texts"])
    for row in balloons:
        row["label_row_candidates"] = []
        for text in row["texts"]:
            label = text["raw_text"]
            matches = [item for item in attribute_rows["rows"]
                       if label != "" and item["fields"]["序号"]["status"] == "declared"
                       and item["fields"]["序号"]["value"] == label]
            row["label_row_candidates"].append({"raw_label": label, "text_frame_index": text["frame_index"],
                "matching_rows": [{"parent_source_ref": deepcopy(item["parent_source_ref"]),
                                   "sequence_field": deepcopy(item["fields"]["序号"]),
                                   "row_status": item["status"]} for item in matches],
                "status": "exact_string_candidate_only" if len(matches) == 1 and labels[label] == 1
                and matches[0]["status"] == "source_bound_attribute_row" else "unknown",
                "physical_owner": "unknown", "binding_confirmed": False})
    result = {"schema_version": SCHEMA, "source_kind": "native_dwg",
              **{key: ownership_graph[key] for key in ("source_sha256", "response_sha256", "projection_sha256")},
              "ownership_record_sha256": ownership_graph["record_sha256"],
              "attribute_rows_record_sha256": attribute_rows["record_sha256"],
              "header_evidence": {"codepage": codepage, "version": version}, "balloons": balloons,
              "unresolved_reactor_arrays": unresolved, "physical_owner": "unknown", "units": "unknown",
              "eligible_for_route_training": False, "original_byte_hashes_verified_by_consumer": False,
              "summary": {"selected_objects": len(balloons), "statuses": dict(Counter(row["status"] for row in balloons)),
                          "incoming_reactor_references": sum(len(row["incoming_reactors"]) for row in balloons)}}
    result["record_sha256"] = _digest(result)
    return result
