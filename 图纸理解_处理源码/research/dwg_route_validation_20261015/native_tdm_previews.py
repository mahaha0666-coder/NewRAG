"""Recover same-object TDM display bytes and vectors, not manufacturing facts.

The caller verifies DWG/response bytes. Coordinates are native preview evidence,
never certified WCS, physical ownership, BOM fields or route-training input.
Literal transcription survives unknown coordinate commands and normals.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
import struct
from typing import Any

from ezdxf.tools.binarydata import EndOfBufferError

from research.dwg_route_validation_20261015.native_attribute_rows import _digest
from research.dwg_route_validation_20261015.native_balloon_coverage import (
    _classes, _ref, _scope, _status, _values,
)
from research.dwg_route_validation_20261015.native_balloon_vectors import _check_payload
from research.dwg_route_validation_20261015.native_dimension_display import strict_handle
from research.dwg_route_validation_20261015.native_layer_roles import (
    _field_evidence, _validate_lossless_structure,
)
from research.dwg_route_validation_20261015.native_relation_features import _guards
from research.dwg_route_validation_20261015.title_proxy_probe import (
    CODEPAGE_ENCODINGS, decode_hex_preview, decode_preview, validate_frames,
)

SCHEMA = "native_tdm_previews.v1"
_TARGETS = {"TDMLABEL", "TDMBOMITEM"}
_FIELDS = ("handle", "type", "entity", "object", "preview", "preview_exists",
           "preview_size", "preview_is_proxy")
_ERRORS = (ValueError, TypeError, IndexError, OverflowError, struct.error, EndOfBufferError)


def _safe(values: tuple) -> list:
    """Nonfinite numbers retain exact IEEE bytes in the parent frame, not JSON NaN."""
    return [value if math.isfinite(value) else None for value in values]


def _literal(body: bytes, encoding: str, frame: Any) -> dict:
    if len(body) < 73:
        raise ValueError("truncated_TEXT2_fixed_fields")
    end = body.find(b"\0", 72)
    if end < 0:
        raise ValueError("truncated_TEXT2_string")
    values = struct.unpack_from("<9d", body)
    return {"frame_index": frame.index, "raw_text": body[72:end].decode(encoding, "strict"),
            "raw_text_hex": body[72:end].hex().upper(), "strict_encoding": encoding,
            "text_byte_start": frame.offset + 80, "text_byte_end": frame.offset + 8 + end,
            "terminator_byte_offset": frame.offset + 8 + end,
            "native_start_point": _safe(values[:3]), "native_normal": _safe(values[3:6]),
            "native_direction": _safe(values[6:]), "literal_status": "strict_source_bytes",
            "library_check": "not_run", "coordinate_status": "unknown"}


def _geometry(opcode: int, body: bytes, frame_index: int) -> dict:
    result = {"frame_index": frame_index, "coordinate_status": "unknown",
              "library_check": "not_run"}
    if opcode in (6, 32):
        count = struct.unpack_from("<I", body)[0]
        if count < 2 or len(body) != 4 + 24 * (count + (opcode == 32)):
            raise ValueError("invalid_POLYLINE_vertex_or_normal_extent")
        values = struct.unpack_from(f"<{(count + (opcode == 32)) * 3}d", body, 4)
        result.update(kind="POLYLINE_WITH_NORMALS" if opcode == 32 else "POLYLINE",
                      vertex_count=count,
                      native_vertices=[_safe(values[index:index + 3]) for index in range(0, count * 3, 3)],
                      native_normal=_safe(values[-3:]) if opcode == 32 else None,
                      normal_status="explicit_native" if opcode == 32 else "not_declared_in_frame")
    else:
        if len(body) != 56:
            raise ValueError("invalid_CIRCLE_extent")
        values = struct.unpack("<7d", body)
        result.update(kind="CIRCLE", native_center=_safe(values[:3]), native_radius=_safe(values[3:4])[0],
                      native_normal=_safe(values[4:]), normal_status="explicit_native")
    return result


def _validate_payload(opcode: int, body: bytes, encoding: str) -> None:
    if opcode != 32:
        _check_payload(opcode, body, encoding)
        return
    # The installed reader's _load_vertices(load_normal=True) reads count vertices
    # followed by ONE normal. Its _polyline ignores that normal; we must not.
    count = struct.unpack_from("<I", body)[0]
    if count < 2 or len(body) != 4 + 24 * (count + 1):
        raise ValueError("invalid_POLYLINE_WITH_NORMALS_extent")
    values = struct.unpack_from(f"<{(count + 1) * 3}d", body, 4)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite_POLYLINE_WITH_NORMALS")
    if values[-3:] != (0., 0., 1.):
        raise ValueError("unsupported_POLYLINE_WITH_NORMALS_normal")


def _library_check(raw_frame: bytes, encoding: str, version: str,
                   geometry: dict | None, literal: dict | None) -> None:
    # Decode a single frame, not a stream whose matrices or state are unverified.
    data = struct.pack("<II", len(raw_frame) + 8, 1) + raw_frame
    decoded = decode_preview(data, encoding=encoding, version=version)
    if literal is not None:
        texts = decoded["texts"]
        if (len(texts) != 1 or not texts[0]["library_text_identical_to_raw"]
                or texts[0]["raw_text"] != literal["raw_text"]
                or texts[0]["raw_start_point"] != literal["native_start_point"]
                or texts[0]["raw_normal"] != literal["native_normal"]
                or texts[0]["raw_text_direction"] != literal["native_direction"]):
            raise ValueError("independent_TEXT2_library_mismatch")
        literal["library_check"] = "exact_literal_and_native_fields"
    elif geometry is not None:
        entities = decoded["entities"]
        if len(entities) != 1:
            raise ValueError("independent_geometry_library_count_mismatch")
        entity = entities[0]
        if geometry["kind"].startswith("POLYLINE"):
            vertices = [row["location"] for row in entity.get("vertices", [])]
            if entity["type"] == "POINT" and len(set(map(tuple, geometry["native_vertices"]))) == 1:
                vertices = [entity["attributes"]["location"]] * geometry["vertex_count"]
            if vertices != geometry["native_vertices"]:
                raise ValueError("independent_POLYLINE_library_mismatch")
            geometry["library_check"] = ("vertices_exact_normal_not_used_by_library"
                if geometry["kind"] == "POLYLINE_WITH_NORMALS" else "vertices_exact")
        else:
            attrs = entity["attributes"]
            if (attrs.get("center") != geometry["native_center"]
                    or attrs.get("radius") != geometry["native_radius"]
                    or list(attrs.get("extrusion", [0., 0., 1.])) != geometry["native_normal"]):
                raise ValueError("independent_CIRCLE_library_mismatch")
            geometry["library_check"] = "native_circle_exact"


def _preview(value: Any, size: Any, encoding: str | None, version: Any) -> dict:
    result = {"preview_sha256": None, "status": "unknown", "reasons": [], "frames": [],
              "texts": [], "geometry": [], "coordinate_status": "unknown"}
    try:
        if type(size) is not int or size < 0:
            raise ValueError("invalid_or_conflicting_preview_size")
        data = decode_hex_preview(value, size)
        result["preview_sha256"] = hashlib.sha256(data).hexdigest()
        frames = validate_frames(data)
        result["header_raw_hex"] = data[:8].hex().upper()
        if not frames:
            raise ValueError("empty_native_preview_stream")
        for frame in frames:
            raw = data[frame.offset:frame.offset + frame.size]
            body = raw[8:]
            entry = {"index": frame.index, "offset": frame.offset, "size": frame.size,
                     "opcode": frame.opcode, "name": frame.name, "raw_hex": raw.hex().upper(),
                     "sha256": hashlib.sha256(raw).hexdigest(), "payload_authentication": "not_validated",
                     "reasons": []}
            result["frames"].append(entry)
            geometry = literal = None
            try:
                if frame.opcode == 11 and encoding is not None:
                    literal = _literal(body, encoding, frame)
                    result["texts"].append(literal)
                elif frame.opcode in (6, 32, 2):
                    geometry = _geometry(frame.opcode, body, frame.index)
                    result["geometry"].append(geometry)
                if encoding is None or version != "AC1018":
                    raise ValueError("unsupported_or_conflicting_FILEHEADER")
                _validate_payload(frame.opcode, body, encoding)
                entry["payload_authentication"] = ("unverified_style_only" if frame.opcode == 26
                                                       else "independently_checked_frame_layout")
                if geometry is not None or literal is not None:
                    _library_check(raw, encoding, version, geometry, literal)
            except _ERRORS as error:
                entry["reasons"].append(f"{type(error).__name__}:{error}")
        result["reasons"] = [f"frame_{row['index']}:{reason}" for row in result["frames"]
                             for reason in row["reasons"]]
        if not result["reasons"]:
            result["status"] = "native_preview_evidence"
            result["coordinate_status"] = "same_object_native_preview_only_not_WCS"
            for item in result["texts"] + result["geometry"]:
                item["coordinate_status"] = result["coordinate_status"]
    except _ERRORS as error:
        result["reasons"].append(f"{type(error).__name__}:{error}")
    return result


def build_tdm_previews(parsed: dict[str, Any], *, source_sha256: str,
                       response_sha256: str) -> dict[str, Any]:
    """Project only actual CLASSES/type-linked TDMLABEL and TDMBOMITEM objects."""
    if not isinstance(parsed, dict):
        raise ValueError("An original lossless native projection is required")
    _guards(parsed)
    _validate_lossless_structure(parsed)
    for field, value in (("source_sha256", source_sha256), ("response_sha256", response_sha256)):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Original source and response SHA256 are required")
        ev = _field_evidence(parsed, field, "#")
        if ev["conflicting"] or (ev["occurrence_count"] and ev["raw_value"] != value):
            raise ValueError("Native source metadata disagrees")
    objects_ev = _field_evidence(parsed, "OBJECTS", "#")
    objects = objects_ev["raw_value"]
    if objects_ev["conflicting"] or not isinstance(objects, list) or any(not isinstance(o, dict) for o in objects):
        raise ValueError("Unambiguous native OBJECTS are required")
    classes, _, class_unknown = _classes(parsed)
    header_ev = _field_evidence(parsed, "FILEHEADER", "#")
    header = header_ev["raw_value"] if isinstance(header_ev["raw_value"], dict) else {}
    cp, version = (_field_evidence(header, key, "#/FILEHEADER") for key in ("codepage", "version"))
    encoding = CODEPAGE_ENCODINGS.get(cp["raw_value"]) if _status(cp, int) in {"declared", "identical_repeated"} else None
    native_version = version["raw_value"] if _status(version) in {"declared", "identical_repeated"} else None
    source = {"source_sha256": source_sha256, "raw_response_sha256": response_sha256}
    refs = [_ref(i, obj, source) for i, obj in enumerate(objects)]
    # A conflicting object's possible handles also make another object ambiguous.
    handles = Counter(handle for i, obj in enumerate(objects) for handle in {
        strict_handle(value, reference=False)
        for value in _values(_field_evidence(obj, "handle", f"#/OBJECTS/{i}"))}
        if handle is not None)
    rows = []
    for position, obj in enumerate(objects):
        pointer = f"#/OBJECTS/{position}"
        fields = {key: _field_evidence(obj, key, pointer) for key in _FIELDS}
        candidates = {candidate["json_pointer"]: candidate for value in _values(fields["type"])
                      if type(value) is int for candidate in classes.get(value, [])}
        if not any(type(name) is str and name in _TARGETS
                   for c in candidates.values() for name in _values(c["dxfname_evidence"])):
            continue
        reasons = []
        names = [candidate["dxfname_evidence"]["raw_value"] for candidate in candidates.values()]
        if (len(candidates) != 1 or _status(fields["type"], int) not in {"declared", "identical_repeated"}
                or any(_status(c[key], kind) not in {"declared", "identical_repeated"}
                       for c in candidates.values() for key, kind in (("number_evidence", int), ("dxfname_evidence", str)))):
            reasons.append("ambiguous_or_conflicting_actual_class")
        declared_class = names[0] if not reasons else None
        ref = refs[position]
        handle = strict_handle(fields["handle"]["raw_value"], reference=False)
        if ref["native_handle"] is None or handles[handle] != 1:
            reasons.append("invalid_conflicting_or_duplicate_object_identity")
        if any(ev["conflicting"] for ev in fields.values()):
            reasons.append("conflicting_native_field")
        if _status(fields["preview_exists"], int) not in {"declared", "identical_repeated"} or fields["preview_exists"]["raw_value"] != 1:
            reasons.append("preview_not_declared_present")
        previews = []
        for occurrence in fields["preview"]["occurrences"]:
            preview = _preview(occurrence["raw_value"], fields["preview_size"]["raw_value"], encoding, native_version)
            preview.update(occurrence_ordinal=occurrence["occurrence_ordinal"], source_pointer=occurrence["source_pointer"],
                           sidecar_pointer=occurrence["sidecar_pointer"],
                           source_binding_status="unknown" if reasons else "actual_type_class_join")
            if reasons or header_ev["conflicting"]:
                preview["coordinate_status"] = "unknown"
                for item in preview["texts"] + preview["geometry"]:
                    item["coordinate_status"] = "unknown"
            previews.append(preview)
        rows.append({"source_ref": ref, "class_refs": list(candidates.values()),
                     "declared_class": declared_class, "field_evidence": fields,
                     "raw_scope": _scope(obj, pointer), "reasons": reasons,
                     "status": "native_preview_evidence" if not reasons and previews and all(
                         p["status"] == "native_preview_evidence" for p in previews) else "unknown",
                     "preview_candidates": previews, "physical_owner": "unknown",
                     "eligible_for_route_training": False})
    previews = [p for row in rows for p in row["preview_candidates"]]
    summary = {"selected_objects": len(rows), "class_counts": dict(Counter(row["declared_class"] or "unknown" for row in rows)),
               "object_status_counts": dict(Counter(row["status"] for row in rows)),
               "preview_candidates": len(previews), "strict_literal_strings": sum(len(p["texts"]) for p in previews),
               "coordinate_status_counts": dict(Counter(p["coordinate_status"] for p in previews)),
               "frame_opcode_counts": dict(Counter(str(f["opcode"]) for p in previews for f in p["frames"])),
               "geometry_counts": dict(Counter(g["kind"] for p in previews for g in p["geometry"])),
               "literal_library_check_counts": dict(Counter(t["library_check"] for p in previews for t in p["texts"])),
               "unverified_style_frames": sum(f["payload_authentication"] == "unverified_style_only" for p in previews for f in p["frames"])}
    result = {"schema": SCHEMA, "source_sha256": source_sha256, "response_sha256": response_sha256,
              "projection_sha256": _digest(parsed), "fileheader_evidence": header_ev,
              "unresolved_class_declarations": class_unknown, "objects": rows, "summary": summary,
              "CAD_context_validated": False, "WCS_validated": False, "physical_ownership_validated": False,
              "unknown_plugin_payload_decoded": False, "eligible_for_route_training": False}
    result["record_sha256"] = _digest(result)
    return result
