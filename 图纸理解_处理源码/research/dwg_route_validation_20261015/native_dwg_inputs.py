"""Pure native-DWG inputs; no PDF pairing, image cache, encoder or route model.

Callers verify original DWG and response byte hashes before invoking this module.
The supplied lossless projection is rebuilt against the old IR without mutation.
Raw measurements are CAD values, never reconstructed display strings or mm.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Literal

from research.dwg_route_validation_20261015.native_layer_roles import (
    _field_evidence, project_layers,
)
from research.dwg_route_validation_20261015.native_relation_features import _guards
from research.dwg_route_validation_20261015.parser_probe import (
    TEXT_FIELDS, TEXT_TYPES, handle_value, hex_handle, object_kind,
)

SCHEMA = "native_dwg_inputs.v1"
FEATURE_SCHEMA = "native_dwg_input_features.v1"
Mode = Literal["C0", "C1"]
_BODY_FIELDS = {"TEXT": "text_value", "MTEXT": "text"}
_TEXT_COORDINATES = ("ins_pt", "insertion_pt", "insert", "extrusion", "elevation",
                     "rotation", "text_direction", "x_axis_dir", "normal", "height")
_PATH_START = r"(?:[A-Za-z][A-Za-z0-9+.-]*://|[A-Za-z]:[\\/]|/|\\\\)"
_QUOTED_PATHS = re.compile("|".join(
    re.escape(left) + _PATH_START + "[^" + re.escape(right) + r"\r\n]*" + re.escape(right)
    for left, right in (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’"))
))
_RESOURCE_PATHS = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9+.-]*://|[A-Za-z]:[\\/]|\\\\)[^\s<>\"'“”‘’（）()\[\]{}，；]+"
    r"|(?<![\w.])/(?:[^\s/<>\"'“”‘’（）()\[\]{}，；]+/)*[^\s/<>\"'“”‘’（）()\[\]{}，；]+"
)
_IDENTIFIERS = re.compile(
    r"(?<!\d)\d{10,14}(?!\d)|[A-Za-z][A-Za-z0-9.]*[#＃][A-Za-z0-9.-]+"
    r"|(?<![A-Fa-f0-9])[A-Fa-f0-9]{64}(?![A-Fa-f0-9])"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _source_ref(position: int, obj: dict, source: str, response: str) -> dict:
    return {"source_sha256": source, "response_sha256": response,
            "json_pointer": f"#/OBJECTS/{position}",
            "native_handle": hex_handle(handle_value(obj.get("handle")))}


def _visibility(obj: dict, position: int, index: dict[int, tuple[int, dict]]) -> dict:
    """Source-declared hidden/off/frozen state, not final viewport visibility."""
    pointer = f"#/OBJECTS/{position}"
    invisible = _field_evidence(obj, "invisible", pointer)
    layer_ref = _field_evidence(obj, "layer", pointer)
    target = index.get(handle_value(layer_ref["raw_value"]))
    flag, name = None, None
    layer_source = None
    if target is not None and object_kind(target[1]) == "LAYER":
        layer_position, layer = target
        layer_source = {"json_pointer": f"#/OBJECTS/{layer_position}",
                        "native_handle": hex_handle(handle_value(layer.get("handle")))}
        flag = _field_evidence(layer, "flag0", layer_source["json_pointer"])
        name = _field_evidence(layer, "name", layer_source["json_pointer"])
    reasons = []
    if invisible["conflicting"] or type(invisible["raw_value"]) is not int:
        reasons.append("entity_visibility_unknown_or_conflicting")
    elif invisible["raw_value"] == 1:
        reasons.append("entity_declared_invisible")
    elif invisible["raw_value"] != 0:
        reasons.append("entity_visibility_value_unsupported")
    if layer_ref["conflicting"] or flag is None:
        reasons.append("layer_visibility_unresolved_or_conflicting")
    elif flag["conflicting"] or type(flag["raw_value"]) is not int or flag["raw_value"] < 0:
        reasons.append("layer_visibility_unknown_or_conflicting")
    elif flag["raw_value"] & 3:  # Original LAYER flag0: frozen=1, off=2.
        reasons.append("layer_declared_frozen_or_off")
    return {"source_declared_not_hidden": not reasons, "reasons": reasons,
            "entity_invisible_evidence": invisible, "layer_reference_evidence": layer_ref,
            "layer_source_ref": layer_source, "layer_flag0_evidence": flag,
            "layer_name_evidence": name,
            "final_viewport_visibility": "unknown"}


def _measurement_status(evidence: dict) -> str:
    value = evidence["raw_value"]
    if evidence["conflicting"]:
        return "conflicting"
    if evidence["occurrence_count"] == 0:
        return "missing"
    if type(value) not in (int, float):
        return "invalid_type"
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    return "finite_native_raw" if finite else "invalid_nonfinite_or_unrepresentable"


def build_native_input(parsed: dict[str, Any], ir: dict[str, Any], *,
                       source_sha256: str, response_sha256: str) -> dict[str, Any]:
    """Build one evidence-bound native input; byte verification belongs to caller.

    Every old IR dimension is retained, including missing/invalid values. Only
    explicitly nonhidden modelspace TEXT/MTEXT fields populate the native body.
    Definitions, attributes, prompts, paperspace and unresolved visibility stay
    separate. This is a consumer diagnostic and does not authorize route fitting.
    """
    _guards(parsed)
    _guards(ir)
    if (ir.get("source_sha256") != source_sha256
            or ir.get("response_sha256") != response_sha256):
        raise ValueError("Explicit original DWG/response hashes differ from the native IR")
    source = deepcopy(parsed)
    old_ir = deepcopy(ir)
    projection_sha = _digest(source)
    layers = project_layers(source, old_ir)
    objects = source["OBJECTS"]
    index = {handle_value(obj.get("handle")): (position, obj)
             for position, obj in enumerate(objects) if handle_value(obj.get("handle")) is not None}
    accepted, deferred = [], []
    for position, obj in enumerate(objects):
        kind = object_kind(obj)
        if kind not in TEXT_TYPES:
            continue
        ref = _source_ref(position, obj, source_sha256, response_sha256)
        visibility = _visibility(obj, position, index)
        for field in TEXT_FIELDS:
            evidence = _field_evidence(obj, field, ref["json_pointer"])
            if not evidence["occurrence_count"]:
                continue
            value = evidence["raw_value"]
            reasons = []
            if kind not in _BODY_FIELDS:
                reasons.append("attribute_or_definition_requires_instance_scope")
            elif field != _BODY_FIELDS[kind]:
                reasons.append("metadata_or_nonbody_text_field")
            if type(obj.get("entmode")) is not int or obj["entmode"] != 2:
                reasons.append("not_explicit_modelspace_text")
            if ref["native_handle"] is None:
                reasons.append("missing_native_text_handle")
            if evidence["conflicting"]:
                reasons.append("conflicting_text_field")
            elif not isinstance(value, str) or not value:
                reasons.append("empty_or_nonstring_text")
            reasons.extend(visibility["reasons"])
            row = {"native_kind": kind, "native_entmode": obj.get("entmode"),
                   "field": field, "raw_value": deepcopy(value), "source_ref": ref,
                   "field_evidence": evidence, "visibility": visibility,
                   "raw_coordinates": {key: deepcopy(obj[key]) for key in _TEXT_COORDINATES if key in obj},
                   "reasons": reasons, "physical_owner": "unknown", "confidence": None}
            (deferred if reasons else accepted).append(row)
    dimensions = []
    for node in old_ir["dimensions"]:
        position, obj = index[int(node["id"], 16)]
        pointer = f"#/OBJECTS/{position}"
        measurement = _field_evidence(obj, "act_measurement", pointer)
        user_text = _field_evidence(obj, "user_text", pointer)
        visibility = _visibility(obj, position, index)
        dimensions.append({**deepcopy(node), "raw_user_text": user_text["raw_value"],
                           "raw_user_text_evidence": user_text,
                           "raw_measurement_evidence": measurement,
                           "raw_measurement_status": _measurement_status(measurement),
                           "visibility": visibility, "displayed_text": "unknown",
                           "unit": "unknown", "physical_owner": "unknown"})
    record = {"schema_version": SCHEMA, "source_kind": "native_dwg",
              "item_code": old_ir["item_code"], "source_sha256": source_sha256,
              "response_sha256": response_sha256, "projection_sha256": projection_sha,
              "native_ir_sha256": _digest(old_ir),
              "body_text": "\n".join(row["raw_value"] for row in accepted),
              "accepted_text": accepted, "deferred_text": deferred, "dimensions": dimensions,
              "geometry_nodes": deepcopy(old_ir["geometry_nodes"]),
              "relations": deepcopy(old_ir["relations"]), "declared_layers": layers,
              "units": "unknown", "scope": "unknown", "physical_owner": "unknown",
              "confidence": None, "eligible_for_route_training": False,
              "original_byte_hashes_verified_by_consumer": False,
              "policy": {"body_source": "explicit_modelspace_TEXT_or_MTEXT_source_not_hidden",
                         "raw_measurement_is_not_displayed_dimension": True,
                         "no_unit_conversion_or_geometric_manufacturing_inference": True,
                         "final_viewport_visibility": "unknown"}}
    record["record_sha256"] = _digest(record)
    return record


def _validate_record(record: dict[str, Any], mode: Mode) -> None:
    if mode not in ("C0", "C1"):
        raise ValueError("Native consumer mode must be C0 or C1")
    if record.get("schema_version") != SCHEMA or record.get("source_kind") != "native_dwg":
        raise ValueError("A standalone native DWG input record is required")
    _guards(record)
    content = {key: value for key, value in record.items() if key != "record_sha256"}
    if record.get("record_sha256") != _digest(content):
        raise ValueError("Native input content no longer matches its frozen record digest")


def _sanitize_text(value: str, record: dict[str, Any]) -> str:
    # Evidence remains exact. Identifiers, source hashes and paths do not become
    # predictive text; handles and JSON pointers are never read as feature data.
    value = _RESOURCE_PATHS.sub(" ", _QUOTED_PATHS.sub(" ", value))
    tokens = {record.get(key, "") for key in ("item_code", "source_sha256", "response_sha256",
                                             "projection_sha256", "native_ir_sha256", "record_sha256")}
    for token in sorted(tokens, key=len, reverse=True):
        if token:
            value = value.replace(token, " ")
    value = _IDENTIFIERS.sub(" ", value)
    return "\n".join(" ".join(line.split()) for line in value.splitlines())


def sanitized_predictive_text(record: dict[str, Any]) -> str:
    """Native body for a real text encoder; exact original stays in body_text."""
    _validate_record(record, "C0")
    return _sanitize_text(record["body_text"], record)


def _text_features(value: str, prefix: str, record: dict[str, Any]) -> dict[str, float]:
    value = _sanitize_text(value, record)
    result: Counter[str] = Counter()
    for line in value.splitlines():
        for width in (1, 2, 3):
            result.update(prefix + _canonical(line[start:start + width]).decode("utf-8")
                          for start in range(len(line) - width + 1)
                          if line[start:start + width].strip())
    return {key: float(count) for key, count in result.items()}


def predictive_features(record: dict[str, Any], *, mode: Mode = "C0") -> dict[str, float]:
    """Label-free actual native channels, not an encoder or a route prediction.

    C0/C1 share body and dimension-kind/text coverage. C1 only adds finite raw
    measurements, separated by native dimension kind. Sorted value positions
    encode a multiset without exposing entity IDs or mixing angle/linear kinds.
    Presence counts retain genuine zero-valued measurements.
    """
    _validate_record(record, mode)
    features = _text_features(record["body_text"], "native_text/", record)
    numbers: dict[str, list[int | float]] = defaultdict(list)
    for node in record["dimensions"]:
        if not node["visibility"]["source_declared_not_hidden"]:
            continue
        kind = node["native_kind"]
        key = f"native_dimension_kind/{kind}"
        features[key] = features.get(key, 0.) + 1.
        user_text = node["raw_user_text_evidence"]
        if not user_text["conflicting"] and isinstance(user_text["raw_value"], str):
            for name, count in _text_features(user_text["raw_value"], f"native_dimension_text/{kind}/", record).items():
                features[name] = features.get(name, 0.) + count
        if node["raw_measurement_status"] == "finite_native_raw":
            numbers[kind].append(node["raw_measurement"])
    if mode == "C1":
        for kind, values in sorted(numbers.items()):
            features[f"native_measurement/{kind}/count"] = float(len(values))
            for rank, value in enumerate(sorted(values)):
                features[f"native_measurement/{kind}/rank/{rank}"] = float(value)
    return dict(sorted(features.items()))


def cache_fingerprint(record: dict[str, Any], *, mode: Mode = "C0") -> str:
    """Independent source/content/mode cache identity; never an old image digest."""
    features = predictive_features(record, mode=mode)
    return _digest({"consumer_schema": SCHEMA, "feature_schema": FEATURE_SCHEMA,
                    "mode": mode, "source_kind": "native_dwg",
                    "source_sha256": record["source_sha256"],
                    "response_sha256": record["response_sha256"],
                    "projection_sha256": record["projection_sha256"],
                    "native_ir_sha256": record["native_ir_sha256"],
                    "record_sha256": record["record_sha256"], "features": features})
