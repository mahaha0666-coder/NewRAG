"""Optional source-bound native relation vectors; no routes or model heads.

The channel consumes verified direct CAD references as soft structural input.
It does not establish physical part ownership, units, processes or PDF pairing.
Identifiers join evidence only and never become feature names or values.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.sparse import csr_matrix, hstack

from research.r19_route_estimation.limited_search.text_views import input_digest

SCHEMA = "native_dimension_geometry_features.v1"
FORBIDDEN = {"route", "routes", "label", "labels", "gt", "gt_route", "operations",
             "candidates", "prediction", "predictions", "target_route"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _guards(value: Any) -> None:
    if isinstance(value, Mapping):
        if FORBIDDEN.intersection(value):
            raise ValueError("Native features require label-free input")
        if value.get("eligible_for_route_training", False) is not False:
            raise ValueError("The source gate does not authorize route training")
        for child in value.values():
            _guards(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _guards(child)


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("CAD coordinates must be finite numbers")
    return float(value)


def _point(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
        raise ValueError("An original CAD point is required")
    values = tuple(_number(item) for item in value)
    return (values[0], values[1], 0.) if len(values) == 2 else values


def _index(rows: Any, name: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{name} must be an original node list")
    result = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid {name} node")
        identity = row.get("id")
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError(f"Missing or duplicate {name} identity")
        result[identity] = row
    return result


def _coordinates(node: Mapping[str, Any]) -> Mapping[str, Any]:
    values = node.get("cad_coordinates")
    if not isinstance(values, Mapping):
        raise ValueError("Original cad_coordinates mapping is required")
    if node.get("native_visibility", 0) not in (None, 0):
        raise ValueError("Invisible CAD endpoints need a separate visibility review")
    if "extrusion" in values and values["extrusion"] != [0., 0., 1.]:
        raise ValueError("A nonstandard extrusion needs its original CAD frame transform")
    if "elevation" in values and _number(values["elevation"]) != 0:
        raise ValueError("A nonzero elevation cannot be flattened into this channel")
    return values


def _dimension(node: Mapping[str, Any]) -> dict[str, Any]:
    kind = node.get("native_kind")
    if not isinstance(kind, str) or not kind.startswith("DIMENSION_"):
        raise ValueError("Dimension endpoint is not a native DIMENSION")
    text = node.get("raw_text", "")
    if not isinstance(text, str):
        raise ValueError("Dimension text must remain an original literal")
    coords = _coordinates(node)
    anchor = coords.get("text_midpt", coords.get("def_pt"))
    point = _point(anchor)
    # Native text_midpt is 2D; its original common dimension elevation is z.
    if len(anchor) == 2:
        if "elevation" not in coords or coords.get("extrusion") != [0., 0., 1.]:
            raise ValueError("Unknown 2D dimension WCS: original elevation and standard extrusion required")
        point = point[0], point[1], _number(coords["elevation"])
    return {"kind": kind, "text": text, "anchor": point}


def _geometry(node: Mapping[str, Any]) -> dict[str, Any]:
    kind, coords = node.get("native_kind"), _coordinates(node)
    if kind == "LINE":
        if any(not isinstance(coords.get(key), (list, tuple)) or len(coords[key]) != 3 for key in ("start", "end")):
            raise ValueError("Unknown LINE z: both original 3D endpoints required")
        start, end = _point(coords.get("start")), _point(coords.get("end"))
        return {"kind": kind, "anchor": tuple((a + b) / 2 for a, b in zip(start, end, strict=True)),
                "extent": math.dist(start, end), "sweep": 0., "points": [start, end]}
    if kind not in {"ARC", "CIRCLE"}:
        raise ValueError("Only actual LINE/ARC/CIRCLE targets are admitted")
    if not isinstance(coords.get("center"), (list, tuple)) or len(coords["center"]) != 3:
        raise ValueError("Unknown round target z: original 3D center required")
    center, radius = _point(coords.get("center")), _number(coords.get("radius"))
    if radius <= 0:
        raise ValueError("A native round target needs positive radius")
    sweep = 1. if kind == "CIRCLE" else (
        (_number(coords.get("end_angle")) - _number(coords.get("start_angle"))) % math.tau
    ) / math.tau
    return {"kind": kind, "anchor": center, "extent": radius, "sweep": sweep,
            "points": [(center[0] - radius, center[1] - radius, center[2]),
                       (center[0] + radius, center[1] + radius, center[2])]}


def _validated(payload: Mapping[str, Any]) -> tuple[dict, dict, list]:
    _guards(payload)
    source = payload.get("source_sha256")
    if (payload.get("source_kind") != "native_dwg" or not isinstance(source, str)
            or len(source) != 64 or any(char not in "0123456789abcdef" for char in source)):
        raise ValueError("An exact original native DWG source identity is required")
    if payload.get("units") != "unknown" or payload.get("scope") != "unknown":
        raise ValueError("This channel retains unresolved units and physical scope")
    dimensions, geometry = (_index(payload.get(key), key) for key in ("dimensions", "geometry_nodes"))
    relations = payload.get("relations")
    if not isinstance(relations, list):
        raise ValueError("Typed original relations are required")
    accepted = {}
    for edge in relations:
        if (not isinstance(edge, Mapping) or edge.get("status") != "verified_direct_modelspace"
                or edge.get("dictionary_chain_verified") is not True
                or type(edge.get("dimension_entmode")) is not int
                or type(edge.get("target_entmode")) is not int
                or type(edge.get("native_xref_path_length")) is not int
                or edge.get("dimension_entmode") != 2 or edge.get("target_entmode") != 2
                or edge.get("native_xref_path_length") != 1 or edge.get("reference_field") != "xrefs"):
            raise ValueError("Only dictionary-confirmed single direct modelspace xrefs are consumable")
        identity = edge.get("dimension_id"), edge.get("geometry_id")
        if identity[0] not in dimensions or identity[1] not in geometry:
            raise ValueError("Relation endpoints must exist in the same native source")
        # Multiple reference slots are evidence, not extra physical instances.
        accepted[identity] = edge
    return dimensions, geometry, list(accepted.values())


def transform_native(payload: Mapping[str, Any]) -> dict[str, float]:
    """Joint dimension/target descriptors; no handle IDs, unit conversion or routes."""
    dimensions, geometry, edges = _validated(payload)
    if not edges:
        return {}
    # Unrelated block/3D/opaque nodes are retained by the producer, not silently
    # flattened or required to supply descriptors for this direct channel.
    used_dimensions = {edge["dimension_id"] for edge in edges}
    used_geometry = {edge["geometry_id"] for edge in edges}
    dims = {identity: _dimension(dimensions[identity]) for identity in used_dimensions}
    targets = {identity: _geometry(geometry[identity]) for identity in used_geometry}
    points = [node["anchor"] for node in dims.values()]
    points.extend(point for node in targets.values() for point in node["points"])
    low = tuple(min(point[axis] for point in points) for axis in range(3))
    high = tuple(max(point[axis] for point in points) for axis in range(3))
    scale = math.dist(low, high)
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("Native frame has no finite positive extent")
    sums: dict[str, list[float]] = {}
    for edge in edges:
        dim, target = dims[edge["dimension_id"]], targets[edge["geometry_id"]]
        ax, ay, az = ((dim["anchor"][axis] - low[axis]) / scale for axis in range(3))
        tx, ty, tz = ((target["anchor"][axis] - low[axis]) / scale for axis in range(3))
        # Exact user_text can differ from act_measurement; neither is reinterpreted.
        prefix = "native_relation/" + _json([dim["kind"], dim["text"], target["kind"]]) + "/"
        values = {"count": 1., "dx": tx - ax, "dy": ty - ay, "dz": tz - az,
                  "squared_distance": (tx - ax) ** 2 + (ty - ay) ** 2 + (tz - az) ** 2,
                  "x_product": ax * tx, "y_product": ay * ty,
                  "xy_product": ax * ty, "yx_product": ay * tx,
                  "z_product": az * tz, "xz_product": ax * tz, "yz_product": ay * tz,
                  "zx_product": az * tx, "zy_product": az * ty,
                  "target_extent": target["extent"] / scale,
                  "dimension_x_target_extent": ax * target["extent"] / scale,
                  "dimension_y_target_extent": ay * target["extent"] / scale,
                  "target_sweep_fraction": target["sweep"]}
        for name, value in values.items():
            sums.setdefault(prefix + name, []).append(value)
    return {key: math.fsum(sorted(values)) for key, values in sorted(sums.items())}


def cache_fingerprint(row: Mapping[str, Any], payload: Mapping[str, Any] | None = None) -> str:
    """Separate composition cache; legacy text/image cache identities stay intact."""
    _guards(row)
    if payload is None:
        return input_digest(row)
    _compatible(row, payload)
    features = transform_native(payload)
    if not features:
        return input_digest(row)
    return _digest({"schema": SCHEMA, "base_input_sha256": input_digest(row),
                    "native_relation_features": features})


def _compatible(row: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    if (row.get("source_kind") != "native_dwg"
            or row.get("source_sha256") != payload.get("source_sha256")
            or row.get("item_code") != payload.get("item_code")):
        raise ValueError("Unpaired PDF/source_unknown inputs cannot consume the native DWG channel")


class BaseFeatures(Protocol):
    def transform(self, rows: Sequence[dict]) -> csr_matrix: ...


@dataclass(frozen=True)
class NativeRelationFeatures:
    """Optional feature-factory composition, sharing one transform at fit/inference.

    The caller retains its actual TextFeatures/VisualFeatures object. A route
    head would need a separately authorized fit for the expanded feature space.
    Without new channel input the exact original matrix is returned.
    """

    base: BaseFeatures
    feature_names: tuple[str, ...]
    weight: float = 1.

    @classmethod
    def compose(cls, base: BaseFeatures, payloads: Sequence[Mapping[str, Any]],
                *, weight: float = 1.) -> NativeRelationFeatures:
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("The diagnostic channel weight must be positive and finite")
        names = {name for payload in payloads for name in transform_native(payload)}
        return cls(base, tuple(sorted(names)), float(weight))

    def transform(self, rows: Sequence[dict], payloads: Sequence[Mapping[str, Any] | None] | None = None,
                  *, edges_enabled: bool = True) -> csr_matrix:
        if type(edges_enabled) is not bool:
            raise ValueError("The edge control must be boolean")
        for row in rows:
            _guards(row)
        base = self.base.transform(rows)
        if payloads is None:
            return base
        if len(rows) != len(payloads):
            raise ValueError("Every row needs its own optional native source")
        values = []
        for row, payload in zip(rows, payloads, strict=True):
            if payload is None:
                values.append({})
                continue
            _compatible(row, payload)
            feature = transform_native(payload)
            if set(feature) - set(self.feature_names):
                raise ValueError("Unregistered relation vocabulary requires a new research feature contract")
            values.append(feature if edges_enabled else {})
        lookup = {name: index for index, name in enumerate(self.feature_names)}
        coordinates = [(index, lookup[name], value * self.weight)
                       for index, features in enumerate(values) for name, value in features.items()]
        local = csr_matrix(([value for _, _, value in coordinates],
                            ([row for row, _, _ in coordinates], [col for _, col, _ in coordinates])),
                           shape=(len(rows), len(self.feature_names)))
        # No renormalization: text/image block remains bit-identical across arms.
        return hstack((base, local), format="csr")


def _renamed(payload: Mapping[str, Any]) -> dict:
    value = deepcopy(payload)
    nodes = value["dimensions"] + value["geometry_nodes"]
    names = {node["id"]: f"renamed_{index}" for index, node in enumerate(reversed(nodes))}
    for node in nodes:
        node["id"] = names[node["id"]]
    for edge in value["relations"]:
        edge["dimension_id"] = names[edge["dimension_id"]]
        edge["geometry_id"] = names[edge["geometry_id"]]
    for key in ("dimensions", "geometry_nodes", "relations"):
        value[key].reverse()
    return value


def consumer_contract(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Execute real label-free TextFeatures plus native joint vectors on CPU.

    Reassignment is a synthetic negative control, never a verified CAD fact.
    No visual cache is borrowed from a PDF; full R6 VisualFeatures is not tested.
    """
    from research.r19_route_estimation.limited_search.models import TextFeatures

    repo = Path(__file__).resolve().parents[2]
    source_pins = {}
    for name in ("models.py", "text_views.py"):
        path = Path("research/r19_route_estimation/limited_search") / name
        current = hashlib.sha256((repo / path).read_bytes()).hexdigest()
        frozen = hashlib.sha256((repo / "release/delivery-20261007-v1/baseline/snapshot" / path).read_bytes()).hexdigest()
        if current != frozen:
            raise ValueError("Native text feature code differs from the frozen R6 implementation")
        source_pins[str(path)] = current
    rows = [{"item_code": payload["item_code"], "source_kind": "native_dwg",
             "source_sha256": payload["source_sha256"], "base_text": payload.get("original_text", "") or "empty",
             "evidence_text": "", "analysis_sha256": "", "split": "train"} for payload in payloads]
    before = deepcopy(rows)
    base = TextFeatures.fit(rows, {"embedding_path": None})
    composed = NativeRelationFeatures.compose(base, payloads)
    original = base.transform(rows)
    active = composed.transform(rows, payloads)
    removed = composed.transform(rows, payloads, edges_enabled=False)
    if not np.isfinite(active.data).all() or not np.isfinite(removed.data).all():
        raise ValueError("The composed feature matrix contains nonfinite values")
    cases = []
    for index, payload in enumerate(payloads):
        features = transform_native(payload)
        renamed = _renamed(payload)
        if features != transform_native(renamed):
            raise ValueError("Renaming or source inventory order changed native features")
        changed = None
        for first in range(len(payload["relations"])):
            if changed is not None:
                break
            for second in range(first + 1, len(payload["relations"])):
                trial = deepcopy(payload)
                left, right = trial["relations"][first], trial["relations"][second]
                if left["dimension_id"] == right["dimension_id"] or left["geometry_id"] == right["geometry_id"]:
                    continue
                left["geometry_id"], right["geometry_id"] = right["geometry_id"], left["geometry_id"]
                if len({(edge["dimension_id"], edge["geometry_id"]) for edge in trial["relations"]}) != len(trial["relations"]):
                    continue
                trial_features = transform_native(trial)
                if trial_features != features and set(trial_features) <= set(composed.feature_names):
                    changed = trial
                    break
        row = rows[index]
        if changed is not None:
            changed_matrix = composed.transform([row], [changed])
            if (changed_matrix != active[index]).nnz == 0:
                raise ValueError("Association-only control did not reach the final vector")
        pdf_unknown = {**row, "source_kind": "pdf", "pair_version_verified": False}
        try:
            composed.transform([pdf_unknown], [payload])
        except ValueError:
            rejected = True
        else:
            raise ValueError("An unpaired PDF consumed the DWG channel")
        cases.append({"item_code": payload["item_code"], "distinct_edges": len(payload["relations"]),
                      "native_relation_features": features,
                      "feature_columns_nonzero": sum(value != 0 for value in features.values()),
                      "renaming_and_order_invariant": True,
                      "association_only_vector_change": changed is not None,
                      "association_control_status": "passed" if changed is not None else "no_discriminating_swap_available",
                      "source_unknown_pdf_rejected": rejected,
                      "composition_cache_changes_with_edges": not features or cache_fingerprint(row, payload) != input_digest(row),
                      "association_cache_changes": changed is None or cache_fingerprint(row, changed) != cache_fingerprint(row, payload)})
    if rows != before or (composed.transform(rows) != original).nnz or (active[:, :original.shape[1]] != original).nnz:
        raise ValueError("Composition altered old inputs or the base feature matrix")
    return {"schema_version": "native_relation_consumer_contract.v1", "cases": cases,
            "base_feature_class": "TextFeatures", "pretrained_embedding_used": False,
            "frozen_r6_feature_source_pins": source_pins,
            "real_native_text_transform_executed": True, "full_frozen_r6_visual_transform_measured": False,
            "base_vector_columns": original.shape[1], "relation_vector_columns": len(composed.feature_names),
            "final_vector_shape": list(active.shape), "changed_final_vector_entries": (active != removed).nnz,
            "unchanged_original_text_and_base_vectors": True, "no_channel_exact_legacy_matrix": True,
            "units": "unknown", "scope": "unknown", "eligible_for_route_training": False,
            "native_z_retained": True, "nonstandard_extrusion_or_hidden_endpoint_admitted": False,
            "coordinate_contract": "native 3D geometry retained; 2D dimension text_midpt uses explicit original elevation only with extrusion [0,0,1]",
            "route_training_performed": False, "route_prediction_performed": False,
            "association_reassignment_is_synthetic_control_not_source_fact": True,
            "pdf_version_gate_relaxed": False, "visual_or_text_encoder_called": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Pinned native IR payload list")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true", help="Recompute in a fresh process and compare sealed result")
    args = parser.parse_args()
    data = json.loads(args.inputs.read_text())
    input_pins = {str(args.inputs.resolve()): hashlib.sha256(args.inputs.read_bytes()).hexdigest()}
    if isinstance(data, dict) and data.get("schema_version") == "fixed14_native_ir_receipt.v1":
        if len(data["records"]) != 14 or data.get("route_training_admitted") is not False:
            raise ValueError("Require all fixed14 label-free native sources")
        payloads = []
        for record in data["records"]:
            path = Path(record["path"])
            raw = path.read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            if actual != record["sha256"]:
                raise ValueError("Pinned original native IR changed")
            input_pins[str(path.resolve())] = actual
            payloads.append(json.loads(raw))
    else:
        payloads = data if isinstance(data, list) else data["cases"]
    result = consumer_contract(payloads)
    result["input_pins"] = input_pins
    result["input_sha256"] = hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.replay:
        if json.loads(args.output.read_text()) != result:
            raise ValueError("Independent native consumer replay differs")
        print(_json({"exact_result_equal": True, "case_count": len(result["cases"])}))
        return
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(_json({key: result[key] for key in ("base_vector_columns", "relation_vector_columns", "changed_final_vector_entries")}))


if __name__ == "__main__":
    main()
