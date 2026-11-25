"""Frozen native INSERT transforms checked against independent ezdxf geometry.

Consumes the fixed-six ownership evidence, not image/OCR/route-model caches.
Results describe coordinate consistency, never physical-part correctness.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import tarfile
import time

import ezdxf

from research.dwg_route_validation_20261015 import native_attribute_rows, native_insert_transforms
from research.dwg_route_validation_20261015.native_dwg_consumer_probe import (
    BASE, ITEM_CODES, REPO, digest, encode, file_guard, write_new,
)
from src.tools.research.source_graph import closure

PARENT = BASE / "native_cad_ownership_001"
PINS = {
    "PROTOCOL.json": "df1400365d30b36c16d3b17190de91cd5bd8f57c6940abae9210fbc4faf813d7",
    "result.json": "b07d51ea44a5ee47c50fc86d89c2fd59ac4114e62c69c95fb7ae9b1b80873f35",
    "evidence.json.gz": "f186bd9272016eff2fc6b96bcb6b634db550970a38bda6927b4ff88ff6cc6b96",
}
ARCHIVE = BASE / "parser_comparison_001/tools/libredwg-f860b0fb76d5f8effb5a70769a3389cb1e0308c8.tar.gz"
ARCHIVE_SHA = "03c98b4a62969e108b472e716971a3d4ca3ceac4e249e8ca116f7ea0cde2920d"
MEMBER = "libredwg-f860b0fb76d5f8effb5a70769a3389cb1e0308c8/src/out_dxf.c"
MEMBER_SHA = "fc3bd2074b5394cf4c8edf57f4cfdb342644ac48dad968830c782cd8a0e60f31"
SCHEMA = "native_insert_transform_probe.v1"
READ_LIMIT = 256 * 1024 * 1024
OUTPUT_LIMIT = 10 * 1024 * 1024
ABS_TOL = 1e-8
REL_TOL = 1e-11
CONTROL_POINTS = [[0., 0., 0.], [1.25, -2.5, 3.75], [-7., 11., -13.]]
POLICY = {
    "source_count": 6, "selection": "unchanged exposed FIT native DWGs",
    "ownership_evidence_reused": True, "native_rotation_units": "radians",
    "matrix_convention": "column_vector_parent_times_child",
    "abs_tolerance_native_units": ABS_TOL, "relative_tolerance": REL_TOL,
    "control_points": CONTROL_POINTS, "actual_primitive_points": "LINE start/end only",
    "native_attribute_row_projection": True, "attribute_to_physical_geometry_validated": False,
    "dimension_display_transform_validated": False, "physical_part_ownership_validated": False,
    "final_visibility_validated": False, "route_labels_read": False, "query_inputs_read": False,
    "val_test_read": False, "pdf_or_images_read": False, "ocr_calls": 0,
    "reader_calls": 0, "route_training_performed": False, "gpu_used": False,
    "max_graph_bytes": READ_LIMIT, "max_output_bytes": OUTPUT_LIMIT,
}


def package_pins() -> dict[str, str]:
    root = Path(ezdxf.__file__).resolve().parent
    matrix_module = importlib.import_module(ezdxf.math.Matrix44.__module__)
    paths = [root / "__init__.py", root / "entities/insert.py", root / "math/ucs.py",
             root / "math/_matrix44.py", Path(matrix_module.__file__).resolve()]
    return {str(path): digest(path) for path in sorted(set(paths))}


def output_path(path: Path) -> Path:
    path = path.resolve()
    if path.parent != BASE.resolve() or not path.name.startswith("native_insert_transform_"):
        raise ValueError("Choose a new native_insert_transform_* immediate output child")
    return path


def verify(protocol: dict) -> None:
    if (protocol.get("schema_version") != SCHEMA or protocol.get("policy") != POLICY
            or protocol.get("python_version") != platform.python_version()
            or protocol.get("ezdxf_version") != ezdxf.__version__
            or protocol.get("package_pins") != package_pins()):
        raise ValueError("Frozen transform policy, code environment or reference package changed")
    for name, pin in PINS.items():
        if digest(PARENT / name) != pin:
            raise ValueError("Fixed six-source native ownership evidence changed")
    parent = json.loads((PARENT / "PROTOCOL.json").read_bytes())
    expected = dict(parent["source_pins"])
    expected.update({str((PARENT / name).relative_to(REPO)): pin for name, pin in PINS.items()})
    expected[str(ARCHIVE.relative_to(REPO))] = ARCHIVE_SHA
    if protocol["source_pins"] != expected or protocol["item_codes"] != ITEM_CODES:
        raise ValueError("Transform evidence read boundary changed")
    required = {str(Path(__file__).relative_to(REPO)),
                str(Path(native_insert_transforms.__file__).relative_to(REPO)),
                str(Path(native_attribute_rows.__file__).relative_to(REPO))}
    if not required.issubset(protocol["code_pins"]):
        raise ValueError("Transform implementation must be pinned")
    for name, pin in {**protocol["source_pins"], **protocol["code_pins"]}.items():
        if digest(REPO / name) != pin:
            raise ValueError(f"Frozen transform source/code changed: {name}")
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        member = archive.extractfile(MEMBER)
        if member is None:
            raise ValueError("Missing pinned native angle-conversion authority")
        with member:
            raw = member.read(2 * 1024 * 1024 + 1)
    if hashlib.sha256(raw).hexdigest() != MEMBER_SHA:
        raise ValueError("Native radians authority member changed")


def freeze(output: Path) -> dict:
    output = output_path(output)
    if output.exists():
        raise ValueError("Require a new transform output directory")
    for name, pin in PINS.items():
        if digest(PARENT / name) != pin:
            raise ValueError("Require unchanged fixed-six ownership results")
    parent = json.loads((PARENT / "PROTOCOL.json").read_bytes())
    source_pins = dict(parent["source_pins"])
    source_pins.update({str((PARENT / name).relative_to(REPO)): pin for name, pin in PINS.items()})
    source_pins[str(ARCHIVE.relative_to(REPO))] = ARCHIVE_SHA
    code_pins = dict(parent["code_pins"])
    code_pins.update({str(path.relative_to(REPO)): digest(path)
                      for path in sorted(closure([Path(__file__)], REPO))})
    protocol = {"schema_version": SCHEMA, "policy": POLICY,
                "python_version": platform.python_version(), "ezdxf_version": ezdxf.__version__,
                "package_pins": package_pins(), "source_pins": source_pins, "code_pins": code_pins,
                "item_codes": ITEM_CODES, "native_angle_authority": {
                    "archive_member": MEMBER, "member_sha256": MEMBER_SHA,
                    "field_rd_lines": [655, 660], "value_bd_lines": [620, 625]}}
    verify(protocol)
    output.mkdir(exist_ok=False)
    write_new(output / "PROTOCOL.json", encode(protocol))
    write_new(output / "producer.executed.py", Path(__file__).read_bytes())
    return {"protocol_sha256": digest(output / "PROTOCOL.json")}


def raw_field(evidence: dict, field: str) -> object:
    row = evidence[field]
    if row["conflicting"] or not row["occurrence_count"]:
        raise ValueError("Verified transform consumed missing/conflicting evidence")
    return row["raw_value"]


def reference_matrices(graph: dict, required_edges: set[int]) -> dict[int, object]:
    """Build independent ezdxf INSERTs in memory, without writing a DXF."""
    doc = ezdxf.new()
    nodes = {node["source_ref"]["json_pointer"]: node for node in graph["objects"]}
    matrices = {}
    for edge in graph["reference_edges"]:
        if (edge["id"] not in required_edges or edge["relation"] != "insert_definition"
                or not edge["target_ref"]):
            continue
        node = nodes[edge["source_ref"]["json_pointer"]]
        definition = nodes[edge["target_ref"]["json_pointer"]]
        fields = node["field_evidence"]
        if node["native_kind"] != "INSERT":
            continue
        try:
            point, scale, rotation, extrusion = [raw_field(fields, name) for name in
                                                ("ins_pt", "scale", "rotation", "extrusion")]
            base = raw_field(definition["field_evidence"], "base_pt")
        except (KeyError, ValueError):
            continue
        name = f"NATIVE_{edge['id']}"
        doc.blocks.new(name, base_point=base)
        insert = doc.modelspace().add_blockref(name, point, dxfattribs={
            "xscale": scale[0], "yscale": scale[1], "zscale": scale[2],
            "rotation": math.degrees(rotation), "extrusion": extrusion})
        matrices[edge["id"]] = insert.matrix44()
    return matrices


def close_points(actual: list[float], expected: list[float]) -> float:
    if len(actual) != 3 or len(expected) != 3:
        raise ValueError("Only full 3D native coordinates may be compared")
    for a, e in zip(actual, expected, strict=True):
        if not (math.isfinite(a) and math.isfinite(e)) or abs(a - e) > ABS_TOL + REL_TOL * max(abs(a), abs(e)):
            raise ValueError("Independent native INSERT coordinate comparison failed")
    return max(abs(a - e) for a, e in zip(actual, expected, strict=True))


def compare_case(graph: dict, transforms: dict) -> dict:
    required = {edge for row in transforms["instances"] if row["status"] == "verified_affine"
                for edge in row["path_edge_ids"]}
    matrices = reference_matrices(graph, required)
    instances = {tuple(row["path_edge_ids"]): row for row in graph["modelspace_instances"]}
    nodes = {node["source_ref"]["json_pointer"]: node for node in graph["objects"]}
    counts, skipped = Counter(), Counter()
    checked_lines, max_delta, max_roundtrip = [], 0., 0.
    for row in transforms["instances"]:
        if row["status"] != "verified_affine":
            skipped.update(row["reasons"])
            continue
        path = row["path_edge_ids"]
        if any(edge not in matrices for edge in path):
            raise ValueError("Verified native path missing an independent reference transform")
        counts["verified_paths"] += 1
        counts["nested_paths"] += len(path) > 1
        original = instances[tuple(path)]
        checks = [(None, None, list(point)) for point in CONTROL_POINTS]
        for member in original["member_source_refs"]:
            node = nodes[member["json_pointer"]]
            if node["native_kind"] != "LINE":
                continue
            try:
                endpoints = [raw_field(node["field_evidence"], name) for name in ("start", "end")]
            except (KeyError, ValueError):
                counts["line_fields_unresolved"] += 1
                continue
            if any(not isinstance(point, list) or len(point) != 3 or any(
                    type(value) not in {int, float} or not math.isfinite(value) for value in point)
                   for point in endpoints):
                counts["line_fields_unresolved"] += 1
                continue
            checks.extend((member, field, point) for field, point in zip(("start", "end"), endpoints, strict=True))
            counts["line_occurrences"] += 1
        for source_ref, field, point in checks:
            expected = point
            for edge in reversed(path):
                expected = list(matrices[edge].transform(expected))
            actual = native_insert_transforms.transform_point(row["matrix"], point)
            max_delta = max(max_delta, close_points(actual, expected))
            back = native_insert_transforms.transform_point(row["inverse"], actual)
            max_roundtrip = max(max_roundtrip, close_points(back, point))
            counts["checked_points"] += 1
            if source_ref is not None:
                checked_lines.append({"path_edge_ids": path, "source_ref": source_ref,
                                      "field": field, "source_pointer": f"{source_ref['json_pointer']}/{field}",
                                      "native_point": point, "modelspace_point": actual})
    return {"counts": dict(counts), "unknown_reasons": dict(skipped),
            "max_absolute_reference_delta_native_units": max_delta,
            "max_absolute_roundtrip_delta_native_units": max_roundtrip,
            "line_endpoint_evidence": checked_lines}


def compute() -> tuple[dict, bytes]:
    with gzip.open(PARENT / "evidence.json.gz", "rb") as stream:
        raw = stream.read(READ_LIMIT + 1)
    if len(raw) > READ_LIMIT:
        raise ValueError("Ownership evidence exceeds frozen read bound")
    source = json.loads(raw)
    del raw
    results = json.loads((PARENT / "result.json").read_bytes())
    if [row["item_code"] for row in source["cases"]] != ITEM_CODES:
        raise ValueError("All six ownership cases must remain in evaluation")
    cases, evidence = [], []
    for case, expected in zip(source["cases"], results["cases"], strict=True):
        graph = case["graph"]
        before = encode(graph)
        if (case["item_code"] != expected["item_code"]
                or hashlib.sha256(before).hexdigest() != expected["graph_sha256"]):
            raise ValueError("Source ownership graph content changed")
        transforms = native_insert_transforms.build_instance_transforms(graph)
        comparison = compare_case(graph, transforms)
        attribute_rows = native_attribute_rows.project_attribute_rows(graph)
        if encode(graph) != before:
            raise ValueError("Coordinate validation mutated ownership evidence")
        evidence.append({"item_code": case["item_code"], "transforms": transforms,
                         "comparison": comparison, "attribute_rows": attribute_rows})
        cases.append({"item_code": case["item_code"], "source_sha256": graph["source_sha256"],
                      "summary": transforms["summary"],
                      "attribute_row_summary": attribute_rows["summary"],
                      "comparison": {key: value for key, value in comparison.items() if key != "line_endpoint_evidence"},
                      "transform_record_sha256": hashlib.sha256(encode(transforms)).hexdigest()})
    compressed = gzip.compress(encode({"schema_version": SCHEMA, "cases": evidence}), mtime=0)
    result = {"schema_version": SCHEMA, "policy": POLICY, "denominator": 6,
              "cases": cases, "evidence_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
              "reference": "ezdxf Insert.matrix44; in-memory objects, no DXF/image files"}
    return result, compressed


def run(output: Path, *, replay: bool) -> dict:
    output = output_path(output)
    protocol = json.loads((output / "PROTOCOL.json").read_bytes())
    started = time.monotonic()
    with file_guard(protocol, output) as audit:
        verify(protocol)
        result, evidence = compute()
        verify(protocol)
    payload = encode(result)
    if replay:
        original = json.loads((output / "execution.json").read_bytes())
        if (original["pid"] == os.getpid()
                or original["protocol_sha256"] != digest(output / "PROTOCOL.json")
                or original["result_sha256"] != digest(output / "result.json")
                or (output / "result.json").read_bytes() != payload
                or (output / "evidence.json.gz").read_bytes() != evidence):
            raise ValueError("Independent transform replay differs")
    else:
        retained = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        if retained + len(payload) + len(evidence) + 65536 > OUTPUT_LIMIT:
            raise ValueError("Complete transform output exceeds frozen bound")
        write_new(output / "result.json", payload)
        write_new(output / "evidence.json.gz", evidence)
    receipt = {"schema_version": SCHEMA, "pid": os.getpid(),
               "elapsed_seconds": time.monotonic() - started,
               "protocol_sha256": digest(output / "PROTOCOL.json"),
               "result_sha256": digest(output / "result.json"),
               "evidence_gzip_sha256": hashlib.sha256(evidence).hexdigest(),
               "independent_byte_equal_replay": replay,
               "observed_pinned_repo_reads": sorted(audit["observed"]), "policy": POLICY}
    write_new(output / ("replay.json" if replay else "execution.json"), encode(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "replay"))
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    receipt = freeze(options.output) if options.action == "freeze" else run(
        options.output, replay=options.action == "replay")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
