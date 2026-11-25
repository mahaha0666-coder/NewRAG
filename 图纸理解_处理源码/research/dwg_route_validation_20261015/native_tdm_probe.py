"""Freeze, extract and replay three native TDM preview sources without OCR.

This diagnostic preserves native literals and graphic commands. It does not
infer a BOM schema, a WCS transform, or physical manufacturing ownership.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import platform
import struct
import time

import ezdxf

from research.dwg_route_validation_20261015 import native_tdm_cells, native_tdm_previews
from research.dwg_route_validation_20261015.duplicate_audit import lossless_projection
from research.dwg_route_validation_20261015.native_balloon_probe import library_pins
from research.dwg_route_validation_20261015.native_dwg_consumer_probe import (
    BASE, REPO, digest, encode, file_guard, sha, write_new,
)
from src.tools.research.source_graph import closure

PARENT = BASE / "native_balloon_coverage_001"
PINS = {
    str((PARENT / "PROTOCOL.json").relative_to(REPO)): "dfee8d49c7f3f746398eea5cebb979ea0bcd3f4f0f977b48e4a2468a1d3b61d5",
    str((PARENT / "result.json").relative_to(REPO)): "ee612518c69ae953ede6f912b419c21dea7dde5c8d039ecdd5aa2a3ffd23ab1f",
}
ITEM_CODES = ["12500180941", "125002312743", "12502370454"]
SCHEMA = "native_tdm_probe.v1"
READ_LIMIT = 128 * 1024 * 1024
EVIDENCE_LIMIT = 64 * 1024 * 1024
OUTPUT_LIMIT = 10 * 1024 * 1024
POLICY = {
    "source_count": 3,
    "selection": "actual TDMLABEL and TDMBOMITEM counts > 0; source_sha256,fit_index order; first per physical group; first three",
    "previously_exposed_fit_groups_only": True,
    "eligible_files": 89, "eligible_physical_groups": 46,
    "native_dwg_read_purpose": "byte hash only",
    "geometry_source": "frozen native reader responses",
    "literal_fidelity_separate_from_coordinate_status": True,
    "opcode32_normals_independently_checked": True,
    "native_coordinates_not_automatically_wcs": True,
    "cells": "closed native axis-aligned line boundaries; text insertion candidates only",
    "schema_propagation_performed": False,
    "field_roles_inferred": False, "physical_part_ownership_validated": False,
    "final_visibility_validated": False, "units": "unknown",
    "route_labels_read": False, "query_inputs_read": False, "val_test_read": False,
    "pdf_or_images_read": False, "reader_calls": 0, "ocr_calls": 0,
    "route_training_performed": False, "gpu_used": False,
    "max_read_bytes": READ_LIMIT, "max_evidence_json_bytes": EVIDENCE_LIMIT,
    "max_output_bytes": OUTPUT_LIMIT,
}


def output_path(path: Path) -> Path:
    path = path.resolve()
    if path.parent != BASE.resolve() or not path.name.startswith("native_tdm_probe_"):
        raise ValueError("Choose a new native_tdm_probe_* immediate output child")
    return path


def selected_sources() -> list[dict]:
    """Selection uses class presence and source identities, never route answers."""
    for name, pin in PINS.items():
        if digest(REPO / name) != pin:
            raise ValueError("Frozen coverage protocol/result changed")
    parent = json.loads((PARENT / "PROTOCOL.json").read_bytes())
    coverage = json.loads((PARENT / "result.json").read_bytes())
    by_index = {row["fit_index"]: row for row in parent["inventory"]}
    if len(by_index) != len(parent["inventory"]):
        raise ValueError("Coverage inventory contains duplicate indices")
    eligible = []
    for case in coverage["cases"]:
        counts = case.get("audit", {}).get("type_linked_class_object_counts", {})
        if counts.get("TDMLABEL", 0) <= 0 or counts.get("TDMBOMITEM", 0) <= 0:
            continue
        row = by_index[case["fit_index"]]
        if (not row["original_exists"] or case["item_code"] != row["item_code"]
                or case["group_id"] != row["group_id"]):
            raise ValueError("Coverage source identity mismatch")
        eligible.append(row)
    if (len(eligible) != POLICY["eligible_files"] or
            len({row["group_id"] for row in eligible}) != POLICY["eligible_physical_groups"]):
        raise ValueError("TDM source eligibility changed")
    first = {}
    for row in sorted(eligible, key=lambda row: (row["source_sha256"], row["fit_index"])):
        first.setdefault(row["group_id"], row)
    selected = list(first.values())[:POLICY["source_count"]]
    if [row["item_code"] for row in selected] != ITEM_CODES:
        raise ValueError("Fixed three-source selection changed")
    return selected


def source_pins(inventory: list[dict]) -> dict[str, str]:
    return {**PINS, **{row[path]: row[pin] for row in inventory for path, pin in (
        ("source_path", "source_sha256"), ("archive_path", "archive_sha256"))}}


def verify(protocol: dict) -> None:
    if (protocol.get("schema_version") != SCHEMA or protocol.get("policy") != POLICY
            or protocol.get("python_version") != platform.python_version()
            or protocol.get("ezdxf_version") != ezdxf.__version__
            or protocol.get("package_pins") != library_pins()):
        raise ValueError("Frozen TDM policy/environment changed")
    inventory = selected_sources()
    if protocol["inventory"] != inventory or protocol["source_pins"] != source_pins(inventory):
        raise ValueError("TDM source read boundary changed")
    required = {str(Path(__file__).relative_to(REPO)),
                str(Path(native_tdm_previews.__file__).relative_to(REPO)),
                str(Path(native_tdm_cells.__file__).relative_to(REPO))}
    if not required.issubset(protocol["code_pins"]):
        raise ValueError("TDM code closure missing")
    for name, pin in {**protocol["source_pins"], **protocol["code_pins"]}.items():
        if digest(REPO / name) != pin:
            raise ValueError(f"Frozen TDM source/code changed: {name}")


def freeze(output: Path) -> dict:
    output = output_path(output)
    if output.exists():
        raise ValueError("Require a new TDM output directory")
    inventory = selected_sources()
    code = {str(path.relative_to(REPO)): digest(path)
            for path in sorted(closure([Path(__file__)], REPO))}
    protocol = {"schema_version": SCHEMA, "policy": POLICY,
                "python_version": platform.python_version(), "ezdxf_version": ezdxf.__version__,
                "package_pins": library_pins(), "source_pins": source_pins(inventory),
                "inventory": inventory, "code_pins": code}
    verify(protocol)
    output.mkdir(exist_ok=False)
    write_new(output / "PROTOCOL.json", encode(protocol))
    write_new(output / "producer.executed.py", Path(__file__).read_bytes())
    return {"protocol_sha256": digest(output / "PROTOCOL.json"), "item_codes": ITEM_CODES}


def independent_normal_checks(record: dict) -> list[dict]:
    """Re-read opcode32 raw bytes; never use the preview decoder's normal loader."""
    checks = []
    for obj in record["objects"]:
        for preview in obj["preview_candidates"]:
            geometries = {row["frame_index"]: row for row in preview["geometry"]}
            for frame in preview["frames"]:
                if frame["opcode"] != 32:
                    continue
                raw = bytes.fromhex(frame["raw_hex"])
                check = {"source_pointer": obj["source_ref"]["json_pointer"],
                         "preview_ordinal": preview["occurrence_ordinal"],
                         "frame_index": frame["index"], "raw_sha256": sha(raw)}
                if (len(raw) < 8 or struct.unpack_from("<II", raw) != (len(raw), 32)
                        or sha(raw) != frame["sha256"]):
                    raise ValueError("Independent opcode32 frame identity differs")
                count = struct.unpack_from("<I", raw, 8)[0] if len(raw) >= 12 else 0
                if count < 2 or len(raw) != 12 + 24 * (count + 1):
                    if frame["index"] in geometries or not frame["reasons"]:
                        raise ValueError("Invalid opcode32 extent was accepted")
                    checks.append({**check, "status": "invalid_extent_preserved_unknown"})
                    continue
                values = struct.unpack_from(f"<{3 * (count + 1)}d", raw, 12)
                safe = [value if math.isfinite(value) else None for value in values]
                vertices = [safe[index:index + 3] for index in range(0, count * 3, 3)]
                geometry = geometries[frame["index"]]
                if (geometry["native_vertices"] != vertices
                        or geometry["native_normal"] != safe[-3:]
                        or geometry["vertex_count"] != count):
                    raise ValueError("Independent opcode32 vertices or normal differ")
                checks.append({**check, "status": "independent_struct_fields_equal",
                               "vertex_count": count, "native_normal": safe[-3:],
                               "normal_is_positive_z": safe[-3:] == [0., 0., 1.]})
    return checks


def compute(protocol: dict) -> tuple[dict, bytes, list[dict]]:
    cases, evidence, timings = [], [], []
    for source in protocol["inventory"]:
        started = time.monotonic()
        with gzip.open(REPO / source["archive_path"], "rb") as stream:
            raw = stream.read(READ_LIMIT + 1)
        if len(raw) > READ_LIMIT or sha(raw) != source["raw_response_sha256"]:
            raise ValueError("TDM raw response exceeds bound or differs from frozen bytes")
        parsed, events = lossless_projection(raw)
        del raw, events
        record = native_tdm_previews.build_tdm_previews(
            parsed, source_sha256=source["source_sha256"],
            response_sha256=source["raw_response_sha256"])
        del parsed
        normals = independent_normal_checks(record)
        cells = native_tdm_cells.build_tdm_cells(record)
        identity = {key: source[key] for key in ("fit_index", "item_code", "group_id")}
        cases.append({**identity, "summary": record["summary"],
                      "record_sha256": record["record_sha256"],
                      "cell_summary": cells["summary"],
                      "cell_record_sha256": cells["record_sha256"],
                      "independent_opcode32_checks": len(normals)})
        evidence.append({**identity, "evidence": record, "cells": cells,
                         "independent_opcode32_checks": normals})
        timings.append({**identity, "archive_decode_projection_and_extraction_seconds":
                        time.monotonic() - started})
    raw_evidence = encode({"schema_version": SCHEMA, "cases": evidence})
    if len(raw_evidence) > EVIDENCE_LIMIT:
        raise ValueError("TDM raw evidence exceeds frozen byte bound")
    packed = gzip.compress(raw_evidence, mtime=0)
    result = {"schema_version": SCHEMA, "policy": POLICY, "cases": cases,
              "evidence_gzip_sha256": sha(packed), "physical_part_bindings_confirmed": 0,
              "route_accuracy_measured": False}
    return result, packed, timings


def run(output: Path, *, replay: bool) -> dict:
    output = output_path(output)
    protocol = json.loads((output / "PROTOCOL.json").read_bytes())
    started = time.monotonic()
    with file_guard(protocol, output) as audit:
        verify(protocol)
        result, packed, timings = compute(protocol)
        verify(protocol)
    payload = encode(result)
    if replay:
        original = json.loads((output / "execution.json").read_bytes())
        if (original["pid"] == os.getpid()
                or original["protocol_sha256"] != digest(output / "PROTOCOL.json")
                or original["result_sha256"] != digest(output / "result.json")
                or original["evidence_gzip_sha256"] != digest(output / "evidence.json.gz")
                or (output / "result.json").read_bytes() != payload
                or (output / "evidence.json.gz").read_bytes() != packed):
            raise ValueError("Independent TDM replay differs")
    else:
        retained = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        if retained + len(payload) + len(packed) + 128 * 1024 > OUTPUT_LIMIT:
            raise ValueError("TDM artifacts exceed frozen output bound")
        write_new(output / "result.json", payload)
        write_new(output / "evidence.json.gz", packed)
    receipt = {"schema_version": SCHEMA, "pid": os.getpid(),
               "elapsed_seconds": time.monotonic() - started,
               "timing_scope": "hash verification, archive decoding and native preview extraction; not fresh DWG read or route inference",
               "protocol_sha256": digest(output / "PROTOCOL.json"),
               "result_sha256": digest(output / "result.json"),
               "evidence_gzip_sha256": sha(packed), "independent_byte_equal_replay": replay,
               "cases": timings, "observed_pinned_repo_reads": sorted(audit["observed"]),
               "policy": POLICY}
    write_new(output / ("replay.json" if replay else "execution.json"), encode(receipt))
    if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > OUTPUT_LIMIT:
        raise ValueError("TDM final artifacts exceed output bound")
    return {key: receipt[key] for key in ("pid", "elapsed_seconds", "result_sha256",
                                        "independent_byte_equal_replay")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "replay"))
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    result = freeze(options.output) if options.action == "freeze" else run(
        options.output, replay=options.action == "replay")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
