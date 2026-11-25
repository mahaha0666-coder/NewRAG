"""Conditional native-preview identity diagnostic, not a part-binding system.

Only three frozen FIT native DWGs are used. Even one exact finite LINE contact
does not validate the assumed preview-to-CAD identity transform or part owner.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import platform
import time

import ezdxf

from research.dwg_route_validation_20261015 import (
    native_cad_ownership, native_insert_transforms, native_tdm_callouts, native_tdm_probe,
)
from research.dwg_route_validation_20261015.duplicate_audit import lossless_projection
from research.dwg_route_validation_20261015.native_balloon_probe import (
    finite_line_distance, library_pins, line_catalog,
)
from research.dwg_route_validation_20261015.native_dwg_consumer_probe import (
    BASE, REPO, digest, encode, file_guard, sha, write_new,
)
from research.dwg_route_validation_20261015.native_insert_transform_probe import compare_case
from src.tools.research.source_graph import closure

PARENT = BASE / "native_tdm_probe_001"
PINS = {str((PARENT / name).relative_to(REPO)): pin for name, pin in (
    ("PROTOCOL.json", "afc1930f4300d5fca00b60b2c68c6f26f72f0283f5efe9fb01abd44aa18de5da"),
    ("result.json", "db17776379265bb2a475bd1fbbe787035c2f1ebf9f52a01cee29f22984a4bf5b"),
    ("evidence.json.gz", "7513ab251e5f0ba6d22b0ff1dcd60fc5421ba87641eec35d88d29a844fd3a016"),
)}
SCHEMA = "native_tdm_contact_probe.v1"
RAW_LIMIT = 128 * 1024 * 1024
EVIDENCE_LIMIT = 64 * 1024 * 1024
OUTPUT_LIMIT = 10 * 1024 * 1024
POLICY = {
    "source_count": 3, "item_codes": native_tdm_probe.ITEM_CODES,
    "native_preview_to_cad_transform": "identity_assumption_for_diagnostic_only",
    "preview_to_cad_identity_validated": False,
    "cad_container": "modelspace context plus unique verified modelspace membership",
    "endpoint_selection": "all qualified support hypotheses; test both free endpoints",
    "line_scope": "native modelspace LINEs and independently checked INSERT LINE occurrences",
    "dimension_display_or_unknown_instance_paths_included": False,
    "degenerate_lines_included": False,
    "distance": "finite 3D clamped segment projection; retain every numeric contact",
    "absolute_tolerance_native_units": 1e-8, "relative_tolerance": 1e-11,
    "nearest_neighbor_assignment": False, "threshold_search": False,
    "connectivity_expansion": False, "physical_part_ownership_validated": False,
    "unique_contact_confirms_identity_or_ownership": False,
    "field_roles_inferred": False, "units": "unknown", "final_visibility": "unknown",
    "native_dwg_read_purpose": "byte hash only; geometry from frozen raw responses",
    "route_labels_read": False, "query_inputs_read": False, "val_test_read": False,
    "pdf_or_images_read": False, "reader_calls": 0, "ocr_calls": 0,
    "route_training_performed": False, "gpu_used": False,
    "max_read_bytes": RAW_LIMIT, "max_evidence_json_bytes": EVIDENCE_LIMIT,
    "max_output_bytes": OUTPUT_LIMIT,
}


def output_path(path: Path) -> Path:
    path = path.resolve()
    if path.parent != BASE.resolve() or not path.name.startswith("native_tdm_contact_"):
        raise ValueError("Choose a new native_tdm_contact_* immediate output child")
    return path


def parent_protocol() -> dict:
    for path, pin in PINS.items():
        if digest(REPO / path) != pin:
            raise ValueError("Frozen TDM preview evidence changed")
    parent = json.loads((PARENT / "PROTOCOL.json").read_bytes())
    native_tdm_probe.verify(parent)
    return parent


def verify(protocol: dict) -> None:
    parent = parent_protocol()
    if (protocol.get("schema_version") != SCHEMA or protocol.get("policy") != POLICY
            or protocol.get("python_version") != platform.python_version()
            or protocol.get("ezdxf_version") != ezdxf.__version__
            or protocol.get("package_pins") != library_pins()
            or protocol.get("inventory") != parent["inventory"]
            or protocol.get("source_pins") != {**parent["source_pins"], **PINS}):
        raise ValueError("Frozen conditional TDM scope/environment changed")
    required = {str(Path(__file__).relative_to(REPO)),
                str(Path(native_tdm_callouts.__file__).relative_to(REPO))}
    if not required.issubset(protocol["code_pins"]):
        raise ValueError("Conditional TDM code closure missing")
    for name, pin in {**protocol["source_pins"], **protocol["code_pins"]}.items():
        if digest(REPO / name) != pin:
            raise ValueError(f"Frozen conditional TDM source/code changed: {name}")


def freeze(output: Path) -> dict:
    output = output_path(output)
    if output.exists():
        raise ValueError("Require a new conditional TDM output directory")
    parent = parent_protocol()
    code = {str(path.relative_to(REPO)): digest(path)
            for path in sorted(closure([Path(__file__)], REPO))}
    protocol = {"schema_version": SCHEMA, "policy": POLICY,
                "python_version": platform.python_version(), "ezdxf_version": ezdxf.__version__,
                "package_pins": library_pins(), "inventory": parent["inventory"],
                "source_pins": {**parent["source_pins"], **PINS}, "code_pins": code}
    verify(protocol)
    output.mkdir(exist_ok=False)
    write_new(output / "PROTOCOL.json", encode(protocol))
    write_new(output / "producer.executed.py", Path(__file__).read_bytes())
    return {"protocol_sha256": digest(output / "PROTOCOL.json")}


def contacts_under_identity(point: list[float], lines: list[dict]) -> dict:
    """An exact numerical contact remains conditional even when unique."""
    contacts = []
    for line in lines:
        if line["start"] == line["end"]:
            raise ValueError("Degenerate LINE must be counted and excluded before contact checks")
        check = finite_line_distance(point, line["start"], line["end"])
        if check["numeric_contact"]:
            contacts.append({"line": line, "check": check})
    return {"endpoint": point, "contact_status": "zero" if not contacts else "unique" if len(contacts) == 1 else "multiple",
            "contacts": contacts, "preview_to_cad_identity_validated": False,
            "physical_owner": "unknown", "manufacturing_evidence_admitted": False}


def diagnose_case(case: dict, graph: dict) -> tuple[dict, dict]:
    topology = native_tdm_callouts.build_callout_topology(case["evidence"], case["cells"])
    if len(topology["objects"]) != case["evidence"]["summary"]["class_counts"].get("TDMLABEL", 0):
        raise ValueError("All frozen native callout objects must remain in the diagnostic")
    transforms = native_insert_transforms.build_instance_transforms(graph)
    comparison = compare_case(graph, transforms)
    catalog = line_catalog(graph, {"comparison": comparison})
    lines = [line for line in catalog if line["start"] != line["end"]]
    nodes = {row["source_ref"]["json_pointer"]: row for row in graph["objects"]}
    spaces = {row["source_ref"]["native_handle"]: row["declared_native_space"] for row in graph["definitions"]}
    contacts = []
    for obj in topology["objects"]:
        ref = obj["source_ref"]
        node = nodes[ref["json_pointer"]]
        if node["source_ref"] != ref:
            raise ValueError("Callout source identity differs from native ownership graph")
        memberships = [edge for edge in graph["reference_edges"] if edge["relation"] == "block_member"
                       and edge["target_ref"] == ref]
        qualified = (node["cad_context"]["cad_space"] == "modelspace" and not node["cad_context"]["reasons"]
                     and len(memberships) == 1 and memberships[0]["status"] == "verified_membership"
                     and spaces[memberships[0]["source_ref"]["native_handle"]] == "modelspace")
        entry = {"source_ref": ref, "cad_context": node["cad_context"],
                 "membership_evidence": memberships, "cad_container_qualified": qualified,
                 "preview_to_cad_identity_validated": False, "endpoint_checks": [],
                 "physical_owner": "unknown", "manufacturing_evidence_admitted": False}
        if qualified and not graph["truncation"]["truncated"]:
            for preview in obj["previews"]:
                for index, support in enumerate(preview["support_candidates"]):
                    for key in ("non_support_free_endpoint", "support_free_endpoint"):
                        entry["endpoint_checks"].append({"preview_ordinal": preview["preview_ref"]["occurrence_ordinal"],
                            "support_candidate_index": index, "endpoint_role": key,
                            **contacts_under_identity(support[key], lines)})
        else:
            entry["diagnostic_skipped_reason"] = "cad_container_unknown_or_graph_truncated"
        contacts.append(entry)
    summary = {"callout_objects": len(contacts), "cad_container_qualified": sum(c["cad_container_qualified"] for c in contacts),
               "qualified_line_occurrences": len(lines), "degenerate_line_occurrences_excluded": len(catalog) - len(lines),
               "endpoint_contact_statuses": dict(Counter(c["contact_status"] for row in contacts for c in row["endpoint_checks"])),
               "endpoint_contact_statuses_by_role": {role: dict(Counter(c["contact_status"]
                   for row in contacts for c in row["endpoint_checks"] if c["endpoint_role"] == role))
                   for role in ("non_support_free_endpoint", "support_free_endpoint")},
               "endpoints_tested": sum(len(c["endpoint_checks"]) for c in contacts),
               "physical_part_bindings_confirmed": 0, "preview_to_cad_identity_validated": False}
    detail = {"topology": topology, "contacts": contacts, "summary": summary,
              "ownership_record_sha256": graph["record_sha256"],
              "transform_record_sha256": transforms["record_sha256"], "transform_summary": transforms["summary"],
              "independent_transform_comparison": {k: v for k, v in comparison.items() if k != "line_endpoint_evidence"},
              "line_catalog_sha256": sha(encode(lines)), "eligible_for_route_training": False}
    return summary, detail


def compute(protocol: dict) -> tuple[dict, bytes, list[dict]]:
    with gzip.open(PARENT / "evidence.json.gz", "rb") as stream:
        raw = stream.read(EVIDENCE_LIMIT + 1)
    if len(raw) > EVIDENCE_LIMIT:
        raise ValueError("Frozen TDM evidence exceeds read bound")
    originals = json.loads(raw)["cases"]
    del raw
    if [c["item_code"] for c in originals] != POLICY["item_codes"]:
        raise ValueError("All three unchanged TDM sources are required")
    cases, evidence, timings = [], [], []
    for source, original in zip(protocol["inventory"], originals, strict=True):
        started = time.monotonic()
        with gzip.open(REPO / source["archive_path"], "rb") as stream:
            raw = stream.read(RAW_LIMIT + 1)
        if len(raw) > RAW_LIMIT or sha(raw) != source["raw_response_sha256"]:
            raise ValueError("Frozen native raw response exceeds bound or differs")
        parsed, events = lossless_projection(raw)
        del raw, events
        graph = native_cad_ownership.build_ownership_graph(parsed,
            source_sha256=source["source_sha256"], response_sha256=source["raw_response_sha256"])
        del parsed
        summary, detail = diagnose_case(original, graph)
        del graph
        identity = {key: source[key] for key in ("fit_index", "item_code", "group_id")}
        cases.append({**identity, "summary": summary, "topology_summary": detail["topology"]["summary"]})
        evidence.append({**identity, "evidence": detail})
        timings.append({**identity, "native_archive_and_conditional_diagnostic_seconds": time.monotonic() - started})
    raw_evidence = encode({"schema_version": SCHEMA, "cases": evidence})
    if len(raw_evidence) > EVIDENCE_LIMIT:
        raise ValueError("Conditional native evidence exceeds frozen byte bound")
    packed = gzip.compress(raw_evidence, mtime=0)
    return {"schema_version": SCHEMA, "policy": POLICY, "cases": cases,
            "evidence_gzip_sha256": sha(packed)}, packed, timings


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
            raise ValueError("Independent conditional native replay differs")
    else:
        retained = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        if retained + len(payload) + len(packed) + 128 * 1024 > OUTPUT_LIMIT:
            raise ValueError("Conditional native artifacts exceed output bound")
        write_new(output / "result.json", payload)
        write_new(output / "evidence.json.gz", packed)
    receipt = {"schema_version": SCHEMA, "pid": os.getpid(), "elapsed_seconds": time.monotonic() - started,
               "protocol_sha256": digest(output / "PROTOCOL.json"), "result_sha256": digest(output / "result.json"),
               "evidence_gzip_sha256": sha(packed), "independent_byte_equal_replay": replay,
               "timing_scope": "source hash, archive, ownership, affine checks and conditional finite LINE contacts; not fresh DWG read/upload/route inference",
               "cases": timings, "observed_pinned_repo_reads": sorted(audit["observed"]), "policy": POLICY}
    write_new(output / ("replay.json" if replay else "execution.json"), encode(receipt))
    if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > OUTPUT_LIMIT:
        raise ValueError("Conditional native final artifacts exceed output bound")
    return {key: receipt[key] for key in ("pid", "elapsed_seconds", "result_sha256", "independent_byte_equal_replay")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "replay"))
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    result = freeze(options.output) if options.action == "freeze" else run(options.output, replay=options.action == "replay")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
