"""Add finite ARC/CIRCLE checks to the unchanged native LINE diagnostic.

No nearest fallback, ownership inference, image, new reader or route label is used.
The preview-to-CAD identity remains an explicit unvalidated hypothesis.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import platform
import tarfile
import time

import ezdxf

from research.dwg_route_validation_20261015 import (
    native_cad_ownership, native_curve_contacts, native_insert_transforms, native_tdm_contact_probe,
)
from research.dwg_route_validation_20261015.duplicate_audit import lossless_projection
from research.dwg_route_validation_20261015.native_balloon_probe import library_pins
from research.dwg_route_validation_20261015.native_dwg_consumer_probe import (
    BASE, REPO, digest, encode, file_guard, sha, write_new,
)
from research.dwg_route_validation_20261015.native_insert_transform_probe import (
    ARCHIVE, ARCHIVE_SHA, MEMBER, MEMBER_SHA, close_points, reference_matrices,
)
from src.tools.research.source_graph import closure

PARENT = BASE / "native_tdm_contact_001"
PINS = {str((PARENT / name).relative_to(REPO)): pin for name, pin in (
    ("PROTOCOL.json", "1f65eea8a4f5e786cc20a32786711b0dcc243cbcf3f5da70d77d4cb142142018"),
    ("result.json", "e6724010791fc2c3e51bc06a75dce6bea9569698f2bace3ad54e38eda54c5129"),
    ("evidence.json.gz", "db4c2b02a1499f65ac4e77674db15881c30b2166a04bec1fde540bd0c65bdba5"),
)}
SCHEMA = "native_tdm_curve_probe.v1"
RAW_LIMIT = 128 * 1024 * 1024
EVIDENCE_LIMIT = 64 * 1024 * 1024
OUTPUT_LIMIT = 10 * 1024 * 1024
ROLES = ("non_support_free_endpoint", "support_free_endpoint")
POLICY = {
    "source_count": 3, "item_codes": native_tdm_contact_probe.POLICY["item_codes"],
    "control": "unchanged frozen LINE contacts and all 32 endpoint checks",
    "single_change": "add finite native ARC/CIRCLE contacts in the same qualified CAD context",
    "curve_normal_support": "explicit native +Z only; no inferred entity defaults",
    "insert_support": "verified invertible similarity transforms, including mirrors",
    "arc_angles": "native radians; counterclockwise finite sweep with endpoints",
    "arc_zero_or_full_turn_ambiguity": "unknown, never silently a full circle",
    "tolerance": "1e-8 + 1e-11 * max(abs(query XYZ), abs(modelspace full-circle AABB extrema))",
    "nearest_neighbor_assignment": False, "threshold_search": False,
    "curve_center_is_boundary": False, "connectivity_expansion": False,
    "no_new_contacts_action": "stop curve-catalog expansion; diagnose coordinate/reference evidence instead of threshold tuning",
    "new_contacts_claim": "conditional geometric coverage only; not physical attribution or route accuracy",
    "preview_to_cad_identity_validated": False, "physical_part_ownership_validated": False,
    "unique_contact_confirms_identity_or_ownership": False,
    "final_visibility": "unknown", "units": "unknown", "field_roles_inferred": False,
    "native_dwg_read_purpose": "byte hash only; geometry from frozen raw responses",
    "query_inputs_read": False, "route_labels_read": False, "val_test_read": False,
    "pdf_or_images_read": False, "reader_calls": 0, "ocr_calls": 0,
    "gpu_used": False, "route_training_performed": False,
    "max_read_bytes": RAW_LIMIT, "max_evidence_json_bytes": EVIDENCE_LIMIT,
    "max_output_bytes": OUTPUT_LIMIT,
}


def package_pins() -> dict:
    root = Path(ezdxf.__file__).resolve().parent
    return {**library_pins(), **{str(root / name): digest(root / name) for name in (
        "entities/arc.py", "entities/circle.py", "entities/dxfgfx.py")}}


def output_path(path: Path) -> Path:
    path = path.resolve()
    if path.parent != BASE.resolve() or not path.name.startswith("native_tdm_curve_"):
        raise ValueError("Choose a new native_tdm_curve_* immediate output child")
    return path


def parent_protocol() -> dict:
    for name, pin in PINS.items():
        if digest(REPO / name) != pin:
            raise ValueError("Frozen LINE control changed")
    protocol = json.loads((PARENT / "PROTOCOL.json").read_bytes())
    native_tdm_contact_probe.verify(protocol)
    if digest(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError("Native angle conversion authority changed")
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        member = archive.extractfile(MEMBER)
        if member is None:
            raise ValueError("Native angle conversion authority missing")
        with member:
            if sha(member.read(2 * 1024 * 1024 + 1)) != MEMBER_SHA:
                raise ValueError("Native angle conversion authority member changed")
    return protocol


def verify(protocol: dict) -> None:
    parent = parent_protocol()
    expected = {**parent["source_pins"], **PINS, str(ARCHIVE.relative_to(REPO)): ARCHIVE_SHA}
    if (protocol.get("schema_version") != SCHEMA or protocol.get("policy") != POLICY
            or protocol.get("python_version") != platform.python_version()
            or protocol.get("ezdxf_version") != ezdxf.__version__
            or protocol.get("package_pins") != package_pins()
            or protocol.get("source_pins") != expected or protocol.get("inventory") != parent["inventory"]):
        raise ValueError("Frozen curve diagnostic scope/environment changed")
    required = {str(Path(__file__).relative_to(REPO)),
                str(Path(native_curve_contacts.__file__).relative_to(REPO))}
    if not required.issubset(protocol["code_pins"]):
        raise ValueError("Native curve code closure missing")
    for name, pin in {**expected, **protocol["code_pins"]}.items():
        if digest(REPO / name) != pin:
            raise ValueError(f"Frozen curve source/code changed: {name}")


def freeze(output: Path) -> dict:
    output = output_path(output)
    if output.exists():
        raise ValueError("Require a new curve diagnostic directory")
    parent = parent_protocol()
    code = {str(path.relative_to(REPO)): digest(path) for path in sorted(closure([Path(__file__)], REPO))}
    protocol = {"schema_version": SCHEMA, "policy": POLICY,
                "python_version": platform.python_version(), "ezdxf_version": ezdxf.__version__,
                "package_pins": package_pins(), "inventory": parent["inventory"],
                "source_pins": {**parent["source_pins"], **PINS, str(ARCHIVE.relative_to(REPO)): ARCHIVE_SHA},
                "code_pins": code, "native_angle_authority": {"member": MEMBER, "sha256": MEMBER_SHA}}
    verify(protocol)
    output.mkdir(exist_ok=False)
    write_new(output / "PROTOCOL.json", encode(protocol))
    write_new(output / "producer.executed.py", Path(__file__).read_bytes())
    return {"protocol_sha256": digest(output / "PROTOCOL.json")}


def independent_samples(graph: dict, catalog: dict) -> dict:
    """Compare local native curves and every INSERT path to in-memory ezdxf."""
    accepted = [row for row in catalog["objects"] if row["status"] == "accepted"]
    matrices = reference_matrices(graph, {edge for row in accepted for edge in row["path_edge_ids"]})
    doc = ezdxf.new()
    checks, maximum = [], 0.
    for ordinal, curve in enumerate(catalog["objects"]):
        if curve["status"] != "accepted":
            continue
        center, radius = curve["native_center"], curve["native_radius"]
        if curve["native_kind"] == "ARC":
            entity = doc.modelspace().add_arc(center, radius, math.degrees(curve["start_angle"]),
                                              math.degrees(curve["end_angle"]))
            angles = [math.radians(value) for value in entity.angles(3)]
        else:
            entity = doc.modelspace().add_circle(center, radius)
            angles = [0., math.tau / 3, 2 * math.tau / 3]
        for angle, local in zip(angles, entity.vertices(map(math.degrees, angles)), strict=True):
            expected = list(local)
            for edge in reversed(curve["path_edge_ids"]):
                expected = list(matrices[edge].transform(expected))
            raw = [center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle), center[2]]
            actual = native_insert_transforms.transform_point(curve["matrix"], raw)
            delta = close_points(actual, expected)
            maximum = max(maximum, delta)
            checks.append({"curve_ordinal": ordinal, "native_angle_radians": angle,
                           "reference_modelspace_point": expected, "absolute_delta": delta})
    return {"samples": checks, "accepted_occurrences": len(accepted), "max_absolute_delta": maximum}


def diagnose_case(original: dict, graph: dict) -> tuple[dict, dict]:
    parent = original["evidence"]
    transforms = native_insert_transforms.build_instance_transforms(graph)
    if (graph["record_sha256"] != parent["ownership_record_sha256"]
            or transforms["record_sha256"] != parent["transform_record_sha256"]):
        raise ValueError("Curve experiment must consume exactly the LINE control geometry/paths")
    catalog = native_curve_contacts.curve_catalog(graph, transforms)
    comparison = independent_samples(graph, catalog)
    checks = []
    for obj in parent["contacts"]:
        for old in obj["endpoint_checks"]:
            hits, unknown = [], []
            for index, curve in enumerate(catalog["objects"]):
                if curve["status"] != "accepted":
                    continue
                result = native_curve_contacts.finite_curve_contact(old["endpoint"], curve)
                if result["numeric_contact"] is True:
                    hits.append({"curve_ordinal": index, "source_ref": curve["source_ref"],
                                 "path_edge_ids": curve["path_edge_ids"], "check": result})
                elif result["status"] == "unknown":
                    unknown.append({"curve_ordinal": index, "check": result})
            count = len(old["contacts"]) + len(hits)
            checks.append({"callout_source_ref": obj["source_ref"],
                **{key: old[key] for key in ("preview_ordinal", "support_candidate_index", "endpoint_role", "endpoint")},
                "baseline_line_status": old["contact_status"], "baseline_line_contacts": old["contacts"],
                "curve_contacts": hits, "curve_checks_unknown": unknown,
                "combined_contact_status": "zero" if not count else "unique" if count == 1 else "multiple",
                "preview_to_cad_identity_validated": False, "physical_owner": "unknown",
                "manufacturing_evidence_admitted": False})
    if len(checks) != parent["summary"]["endpoints_tested"]:
        raise ValueError("All unchanged LINE-control endpoint checks must remain")
    summary = {"curve_catalog": catalog["summary"], "endpoints_tested": len(checks),
               "baseline_statuses_by_role": {role: dict(Counter(c["baseline_line_status"] for c in checks
                   if c["endpoint_role"] == role)) for role in ROLES},
               "combined_statuses_by_role": {role: dict(Counter(c["combined_contact_status"] for c in checks
                   if c["endpoint_role"] == role)) for role in ROLES},
               "zero_line_endpoints_with_curve_contact": sum(c["baseline_line_status"] == "zero" and bool(c["curve_contacts"]) for c in checks),
               "unknown_curve_evaluations": sum(len(c["curve_checks_unknown"]) for c in checks),
               "independent_curve_samples": len(comparison["samples"]),
               "max_independent_sample_delta": comparison["max_absolute_delta"],
               "physical_part_bindings_confirmed": 0, "eligible_for_route_training": False}
    return summary, {"curve_catalog": catalog, "independent_samples": comparison, "endpoint_checks": checks,
                     "summary": summary, "preview_to_cad_identity_validated": False,
                     "eligible_for_route_training": False}


def bounded(path: Path, limit: int) -> bytes:
    with gzip.open(path, "rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Frozen native input exceeds read bound")
    return raw


def compute(protocol: dict) -> tuple[dict, bytes, list[dict]]:
    originals = json.loads(bounded(PARENT / "evidence.json.gz", EVIDENCE_LIMIT))["cases"]
    if [row["item_code"] for row in originals] != POLICY["item_codes"]:
        raise ValueError("All three unchanged native sources are required")
    cases, evidence, timings = [], [], []
    for source, original in zip(protocol["inventory"], originals, strict=True):
        started = time.monotonic()
        raw = bounded(REPO / source["archive_path"], RAW_LIMIT)
        if sha(raw) != source["raw_response_sha256"]:
            raise ValueError("Frozen native response changed")
        parsed, _ = lossless_projection(raw)
        graph = native_cad_ownership.build_ownership_graph(parsed,
            source_sha256=source["source_sha256"], response_sha256=source["raw_response_sha256"])
        del parsed, raw
        summary, detail = diagnose_case(original, graph)
        identity = {key: source[key] for key in ("fit_index", "item_code", "group_id")}
        cases.append({**identity, "summary": summary})
        evidence.append({**identity, "evidence": detail})
        timings.append({**identity, "archive_and_curve_diagnostic_seconds": time.monotonic() - started})
    payload = encode({"schema_version": SCHEMA, "cases": evidence})
    if len(payload) > EVIDENCE_LIMIT:
        raise ValueError("Native curve evidence exceeds output bound")
    packed = gzip.compress(payload, mtime=0)
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
        if (original["pid"] == os.getpid() or original["protocol_sha256"] != digest(output / "PROTOCOL.json")
                or original["result_sha256"] != digest(output / "result.json")
                or original["evidence_gzip_sha256"] != digest(output / "evidence.json.gz")
                or (output / "result.json").read_bytes() != payload or (output / "evidence.json.gz").read_bytes() != packed):
            raise ValueError("Independent native curve replay differs")
    else:
        retained = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        if retained + len(payload) + len(packed) + 128 * 1024 > OUTPUT_LIMIT:
            raise ValueError("Native curve artifact bound exceeded")
        write_new(output / "result.json", payload)
        write_new(output / "evidence.json.gz", packed)
    receipt = {"schema_version": SCHEMA, "pid": os.getpid(), "elapsed_seconds": time.monotonic() - started,
               "protocol_sha256": digest(output / "PROTOCOL.json"), "result_sha256": digest(output / "result.json"),
               "evidence_gzip_sha256": sha(packed), "independent_byte_equal_replay": replay,
               "timing_scope": "source hash, frozen archive, ownership, curve sampling and finite contacts; not new DWG reader parsing/upload/route inference",
               "cases": timings, "observed_pinned_repo_reads": sorted(audit["observed"]), "policy": POLICY}
    write_new(output / ("replay.json" if replay else "execution.json"), encode(receipt))
    if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > OUTPUT_LIMIT:
        raise ValueError("Native curve final artifacts exceed bound")
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
