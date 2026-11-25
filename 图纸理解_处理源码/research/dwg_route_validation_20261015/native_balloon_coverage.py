"""Bounded full-FIT native annotation coverage, without ownership or route labels.

Only already archived source responses are parsed. Preview checks concern bytes
within one native object; CAD context, WCS, physical ownership and manufacturing
scope remain unvalidated. Missing and failed sources retain the frozen denominator.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import platform
import struct
import time
from typing import Any

import ezdxf

from research.dwg_route_validation_20261015.duplicate_audit import lossless_projection
from research.dwg_route_validation_20261015.native_balloon_probe import independent_preview_check, library_pins
from research.dwg_route_validation_20261015.native_balloon_vectors import _decode
from research.dwg_route_validation_20261015.native_dimension_display import strict_handle
from research.dwg_route_validation_20261015.native_dwg_consumer_probe import (
    BASE, REPO, digest, encode, file_guard, sha, write_new,
)
from research.dwg_route_validation_20261015.native_layer_roles import _field_evidence, _validate_lossless_structure
from research.dwg_route_validation_20261015.title_proxy_probe import CODEPAGE_ENCODINGS
from src.tools.research.source_graph import closure

SCHEMA = "native_balloon_coverage.v1"
PARENT = BASE / "native_object_binding_audit_001/PROTOCOL.json"
PARENT_SHA = "fb9719f1eef5d19236b38ba08ecd9a5aed67a51bedb1b5883028ec085db2d6ee"
RAW_LIMIT = 128 * 1024 * 1024
OUTPUT_LIMIT = 20 * 1024 * 1024
EVIDENCE_LIMIT = 12 * 1024 * 1024
CASE_LIMIT = 512 * 1024
_SOURCE_FIELDS = ("fit_index", "item_code", "group_id", "original_exists", "source_path",
                  "source_sha256", "source_bytes", "archive_path", "archive_sha256", "raw_response_sha256")
_PREVIEW_FIELDS = ("type", "handle", "preview_exists", "preview_size", "preview", "preview_is_proxy",
                   "entmode", "invisible", "num_unknown_bits")
POLICY = {"denominator": {"fit": 245, "present": 181, "missing": 64}, "physical_groups": 127,
          "selection": "unchanged parent inventory in original FIT order", "cpu_processes": 1,
          "max_decompressed_response_bytes": RAW_LIMIT, "max_output_bytes": OUTPUT_LIMIT,
          "max_uncompressed_evidence_bytes": EVIDENCE_LIMIT, "max_case_evidence_bytes": CASE_LIMIT,
          "class_selection": "actual CLASSES number to object type; only exact ZWMBALLOON decoded",
          "cad_context_validated": False, "wcs_coordinates_validated": False,
          "physical_ownership_validated": False, "eligible_for_route_training": False,
          "route_labels_read": False, "fitting_inputs_read": False, "query_inputs_read": False,
          "val_test_read": False, "pdf_or_images_read": False, "reader_calls": 0, "ocr_calls": 0,
          "gpu_used": False, "ownership_graph_built": False, "unknown_plugin_payload_decoded": False}


def _values(ev: dict) -> list:
    return [item["raw_value"] for item in ev["occurrences"]]


def _status(ev: dict, expected: type = str) -> str:
    if not ev["occurrence_count"]:
        return "not_declared"
    if ev["conflicting"]:
        return "conflicting"
    if type(ev["raw_value"]) is not expected:
        return "invalid_type"
    return "declared_empty" if ev["raw_value"] == "" else (
        "identical_repeated" if ev["occurrence_count"] > 1 else "declared")


def _kind(obj: dict, pointer: str) -> str:
    fields = [_field_evidence(obj, key, pointer) for key in ("entity", "object")]
    values = [ev["raw_value"] for ev in fields if ev["occurrence_count"]]
    return values[0] if (len(values) == 1 and type(values[0]) is str
                         and not any(ev["conflicting"] for ev in fields)) else "unknown"


def _scope(obj: dict, pointer: str) -> dict:
    return {key: _field_evidence(obj, key, pointer)
            for key in ("entmode", "ownerhandle", "invisible", "block_header")}


def _ref(position: int, obj: dict, source: dict) -> dict:
    ev = _field_evidence(obj, "handle", f"#/OBJECTS/{position}")
    value = strict_handle(ev["raw_value"], reference=False) if not ev["conflicting"] else None
    return {"source_sha256": source["source_sha256"], "response_sha256": source["raw_response_sha256"],
            "json_pointer": f"#/OBJECTS/{position}", "original_handle": deepcopy(ev["raw_value"]),
            "native_handle": f"{value:X}" if value is not None else None}


def _classes(parsed: dict) -> tuple[dict, dict, list]:
    ev = _field_evidence(parsed, "CLASSES", "#")
    classes = ev["raw_value"]
    if ev["conflicting"] or not isinstance(classes, list) or any(not isinstance(row, dict) for row in classes):
        raise ValueError("CLASSES_missing_conflicting_or_invalid")
    index, names, unknown = defaultdict(list), Counter(), []
    for position, row in enumerate(classes):
        pointer = f"#/CLASSES/{position}"
        number, name = (_field_evidence(row, key, pointer) for key in ("number", "dxfname"))
        record = {"json_pointer": pointer, "number_evidence": number, "dxfname_evidence": name}
        if _status(name) in {"declared", "identical_repeated"}:
            names[name["raw_value"]] += 1
        if (_status(number, int) not in {"declared", "identical_repeated"}
                or _status(name) not in {"declared", "identical_repeated"}):
            unknown.append(record)
        for value in {item for item in _values(number) if type(item) is int}:
            index[value].append(record)
    return index, dict(names), unknown


def _attribute_stats(objects: list) -> dict:
    stats = {kind: {"objects": 0, "field_statuses": {}, "exact_tag_counts": {}}
             for kind in ("ATTRIB", "ATTDEF")}
    for position, obj in enumerate(objects):
        pointer = f"#/OBJECTS/{position}"
        kind = _kind(obj, pointer)
        if kind not in stats:
            continue
        current = stats[kind]
        current["objects"] += 1
        for key, expected in (("tag", str), ("text_value", str), ("prompt", str),
                              ("ownerhandle", list), ("entmode", int)):
            ev = _field_evidence(obj, key, pointer)
            statuses = current["field_statuses"].setdefault(key, Counter())
            statuses[_status(ev, expected)] += 1
            if key == "tag" and _status(ev) in {"declared", "declared_empty", "identical_repeated"}:
                tag = ev["raw_value"]
                current["exact_tag_counts"][tag] = current["exact_tag_counts"].get(tag, 0) + 1
    return stats


def _reactors(objects: list, selected: dict[int, dict], handles: dict, identities: dict,
              source: dict) -> tuple[list, dict]:
    edges, counts = [], Counter()
    for position, obj in enumerate(objects):
        pointer = f"#/OBJECTS/{position}"
        array = _field_evidence(obj, "reactors", pointer)
        count = _field_evidence(obj, "num_reactors", pointer)
        if not array["occurrence_count"]:
            if count["conflicting"] or (count["occurrence_count"] and
                    (type(count["raw_value"]) is not int or count["raw_value"] != 0)):
                counts["missing_array_with_declared_count"] += 1
            continue
        # Identical field repetitions are one declaration, with their evidence retained.
        occurrences = array["occurrences"] if array["conflicting"] else array["occurrences"][:1]
        for occurrence in occurrences:
            refs = occurrence["raw_value"]
            if not isinstance(refs, list):
                counts["invalid_array"] += 1
                continue
            origin = identities[position]
            resolved = [strict_handle(raw, reference=True, origin=origin) for raw in refs]
            bad_array = array["conflicting"] or count["conflicting"] or (count["occurrence_count"] and
                (type(count["raw_value"]) is not int or count["raw_value"] != len(refs)))
            if bad_array:
                counts["conflicting_or_inconsistent_array"] += 1
            for slot, (raw, target) in enumerate(zip(refs, resolved, strict=True)):
                candidates = handles.get(target, []) if target is not None else []
                reasons = ["conflicting_or_inconsistent_array"] if bad_array else []
                if target is None:
                    reasons.append("invalid_reference")
                elif len(candidates) != 1 or identities[candidates[0]] != target:
                    reasons.append("missing_or_ambiguous_target_identity")
                if origin is None or len(handles.get(origin, [])) != 1:
                    reasons.append("invalid_or_ambiguous_source_identity")
                if target is not None and resolved.count(target) != 1:
                    reasons.append("duplicate_reference_slot")
                counts["all_reference_slots"] += 1
                for reason in reasons:
                    counts[reason] += 1
                selected_positions = [index for index in candidates if index in selected]
                if not selected_positions:
                    continue
                edge = {"source_ref": _ref(position, obj, source), "source_native_kind": _kind(obj, pointer),
                        "source_raw_scope": _scope(obj, pointer), "reference_pointer": f"{pointer}/reactors/{slot}",
                        "original_reference": raw, "occurrence_ordinal": occurrence["occurrence_ordinal"],
                        "array_occurrence_count": array["occurrence_count"], "count_evidence": count,
                        "target_candidates": [selected[index]["source_ref"] for index in selected_positions],
                        "status": "unknown" if reasons else "decoded_native_reference_only",
                        "reasons": reasons, "physical_owner": "unknown", "dependency_semantics": "unknown"}
                edges.append(edge)
                for index in selected_positions:
                    selected[index]["incoming_reactor_edge_indices"].append(len(edges) - 1)
    return edges, dict(counts)


def audit_case(parsed: dict[str, Any], source: dict[str, Any]) -> tuple[dict, dict]:
    """Light native coverage only: never build CAD ownership or a model input."""
    _validate_lossless_structure(parsed)
    objects = parsed.get("OBJECTS")
    if not isinstance(objects, list) or any(not isinstance(obj, dict) for obj in objects):
        raise ValueError("OBJECTS_missing_conflicting_or_invalid")
    classes, names, class_unknown = _classes(parsed)
    header = parsed.get("FILEHEADER", {})
    header = header if isinstance(header, dict) else {}
    cp, version = (_field_evidence(header, key, "#/FILEHEADER") for key in ("codepage", "version"))
    encoding = CODEPAGE_ENCODINGS.get(cp["raw_value"]) if _status(cp, int) in {
        "declared", "identical_repeated"} else None
    version_value = version["raw_value"] if _status(version) in {"declared", "identical_repeated"} else None
    handles, identities = defaultdict(list), {}
    for position, obj in enumerate(objects):
        ev = _field_evidence(obj, "handle", f"#/OBJECTS/{position}")
        identity = strict_handle(ev["raw_value"], reference=False) if not ev["conflicting"] else None
        identities[position] = identity
        for handle in {strict_handle(value, reference=False) for value in _values(ev)} - {None}:
            handles[handle].append(position)
    selected, counts, linked_classes, labels = {}, Counter(), Counter(), Counter()
    for position, obj in enumerate(objects):
        pointer = f"#/OBJECTS/{position}"
        type_ev = _field_evidence(obj, "type", pointer)
        matches = [row for value in {item for item in _values(type_ev) if type(item) is int}
                   for row in classes.get(value, [])]
        for name in {row["dxfname_evidence"]["raw_value"] for row in matches
                     if _status(row["dxfname_evidence"]) in {"declared", "identical_repeated"}}:
            linked_classes[name] += 1
        if not any("ZWMBALLOON" in _values(row["dxfname_evidence"]) for row in matches):
            continue
        reasons = []
        if (len(matches) != 1 or type_ev["conflicting"] or
                any(row["number_evidence"]["conflicting"] or row["dxfname_evidence"]["conflicting"]
                    or row["dxfname_evidence"]["raw_value"] != "ZWMBALLOON" for row in matches)):
            reasons.append("ambiguous_or_conflicting_class_type_join")
        if identities[position] is None or len(handles[identities[position]]) != 1:
            reasons.append("invalid_or_ambiguous_object_identity")
        row = {"source_ref": _ref(position, obj, source), "class_refs": matches,
               "field_evidence": {key: _field_evidence(obj, key, pointer) for key in _PREVIEW_FIELDS},
               "status": "unknown", "reasons": reasons, "frames": [], "geometry": [], "texts": [],
               "preview_sha256": None, "coordinate_status": "unknown", "style_payloads_unverified": []}
        _decode(row, encoding, version_value)
        independent = None
        if row["status"] == "native_vector_evidence":
            try:
                checked = independent_preview_check(bytes.fromhex(obj["preview"]), row)
                independent = {"status": checked["status"], "counts": checked["counts"]}
            except (ValueError, TypeError, IndexError, KeyError, OverflowError, struct.error) as error:
                row["status"] = "unknown"
                row["reasons"].append(f"independent_check_failed:{type(error).__name__}:{error}")
        status = "same_object_preview_checked" if row["status"] == "native_vector_evidence" else "unknown_preview"
        texts = [{key: text[key] for key in ("raw_text", "raw_text_hex", "frame_index", "byte_offset",
            "text_byte_start", "text_byte_end", "raw_start_point", "raw_normal", "raw_text_direction")}
                 for text in row["texts"]]
        for text in texts:
            labels[text["raw_text"]] += 1
        selected[position] = {"source_ref": row["source_ref"], "class_refs": matches, "status": status,
            "reasons": row["reasons"], "raw_scope": _scope(obj, pointer), "preview_sha256": row["preview_sha256"],
            "preview_fields": {key: ev for key, ev in row["field_evidence"].items() if key != "preview"},
            "preview_field_receipt": {"source_pointer": f"{pointer}/preview",
                "occurrence_count": row["field_evidence"]["preview"]["occurrence_count"],
                "conflicting": row["field_evidence"]["preview"]["conflicting"]},
            "frames": [{key: value for key, value in frame.items() if key != "raw_hex"} for frame in row["frames"]],
            "texts": texts, "geometry_counts": dict(Counter(item["type"] for item in row["geometry"])),
            "independent_check": independent, "incoming_reactor_edge_indices": [],
            "cad_context": "unvalidated", "coordinate_scope": "unvalidated_same_object_preview_only",
            "physical_owner": "unknown", "physical_part_confirmed": False}
        counts[status] += 1
    edges, reactor_counts = _reactors(objects, selected, handles, identities, source)
    counts["selected_objects"] = len(selected)
    counts["objects_with_decoded_incoming_reference"] = sum(any(
        edges[index]["status"] == "decoded_native_reference_only" for index in row["incoming_reactor_edge_indices"])
        for row in selected.values())
    summary = {"counts": dict(counts), "declared_class_name_counts": names,
               "type_linked_class_object_counts": dict(linked_classes), "unknown_class_declarations": len(class_unknown),
               "literal_preview_text_counts": dict(labels), "reactor_counts": reactor_counts,
               "incoming_reactor_source_kinds": dict(Counter(edge["source_native_kind"] for edge in edges)),
               "attributes": _attribute_stats(objects), "physical_part_bindings_confirmed": 0}
    evidence = {"header_evidence": {"codepage": cp, "version": version},
                "unknown_class_declarations": class_unknown, "balloons": list(selected.values()),
                "incoming_reactors": edges, "cad_context_validated": False,
                "coordinates_validated_against_CAD_space": False, "physical_owner": "unknown"}
    return summary, evidence


def _parent() -> dict:
    if digest(PARENT) != PARENT_SHA:
        raise ValueError("Frozen full-FIT source protocol changed")
    parent = json.loads(PARENT.read_bytes())
    if (parent.get("denominator") != POLICY["denominator"] or len(parent["inventory"]) != 245
            or any(parent.get(key) is not False for key in ("route_labels_read", "val_test_read", "query_input_files_read"))):
        raise ValueError("Require unchanged label-free 245/181/64 parent")
    return parent


def _inventory(parent: dict) -> list:
    rows = [{key: row[key] for key in _SOURCE_FIELDS if key in row} for row in parent["inventory"]]
    if (sum(row["original_exists"] is True for row in rows) != 181 or
            sum(row["original_exists"] is False for row in rows) != 64 or
            len({row["group_id"] for row in rows}) != 127):
        raise ValueError("Full FIT source or physical-group denominator changed")
    for row in rows:
        code = row["item_code"]
        if row["source_path"] != f"data/frozen/route_drawing_ready_20260909/assets/dwg/{code}.dwg":
            raise ValueError("Unexpected original source path")
        if row["original_exists"] and row["archive_path"] != (
                f"outputs/dwg_route_validation_20261015/train_fit_native_probe_001/cases/{code}/stdout.json.gz"):
            raise ValueError("Unexpected native archive path")
    return rows


def _pins(rows: list) -> dict:
    return {str(PARENT.relative_to(REPO)): PARENT_SHA, **{row[path]: row[pin] for row in rows
        if row["original_exists"] for path, pin in (("source_path", "source_sha256"), ("archive_path", "archive_sha256"))}}


def output_path(path: Path) -> Path:
    path = path.resolve()
    if path.parent != BASE.resolve() or not path.name.startswith("native_balloon_coverage_"):
        raise ValueError("Choose a new native_balloon_coverage_* immediate output child")
    return path


def verify(protocol: dict) -> None:
    parent = _parent()
    if (protocol.get("schema_version") != SCHEMA or protocol.get("policy") != POLICY
            or protocol.get("python_version") != platform.python_version()
            or protocol.get("ezdxf_version") != ezdxf.__version__ or protocol.get("package_pins") != library_pins()
            or protocol.get("inventory") != _inventory(parent) or protocol.get("source_pins") != _pins(_inventory(parent))):
        raise ValueError("Frozen coverage boundary/environment changed")
    current = {str(path.relative_to(REPO)): digest(path) for path in sorted(closure([Path(__file__)], REPO))}
    if protocol.get("code_pins") != current:
        raise ValueError("Coverage code closure changed")


def freeze(output: Path) -> dict:
    output = output_path(output)
    if output.exists():
        raise ValueError("Require a new output directory")
    rows = _inventory(_parent())
    protocol = {"schema_version": SCHEMA, "policy": POLICY, "python_version": platform.python_version(),
                "ezdxf_version": ezdxf.__version__, "package_pins": library_pins(),
                "inventory": rows, "source_pins": _pins(rows),
                "code_pins": {str(path.relative_to(REPO)): digest(path)
                              for path in sorted(closure([Path(__file__)], REPO))}}
    verify(protocol)
    output.mkdir(exist_ok=False)
    write_new(output / "PROTOCOL.json", encode(protocol))
    write_new(output / "producer.executed.py", Path(__file__).read_bytes())
    return {"protocol_sha256": digest(output / "PROTOCOL.json"), "denominator": POLICY["denominator"]}


def summarize(cases: list) -> dict:
    groups = {}
    for case in cases:
        group = groups.setdefault(case["group_id"], {"group_id": case["group_id"], "fit_count": 0,
            "present": 0, "missing": 0, "case_statuses": Counter(), "selected_objects": 0,
            "cases_with_checked_preview": 0, "cases_with_decoded_incoming_reference": 0})
        group["fit_count"] += 1
        group["present"] += case["original_exists"]
        group["missing"] += not case["original_exists"]
        group["case_statuses"][case["status"]] += 1
        counts = case.get("audit", {}).get("counts", {})
        group["selected_objects"] += counts.get("selected_objects", 0)
        group["cases_with_checked_preview"] += counts.get("same_object_preview_checked", 0) > 0
        group["cases_with_decoded_incoming_reference"] += counts.get("objects_with_decoded_incoming_reference", 0) > 0
    totals = {"fit": len(cases), "present": sum(case["original_exists"] for case in cases),
              "missing": sum(not case["original_exists"] for case in cases), "physical_groups": len(groups),
              "groups_with_present": sum(row["present"] > 0 for row in groups.values()),
              "groups_with_missing": sum(row["missing"] > 0 for row in groups.values()),
              "groups_all_present": sum(row["missing"] == 0 for row in groups.values()),
              "groups_all_missing": sum(row["present"] == 0 for row in groups.values()),
              "groups_mixed": sum(row["present"] > 0 and row["missing"] > 0 for row in groups.values()),
              "case_statuses": dict(Counter(case["status"] for case in cases))}
    for key in ("declared_class_name_counts", "type_linked_class_object_counts", "literal_preview_text_counts"):
        totals[key] = dict(sum((Counter(case.get("audit", {}).get(key, {})) for case in cases), Counter()))
    totals["counts"] = dict(sum((Counter(case.get("audit", {}).get("counts", {})) for case in cases), Counter()))
    return {"totals": totals, "physical_groups": list(groups.values())}


def compute(protocol: dict) -> tuple[dict, bytes, list]:
    cases, evidence, receipts, evidence_bytes = [], [], [], 0
    for source in protocol["inventory"]:
        started = time.monotonic()
        case = {key: source[key] for key in ("fit_index", "item_code", "group_id", "original_exists")}
        case["status"] = "missing_frozen_original"
        case["source_byte_hashes_verified"] = case["raw_response_sha256_verified"] = False
        receipt = {"fit_index": source["fit_index"], "source_verification_seconds": None, "processing_seconds": None}
        if source["original_exists"]:
            phase_started = time.monotonic()
            try:
                for path, pin in (("source_path", "source_sha256"), ("archive_path", "archive_sha256")):
                    if digest(REPO / source[path]) != source[pin]:
                        raise ValueError(f"frozen_{path}_hash_mismatch")
                receipt["source_verification_seconds"] = time.monotonic() - phase_started
                case["source_byte_hashes_verified"] = True
                phase_started = time.monotonic()
                with gzip.open(REPO / source["archive_path"], "rb") as stream:
                    raw = stream.read(RAW_LIMIT + 1)
                if len(raw) > RAW_LIMIT or sha(raw) != source["raw_response_sha256"]:
                    raise ValueError("raw_response_limit_or_hash_mismatch")
                case["raw_response_sha256_verified"] = True
                parsed, events = lossless_projection(raw)
                del raw, events
                audit, detail = audit_case(parsed, source)
                del parsed
                case["audit"] = audit
                count = audit["counts"]
                case["status"] = ("audited_no_selected_class_object" if not count["selected_objects"] else
                    "audited_with_unknown_previews" if count.get("unknown_preview", 0) else "audited_preview_sources")
                payload = encode(detail)
                if len(payload) > CASE_LIMIT or evidence_bytes + len(payload) > EVIDENCE_LIMIT - 256 * 1024:
                    case["status"] = "evidence_budget_exceeded"
                    detail = {"status": "evidence_budget_exceeded", "omitted_detail_sha256": sha(payload),
                              "omitted_detail_bytes": len(payload), "source_recoverable_from_pinned_archive": True}
                    payload = encode(detail)
                entry = {"fit_index": source["fit_index"], "item_code": source["item_code"], "evidence": detail}
                evidence_bytes += len(encode(entry)) + 1
                evidence.append(entry)
                del audit, detail, payload
                receipt["processing_seconds"] = time.monotonic() - phase_started
            except (ValueError, TypeError, IndexError, KeyError, OSError, EOFError, OverflowError, struct.error) as error:
                case["status"] = "source_or_audit_failed"
                case["error"] = f"{type(error).__name__}:{error}"
                # Explicitly release a failed source before reading the next archive.
                raw = parsed = events = None
                key = "source_verification_seconds" if receipt["source_verification_seconds"] is None else "processing_seconds"
                receipt[key] = time.monotonic() - phase_started
        receipt["total_seconds"] = time.monotonic() - started
        receipt["status"] = case["status"]
        cases.append(case)
        receipts.append(receipt)
    result = {"schema_version": SCHEMA, "policy": POLICY, "cases": cases, **summarize(cases),
              "route_accuracy_measured": False, "physical_part_bindings_confirmed": 0}
    retained_evidence = encode({"schema_version": SCHEMA, "cases": evidence})
    if len(retained_evidence) > EVIDENCE_LIMIT:
        raise ValueError("Encoded coverage evidence exceeds frozen byte bound")
    packed = gzip.compress(retained_evidence, mtime=0)
    result["evidence_gzip_sha256"] = sha(packed)
    return result, packed, receipts


def run(output: Path, *, replay: bool) -> dict:
    output = output_path(output)
    protocol = json.loads((output / "PROTOCOL.json").read_bytes())
    started = time.monotonic()
    with file_guard(protocol, output) as audit:
        verify(protocol)
        result, packed, cases = compute(protocol)
        verify(protocol)
    payload = encode(result)
    if replay:
        original = json.loads((output / "execution.json").read_bytes())
        if (original["pid"] == os.getpid() or original["protocol_sha256"] != digest(output / "PROTOCOL.json")
                or original["result_sha256"] != digest(output / "result.json")
                or (output / "result.json").read_bytes() != payload or (output / "evidence.json.gz").read_bytes() != packed):
            raise ValueError("Independent coverage replay differs")
    else:
        retained = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        if retained + len(payload) + len(packed) + 2 * 1024 * 1024 > OUTPUT_LIMIT:
            raise ValueError("Coverage output exceeds frozen 20MiB bound")
        write_new(output / "result.json", payload)
        write_new(output / "evidence.json.gz", packed)
    receipt = {"schema_version": SCHEMA, "pid": os.getpid(), "elapsed_seconds": time.monotonic() - started,
        "protocol_sha256": digest(output / "PROTOCOL.json"), "result_sha256": digest(output / "result.json"),
        "evidence_gzip_sha256": sha(packed), "independent_byte_equal_replay": replay, "cases": cases,
        "observed_pinned_repo_reads": sorted(audit["observed"]), "policy": POLICY}
    write_new(output / ("replay.json" if replay else "execution.json"), encode(receipt))
    if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > OUTPUT_LIMIT:
        raise ValueError("Coverage artifacts exceeded output bound")
    return {key: receipt[key] for key in ("schema_version", "pid", "elapsed_seconds", "result_sha256",
                                        "independent_byte_equal_replay")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "replay"))
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    result = freeze(options.output) if options.action == "freeze" else run(options.output, replay=options.action == "replay")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
