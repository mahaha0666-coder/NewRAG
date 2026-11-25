"""Compare two isolated LibreDWG readers using only explicit native references.

Raw stdout/stderr and source hashes are preserved. No geometry proximity,
route labels, GT, Val/Test, or model output enters this probe.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from typing import Any


REPO = Path(__file__).resolve().parents[2]
FIXED_CODES = ("12830010725", "12500575523", "12501700017")
SOURCE_COMMIT = "f860b0fb76d5f8effb5a70769a3389cb1e0308c8"
GEOMETRY_TYPES = frozenset({"LINE", "ARC", "CIRCLE"})
TEXT_TYPES = frozenset({"TEXT", "MTEXT", "ATTRIB", "ATTDEF", "TOLERANCE"})
TEXT_FIELDS = ("text_value", "text", "user_text", "tag", "prompt")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def handle_value(value: Any) -> int | None:
    """Read modern object [code,size,value] or resolved four-slot references.

    This probe's sources are AC1018. Three slots do not establish the meaning
    of historical PRE_R13 references; those are outside this protocol.
    """
    if not isinstance(value, list) or len(value) not in {2, 3, 4}:
        return None
    if any(isinstance(slot, bool) or not isinstance(slot, int) or slot < 0 for slot in value):
        return None
    return value[-1] if value[-1] > 0 else None


def hex_handle(value: int | None) -> str | None:
    return f"{value:X}" if value is not None else None


def object_kind(obj: dict[str, Any] | None) -> str | None:
    if obj is None:
        return None
    if obj.get("entity") == "UNKNOWN_ENT" or obj.get("object") == "UNKNOWN_OBJ":
        return obj.get("dxfname") or obj.get("_source_class_dxfname") or obj.get("entity") or obj.get("object")
    return obj.get("entity") or obj.get("dxfname") or obj.get("object")


def load_response(raw: bytes) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Try lossless UTF-8 then GBK; never repair or replace response bytes."""
    utf8_valid = True
    try:
        text = raw.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError as utf8_error:
        utf8_valid = False
        try:
            text = raw.decode("gbk")
            encoding = "gbk"
        except UnicodeDecodeError as gbk_error:
            return None, {
                "utf8_valid": False,
                "json_success": False,
                "lossy_replacement_used": False,
                "utf8_error": str(utf8_error),
                "gbk_error": str(gbk_error),
            }
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        return None, {
            "utf8_valid": utf8_valid,
            "chosen_lossless_encoding": encoding,
            "json_success": False,
            "lossy_replacement_used": False,
            "json_error": str(error),
        }
    if not isinstance(parsed, dict) or not isinstance(parsed.get("OBJECTS"), list):
        return None, {
            "utf8_valid": utf8_valid,
            "chosen_lossless_encoding": encoding,
            "json_success": False,
            "lossy_replacement_used": False,
            "json_error": "Expected dictionary with OBJECTS list",
        }
    return parsed, {
        "utf8_valid": utf8_valid,
        "chosen_lossless_encoding": encoding,
        "json_success": True,
        "lossy_replacement_used": False,
        "replacement_char_count": text.count("\ufffd"),
    }


def native_index(objects: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    for obj in objects:
        handle = handle_value(obj.get("handle"))
        if handle is not None:
            if handle in index:
                raise ValueError(f"Duplicate native object handle {handle:X}")
            index[handle] = obj
    return index


def text_records(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for obj in objects:
        kind = object_kind(obj) or ""
        if kind not in TEXT_TYPES and not kind.startswith("DIMENSION_"):
            continue
        for field in TEXT_FIELDS:
            value = obj.get(field)
            if isinstance(value, str) and value:
                records.append({
                    "handle": hex_handle(handle_value(obj.get("handle"))),
                    "kind": kind,
                    "field": field,
                    "value": value,
                    "non_ascii": any(ord(char) > 127 for char in value),
                    "replacement_chars": value.count("\ufffd"),
                })
    return records


def association_records(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = native_index(objects)
    records = []
    for array_index, obj in enumerate(objects):
        if object_kind(obj) != "DIMASSOC":
            continue
        assoc_handle = handle_value(obj.get("handle"))
        dimension_handle = handle_value(obj.get("dimensionobj"))
        dimension = index.get(dimension_handle) if dimension_handle else None
        dimension_kind = object_kind(dimension) or ""
        decoded = obj.get("object") == "DIMASSOC" and isinstance(obj.get("ref"), list)
        dictionary_handle = handle_value(obj.get("ownerhandle"))
        dictionary = index.get(dictionary_handle) if dictionary_handle else None
        chain_verified = bool(
            dimension_kind.startswith("DIMENSION_")
            and dictionary is not None
            and object_kind(dictionary) == "DICTIONARY"
            and handle_value(dictionary.get("items", {}).get("ACAD_DIMASSOC")) == assoc_handle
            and handle_value(dictionary.get("ownerhandle")) == dimension_handle
            and assoc_handle in {
                handle_value(ref) for ref in dimension.get("reactors", [])
            }
        )
        refs = []
        if decoded:
            for ref_index, ref in enumerate(obj["ref"]):
                if not isinstance(ref, dict) or not ref:
                    continue
                for field in ("xrefs", "intsectobj"):
                    for slot, raw_handle in enumerate(ref.get(field, [])):
                        target_handle = handle_value(raw_handle)
                        target = index.get(target_handle) if target_handle else None
                        target_kind = object_kind(target)
                        target_owner_handle = handle_value(target.get("ownerhandle")) if target else None
                        target_owner = index.get(target_owner_handle) if target_owner_handle else None
                        raw_geometry_ref = bool(
                            dimension_kind.startswith("DIMENSION_")
                            and target_kind in GEOMETRY_TYPES
                        )
                        single_xref = field == "xrefs" and len(ref.get("xrefs", [])) == 1
                        direct_modelspace = bool(
                            raw_geometry_ref and chain_verified and single_xref
                            and dimension.get("entmode") == 2
                            and target.get("entmode") == 2
                        )
                        refs.append({
                            "ref_index": ref_index,
                            "reference_field": field,
                            "reference_slot": slot,
                            "raw_handle": raw_handle,
                            "resolved_handle": hex_handle(target_handle),
                            "target_exists": target is not None,
                            "target_kind": target_kind,
                            "target_entmode": target.get("entmode") if target else None,
                            "target_object_index": target.get("index") if target else None,
                            "target_owner_raw_reference": target.get("ownerhandle") if target else None,
                            "target_owner_handle": hex_handle(target_owner_handle),
                            "target_owner_kind": object_kind(target_owner),
                            "target_owner_name": target_owner.get("name") if target_owner else None,
                            "osnap_type": ref.get("osnap_type"),
                            "main_subent_type": ref.get("main_subent_type"),
                            "native_xref_path_length": len(ref.get("xrefs", [])),
                            "raw_decoded_geometry_reference": raw_geometry_ref,
                            "target_reactor_contains_dimassoc": assoc_handle in {
                                handle_value(item) for item in target.get("reactors", [])
                            } if target else False,
                            "dimension_entmode": dimension.get("entmode") if dimension else None,
                            "confirmed_explicit_geometry_reference": direct_modelspace,
                            "confirmed_direct_modelspace_geometry_reference": direct_modelspace,
                            "dimension_dictionary_chain_verified": chain_verified,
                            "nearby_geometry_used": False,
                            "raw_json_pointer": (
                                f"#/OBJECTS/{array_index}/ref/{ref_index}/{field}/{slot}"
                            ),
                        })
        records.append({
            "dimassoc_handle": hex_handle(assoc_handle),
            "object_index": obj.get("index"),
            "raw_object_type": obj.get("object"),
            "native_payload_decoded": decoded,
            "unknown_bits_present": bool(obj.get("unknown_bits")),
            "unknown_bits_sha256": hashlib.sha256(
                obj.get("unknown_bits", "").encode("ascii")
            ).hexdigest() if obj.get("unknown_bits") else None,
            "dimension_raw_reference": obj.get("dimensionobj"),
            "dimension_handle": hex_handle(dimension_handle),
            "dimension_kind": dimension_kind or None,
            "dimension_entmode": dimension.get("entmode") if dimension else None,
            "dimension_owner_raw_reference": dimension.get("ownerhandle") if dimension else None,
            "dimension_dictionary_chain_verified": chain_verified,
            "associativity": obj.get("associativity"),
            "decoded_ref_blocks": obj.get("ref") if decoded else None,
            "references": refs,
            "raw_json_pointer": f"#/OBJECTS/{array_index}",
        })
    return records


def summarize_native(parsed: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    classes = {obj["number"]: obj.get("dxfname") for obj in parsed.get("CLASSES", [])}
    objects = [dict(obj) for obj in parsed["OBJECTS"]]
    for obj in objects:
        if obj.get("type") in classes:
            obj["_source_class_dxfname"] = classes[obj["type"]]
    assocs = association_records(objects)
    texts = text_records(objects)
    tdmtitle = []
    for array_index, obj in enumerate(objects):
        if "TDMTITLE" in (object_kind(obj) or "").upper():
            tdmtitle.append({
                "handle": hex_handle(handle_value(obj.get("handle"))),
                "kind": object_kind(obj),
                "raw_storage_type": obj.get("object") or obj.get("entity"),
                "opaque_unknown_storage": (
                    obj.get("object") == "UNKNOWN_OBJ"
                    or obj.get("entity") == "UNKNOWN_ENT"
                ),
                "unknown_bits_present": bool(obj.get("unknown_bits")),
                "fields": sorted(obj),
                "raw_json_pointer": f"#/OBJECTS/{array_index}",
            })
    confirmed_refs = [
        ref for assoc in assocs for ref in assoc["references"]
        if ref["confirmed_explicit_geometry_reference"]
    ]
    raw_geometry_refs = [
        ref for assoc in assocs for ref in assoc["references"]
        if ref["raw_decoded_geometry_reference"]
    ]
    summary = {
        "objects": len(objects),
        "handle_array_lengths": dict(sorted(Counter(
            len(obj["handle"]) for obj in objects if isinstance(obj.get("handle"), list)
        ).items())),
        "dimassoc_objects": len(assocs),
        "dimassoc_payloads_decoded": sum(obj["native_payload_decoded"] for obj in assocs),
        "dimassoc_dictionary_chains_verified": sum(
            obj["dimension_dictionary_chain_verified"] for obj in assocs
        ),
        "dimassocs_with_explicit_geometry_target": sum(
            any(ref["confirmed_explicit_geometry_reference"] for ref in obj["references"])
            for obj in assocs
        ),
        "explicit_geometry_reference_edges": len(confirmed_refs),
        "raw_decoded_geometry_reference_edges": len(raw_geometry_refs),
        "raw_geometry_reference_target_types": dict(sorted(Counter(
            ref["target_kind"] for ref in raw_geometry_refs
        ).items())),
        "direct_modelspace_geometry_reference_edges": sum(
            ref["confirmed_direct_modelspace_geometry_reference"] for ref in raw_geometry_refs
        ),
        "distinct_direct_modelspace_dimension_geometry_edges": len({
            (assoc["dimension_handle"], ref["resolved_handle"])
            for assoc in assocs for ref in assoc["references"]
            if ref["confirmed_direct_modelspace_geometry_reference"]
        }),
        "block_or_other_scope_geometry_reference_edges_entmode_0": sum(
            ref["target_entmode"] == 0 for ref in raw_geometry_refs
        ),
        "multiple_xref_path_geometry_reference_slots_not_independent_edges": sum(
            ref["native_xref_path_length"] > 1 for ref in raw_geometry_refs
        ),
        "geometry_reference_target_types": dict(sorted(Counter(
            ref["target_kind"] for ref in confirmed_refs
        ).items())),
        "unique_geometry_targets": len({ref["resolved_handle"] for ref in confirmed_refs}),
        "reference_edges_with_verified_dictionary_chain": sum(
            ref["dimension_dictionary_chain_verified"] for ref in confirmed_refs
        ),
        "reference_edges_with_target_reactor": sum(
            ref["target_reactor_contains_dimassoc"] for ref in confirmed_refs
        ),
        "unresolved_reference_edges": sum(
            not ref["target_exists"] for obj in assocs for ref in obj["references"]
        ),
        "all_decoded_reference_slots": sum(len(obj["references"]) for obj in assocs),
        "all_nonempty_native_ref_blocks": sum(
            bool(ref) for assoc in assocs for ref in (assoc["decoded_ref_blocks"] or [])
        ),
        "all_resolved_reference_target_types": dict(sorted(Counter(
            ref["target_kind"] for obj in assocs for ref in obj["references"]
            if ref["target_exists"]
        ).items())),
        "tdmtitle_objects": len(tdmtitle),
        "tdmtitle_opaque_unknown_objects": sum(obj["opaque_unknown_storage"] for obj in tdmtitle),
        "text_records": len(texts),
        "non_ascii_text_records": sum(obj["non_ascii"] for obj in texts),
        "text_replacement_chars": sum(obj["replacement_chars"] for obj in texts),
    }
    detail = {"dimassoc": assocs, "tdmtitle": tdmtitle, "texts": texts}
    return summary, detail


def compare_texts(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> dict[str, Any]:
    old_by_key = {(obj["handle"], obj["field"]): obj["value"] for obj in old}
    new_by_key = {(obj["handle"], obj["field"]): obj["value"] for obj in new}
    shared = set(old_by_key) & set(new_by_key)
    changed = [{"handle": key[0], "field": key[1], "old": old_by_key[key], "new": new_by_key[key]}
               for key in sorted(shared) if old_by_key[key] != new_by_key[key]]
    return {
        "shared_handle_field_pairs": len(shared),
        "identical_pairs": len(shared) - len(changed),
        "changed_pairs": changed,
        "old_only_pairs": len(set(old_by_key) - set(new_by_key)),
        "new_only_pairs": len(set(new_by_key) - set(old_by_key)),
    }


def reader_resource_limits() -> None:
    """Bound the isolated child process without changing parent resources."""
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(allowed[:2]))


def run_reader(binary: Path, source: Path, label: str, case_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stdout_path = case_dir / f"{label}.response.json"
    stderr_path = case_dir / f"{label}.response.stderr.txt"
    command = [str(binary), "-O", "JSON", str(source)]
    start = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", LC_ALL="C.UTF-8")
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=60, env=env, check=False, preexec_fn=reader_resource_limits)
        exit_code: int | None = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        exit_code = None
        timed_out = True
    parsed, json_status = load_response(stdout_path.read_bytes())
    status = {
        "command": command,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "stdout_path": str(stdout_path),
        "stdout_sha256": digest(stdout_path),
        "stdout_bytes": stdout_path.stat().st_size,
        "stderr_path": str(stderr_path),
        "stderr_sha256": digest(stderr_path),
        "json": json_status,
    }
    detail: dict[str, Any] = {}
    if parsed is not None:
        status["native"], detail = summarize_native(parsed)
        write_new_json(case_dir / f"{label}.native_details.json", detail)
    write_new_json(case_dir / f"{label}.execution.json", status)
    return status, detail


def check_serialization_fix(baseline_root: Path, output_root: Path, binary: Path) -> dict[str, Any]:
    """Read patched responses once; compare them with sealed baseline bytes.

    Surrogateescape is used only to inspect already sealed invalid binary
    strings. It never makes the original response pass the UTF-8/JSON gate.
    """
    baseline = json.loads((baseline_root / "comparison.json").read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    cases = []
    for original in baseline["cases"]:
        code = original["item_code"]
        source = Path(original["source_path"])
        before = digest(source)
        if before != original["source_sha256_before"]:
            raise ValueError(f"Original source pin changed: {code}")
        case_dir = output_root / "cases" / code
        case_dir.mkdir(parents=True, exist_ok=True)
        status, detail = run_reader(binary.resolve(), source, "patched", case_dir)
        strict_gate = bool(status["json"].get("utf8_valid") and status["json"].get("json_success"))
        row: dict[str, Any] = {
            "item_code": code, "source_sha256_before": before,
            "source_sha256_after": digest(source), "patched": status,
            "strict_utf8_json_pass": strict_gate,
            "baseline_invalid_response_not_relabelled_success": True,
        }
        if strict_gate:
            patched, _ = load_response(Path(status["stdout_path"]).read_bytes())
            unpatched_bytes = Path(original["new"]["stdout_path"]).read_bytes()
            unpatched = json.loads(unpatched_bytes.decode("utf-8-sig", errors="surrogateescape"))
            old, old_status = load_response(Path(original["old"]["stdout_path"]).read_bytes())
            if patched is None or old is None:
                raise ValueError(f"Comparison inputs unavailable: {code}")
            baseline_summary, baseline_detail = summarize_native(unpatched)
            old_summary, old_detail = summarize_native(old)
            patched_index = native_index(patched["OBJECTS"])
            proxy_checks = []
            clean_baseline_objects = []
            clean_patched_objects = []
            for obj in unpatched["OBJECTS"]:
                copied = dict(obj)
                if "proxy_data" in copied:
                    raw_value = copied.pop("proxy_data")
                    raw_payload = raw_value.encode("utf-8", errors="surrogateescape")
                    handle = handle_value(obj.get("handle"))
                    counterpart = patched_index[handle]
                    hex_payload = counterpart["proxy_data"]
                    decoded_hex = bytes.fromhex(hex_payload)
                    proxy_checks.append({
                        "handle": hex_handle(handle), "size": obj.get("proxy_data_size"),
                        "baseline_raw_byte_hex": raw_payload.hex().upper(),
                        "patched_hex": hex_payload,
                        "roundtrip_bytes_equal": raw_payload == decoded_hex,
                        "length_matches_native_size": len(decoded_hex) == obj.get("proxy_data_size"),
                    })
                clean_baseline_objects.append(copied)
            for obj in patched["OBJECTS"]:
                copied = dict(obj)
                copied.pop("proxy_data", None)
                clean_patched_objects.append(copied)
            unicode_comparison = compare_texts(old_detail["texts"], detail["texts"])
            mtext_comparison = compare_texts(
                [obj for obj in old_detail["texts"] if obj["kind"] == "MTEXT" and obj["field"] == "text"],
                [obj for obj in detail["texts"] if obj["kind"] == "MTEXT" and obj["field"] == "text"],
            )
            row.update({
                "baseline_native_reviewed_summary": baseline_summary,
                "old_native_reviewed_summary": old_summary,
                "old_response_lossless_status": old_status,
                "engineering_text_comparison": unicode_comparison,
                "mtext_full_body_comparison": mtext_comparison,
                "proxy_data_roundtrip": proxy_checks,
                "all_objects_equal_except_proxy_data_serialization": clean_baseline_objects == clean_patched_objects,
                "header_and_classes_identical": unpatched["HEADER"] == patched["HEADER"] and unpatched["CLASSES"] == patched["CLASSES"],
                "dimassoc_reference_details_identical": baseline_detail["dimassoc"] == detail["dimassoc"],
            })
            row["all_checks_pass"] = bool(
                row["source_sha256_after"] == before
                and not unicode_comparison["changed_pairs"]
                and unicode_comparison["old_only_pairs"] == unicode_comparison["new_only_pairs"] == 0
                and row["all_objects_equal_except_proxy_data_serialization"]
                and row["header_and_classes_identical"]
                and row["dimassoc_reference_details_identical"]
                and all(item["roundtrip_bytes_equal"] and item["length_matches_native_size"] for item in proxy_checks)
            )
        else:
            row["all_checks_pass"] = False
        write_new_json(case_dir / "comparison.json", row)
        cases.append(row)
        print(json.dumps({"case": code, "all_checks_pass": row["all_checks_pass"], "native": status.get("native")}, ensure_ascii=False), flush=True)
    result = {
        "schema_version": "explicit_native_paths_v2",
        "baseline_comparison_sha256": digest(baseline_root / "comparison.json"),
        "frozen_binary_serialization_protocol_sha256": digest(baseline_root / "BINARY_SERIALIZATION_PROTOCOL.json"),
        "baseline_executed_script_sha256": digest(baseline_root / "tools/parser_probe.executed.py"),
        "review_script_sha256": digest(Path(__file__)),
        "patched_binary_sha256": digest(binary),
        "source_commit": SOURCE_COMMIT,
        "patch_scope": "IS_JSON proxy_data binary serialization only",
        "all_fixed_three_pass": all(obj["all_checks_pass"] for obj in cases),
        "cases": cases,
        "geometry_proximity_used": False,
        "gt_or_val_test_read": False,
        "pdf_version_pairing_gate_relaxed": False,
    }
    write_new_json(output_root / "comparison.json", result)
    return result


def probe_frozen_selection(protocol_path: Path, binary: Path) -> dict[str, Any]:
    """Read only the new binary, honoring the previously frozen source list."""
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output_root = protocol_path.parent
    start = time.monotonic()
    response_bytes = 0
    cases = []
    for sample in protocol["sources"]:
        if time.monotonic() - start >= 600 or response_bytes >= 512 * 1024 * 1024:
            cases.append({"item_code": sample["item_code"], "status": "not_run_budget_exhausted"})
            continue
        code = sample["item_code"]
        source = Path(sample["path"])
        before = digest(source)
        if before != sample["source_sha256"]:
            cases.append({"item_code": code, "status": "source_pin_mismatch", "actual_sha256": before})
            continue
        case_dir = output_root / "cases" / code
        case_dir.mkdir(parents=True, exist_ok=True)
        status, _ = run_reader(binary.resolve(), source, "patched", case_dir)
        response_bytes += status["stdout_bytes"]
        after = digest(source)
        row = {
            "item_code": code, "source_sha256_before": before,
            "source_sha256_after": after, "source_hash_unchanged": before == after,
            "execution": status,
            "strict_utf8_json_pass": bool(status["json"].get("utf8_valid") and status["json"].get("json_success")),
            "original_fixed_source_not_replaced": True,
        }
        write_new_json(case_dir / "result.json", row)
        cases.append(row)
        print(json.dumps({"case": code, "strict_utf8_json_pass": row["strict_utf8_json_pass"], "exit_code": status["exit_code"], "native": status.get("native")}, ensure_ascii=False), flush=True)
    result = {
        "protocol_sha256": digest(protocol_path), "script_sha256": digest(Path(__file__)),
        "binary_sha256": digest(binary), "elapsed_seconds": round(time.monotonic() - start, 3),
        "total_stdout_bytes": response_bytes, "cases": cases,
        "strict_utf8_json_successes": sum(obj.get("strict_utf8_json_pass", False) for obj in cases),
        "fixed_source_count": len(protocol["sources"]), "old_reader_reexecuted": False,
        "gt_or_val_test_read": False, "pdf_version_pairing_gate_relaxed": False,
        "route_training_performed": False,
    }
    write_new_json(output_root / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "serialization-check", "fixed14"), default="baseline")
    parser.add_argument("--old-reader", type=Path, default=Path("/home/userroot/miniconda3/envs/dwg-free/bin/dwgread"))
    parser.add_argument("--new-reader", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=REPO / "outputs/dwg_route_validation_20261015/parser_comparison_001")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--selection", type=Path, default=REPO / "outputs/dwg_dataset_audit_20261015/sample_probe_001/selection.json")
    parser.add_argument("--output", type=Path, default=REPO / "outputs/dwg_route_validation_20261015/parser_comparison_001")
    args = parser.parse_args()
    if args.mode == "serialization-check":
        check_serialization_fix(args.baseline_root, args.output, args.new_reader)
        return
    if args.mode == "fixed14":
        if args.protocol is None:
            parser.error("--protocol is required for fixed14")
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        if digest(args.new_reader) != protocol["patched_binary_sha256"]:
            raise ValueError("Reader does not match the frozen protocol binary SHA256")
        args.output.mkdir(parents=True, exist_ok=False)
        copied_protocol = args.output / "PROTOCOL.json"
        with copied_protocol.open("xb") as stream:
            stream.write(args.protocol.read_bytes())
        probe_frozen_selection(copied_protocol, args.new_reader)
        return
    if (args.output / "comparison.json").exists():
        raise FileExistsError("Refusing to overwrite sealed comparison.json; use a new output directory")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    samples = {obj["item_code"]: obj for obj in selection["samples"] if obj["item_code"] in FIXED_CODES}
    readers = {}
    for label, binary in (("old", args.old_reader.resolve()), ("new", args.new_reader.resolve())):
        version = subprocess.run([str(binary), "--version"], capture_output=True, timeout=10, check=False)
        readers[label] = {"path": str(binary), "sha256": digest(binary), "version_stdout": version.stdout.decode("utf-8", errors="strict").strip(), "version_exit_code": version.returncode}
    cases = []
    for code in FIXED_CODES:
        sample = samples[code]
        source = Path(sample["path"])
        before = digest(source)
        if before != sample["source_sha256"]:
            raise ValueError(f"Source pin mismatch: {code}")
        case_dir = args.output / "cases" / code
        case_dir.mkdir(parents=True, exist_ok=True)
        old_status, old_detail = run_reader(args.old_reader.resolve(), source, "old", case_dir)
        after_old = digest(source)
        new_status, new_detail = run_reader(args.new_reader.resolve(), source, "new", case_dir)
        after_new = digest(source)
        row = {"item_code": code, "source_path": str(source), "source_sha256_before": before, "source_sha256_after_old": after_old, "source_sha256_after_new": after_new, "source_hash_unchanged": before == after_old == after_new, "old": old_status, "new": new_status}
        if old_detail and new_detail:
            row["unicode_text_comparison"] = compare_texts(old_detail["texts"], new_detail["texts"])
            row["mtext_full_body_comparison"] = compare_texts(
                [obj for obj in old_detail["texts"] if obj["kind"] == "MTEXT" and obj["field"] == "text"],
                [obj for obj in new_detail["texts"] if obj["kind"] == "MTEXT" and obj["field"] == "text"],
            )
        write_new_json(case_dir / "comparison.json", row)
        cases.append(row)
        print(json.dumps({"case": code, "source_hash_unchanged": row["source_hash_unchanged"], "old": old_status.get("native"), "new": new_status.get("native")}, ensure_ascii=False), flush=True)
    report = {"source_commit": SOURCE_COMMIT, "script_sha256": digest(Path(__file__)), "selection_sha256": digest(args.selection), "readers": readers, "cases": cases, "route_labels_read": False, "gt_read": False, "val_test_read": False, "model_outputs_read": False, "geometry_proximity_used": False, "unresolved_targets_are_not_inferred": True}
    write_new_json(args.output / "comparison.json", report)


if __name__ == "__main__":
    main()
