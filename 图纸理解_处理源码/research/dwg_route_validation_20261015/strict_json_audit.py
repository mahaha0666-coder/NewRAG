"""Independently check the sealed fixed-14 native JSON responses and hashes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from research.dwg_route_validation_20261015.input_novelty import REPO, SOURCES
from src.tools.research.artifacts import digest, read_json, write_new_json


def reject_constant(token: str) -> None:
    raise ValueError(f"Non-JSON numeric constant: {token}")


def finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("Numeric value exceeds finite float range")
    return value


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def retain_class_markers(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Preserve LibreDWG's repeated inheritance markers; reject other duplicates."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key == "_subclass":
            if not isinstance(value, str):
                raise ValueError("Native subclass marker must be a string")
            result.setdefault(key, []).append(value)
        else:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
    return result


def strict_load(raw: bytes, *, class_markers_as_sequence: bool = False) -> dict[str, Any]:
    value = json.loads(
        raw.decode("utf-8"),
        parse_constant=reject_constant,
        parse_float=finite_float,
        object_pairs_hook=retain_class_markers if class_markers_as_sequence else unique_object,
    )
    if not isinstance(value, dict) or not isinstance(value.get("OBJECTS"), list):
        raise ValueError("Native OBJECTS list is required")
    return value


def run(summary_path: Path, *, class_markers_as_sequence: bool = False) -> dict[str, Any]:
    selected = read_json(SOURCES / "selection.json")
    expected = {row["item_code"]: row for row in selected["samples"]}
    summary = read_json(summary_path)
    cases = summary["cases"]
    if (len(cases) != 14 or len(expected) != 14
            or {row["item_code"] for row in cases} != set(expected)):
        raise ValueError("All unchanged fixed-14 sources must be retained")
    records = []
    for case in cases:
        execution = case["execution"]
        original = Path(expected[case["item_code"]]["path"])
        if digest(original) != expected[case["item_code"]]["source_sha256"]:
            raise ValueError("Original DWG hash changed")
        response = REPO / execution["stdout_path"]
        response_hash = digest(response)
        if response_hash != execution["stdout_sha256"]:
            raise ValueError("Sealed raw response changed")
        record = {
            "item_code": case["item_code"],
            "source_sha256": digest(original),
            "response_path": str(response),
            "response_sha256": response_hash,
            "strict_utf8_json_pass": False,
        }
        try:
            parsed = strict_load(response.read_bytes(), class_markers_as_sequence=class_markers_as_sequence)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            record["failure"] = str(error)
        else:
            objects = parsed["OBJECTS"]
            record.update(strict_utf8_json_pass=True, native_objects=len(objects))
            if class_markers_as_sequence:
                record["objects_with_multiple_subclass_markers_retained"] = sum(
                    len(obj.get("_subclass", [])) > 1 for obj in objects
                )
            del parsed
        records.append(record)
    return {
        "schema_version": "fixed_native_strict_json_audit.v2" if class_markers_as_sequence else "fixed_native_strict_json_audit.v1",
        "source_pins": {
            str(summary_path.resolve()): digest(summary_path),
            str(SOURCES / "selection.json"): digest(SOURCES / "selection.json"),
            str(Path(__file__).resolve()): digest(Path(__file__)),
        },
        "checks": ["strict UTF-8", "no NaN/Infinity", "finite float range", "raw response hash", "original DWG hash"],
        "duplicate_key_policy": (
            "Only _subclass string markers become an ordered list; reject all other duplicate keys"
            if class_markers_as_sequence else "Reject all duplicate keys"
        ),
        "original_raw_json_has_duplicate_subclass_keys": class_markers_as_sequence,
        "original_response_modified": False,
        "pipeline_input_modified": False,
        "records": records,
        "summary": {
            "fixed_sources": len(records),
            "strict_success": sum(row["strict_utf8_json_pass"] for row in records),
            "failed": sum(not row["strict_utf8_json_pass"] for row in records),
        },
        "route_labels_read": False,
        "val_test_read": False,
        "source_pairing_certified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--retain-class-markers", action="store_true")
    args = parser.parse_args()
    result = run(args.summary, class_markers_as_sequence=args.retain_class_markers)
    if args.replay:
        if read_json(args.output) != result:
            raise ValueError("Independent strict JSON audit differs")
        print(json.dumps({"exact_result_equal": True, "result_sha256": digest(args.output)}))
    else:
        write_new_json(args.output, result)
        print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
