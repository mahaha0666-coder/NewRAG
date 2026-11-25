"""Audit every JSON key occurrence before confirming native reference counts.

All duplicate values are retained in a private diagnostic sidecar. Only
identical occurrences can yield one value; conflicting keys are omitted from
the reference consumer. This is not a replacement for the production loader.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from research.dwg_route_validation_20261015.input_novelty import REPO
from research.dwg_route_validation_20261015.parser_probe import summarize_native
from research.dwg_route_validation_20261015.strict_json_audit import finite_float, reject_constant
from src.tools.research.artifacts import digest, read_json, write_new_json


SIDECAR = "__native_duplicate_occurrences__"
METRICS = (
    "dimassoc_payloads_decoded", "direct_modelspace_geometry_reference_edges",
    "distinct_direct_modelspace_dimension_geometry_edges",
    "reference_edges_with_verified_dictionary_chain", "unresolved_reference_edges",
)


def lossless_projection(raw: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []

    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[Any]] = {}
        for key, value in pairs:
            if key == SIDECAR:
                raise ValueError("Source collides with diagnostic sidecar key")
            grouped.setdefault(key, []).append(value)
        result: dict[str, Any] = {}
        repeated = {}
        for key, values in grouped.items():
            if len(values) == 1:
                result[key] = values[0]
                continue
            repeated[key] = values
            encodings = {json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) for value in values}
            identical = len(encodings) == 1
            events.append({"key": key, "occurrences": len(values), "identical": identical})
            if identical:
                result[key] = values[0]
        if repeated:
            result[SIDECAR] = repeated
        return result

    parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=object_hook,
                        parse_constant=reject_constant, parse_float=finite_float)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("OBJECTS"), list):
        raise ValueError("Native OBJECTS must be unambiguous")
    return parsed, events


def run(summary_path: Path) -> dict[str, Any]:
    previous = read_json(summary_path)
    if len(previous["cases"]) != 14:
        raise ValueError("All fixed14 sources are required")
    records = []
    for case in previous["cases"]:
        response = REPO / case["execution"]["stdout_path"]
        if digest(response) != case["execution"]["stdout_sha256"]:
            raise ValueError("Frozen raw response changed")
        parsed, events = lossless_projection(response.read_bytes())
        current, _ = summarize_native(parsed)
        expected = case["execution"]["native"]
        observed = {key: current[key] for key in METRICS}
        equal = all(observed[key] == expected[key] for key in METRICS)
        records.append({
            "item_code": case["item_code"], "response_sha256": digest(response),
            "duplicate_field_groups": len(events),
            "duplicate_keys": dict(sorted(Counter(event["key"] for event in events).items())),
            "conflicting_duplicate_keys": dict(sorted(Counter(event["key"] for event in events if not event["identical"]).items())),
            "reference_metrics_without_last_value_overwrite": observed,
            "reference_counts_equal_to_previous_projection": equal,
        })
    return {
        "schema_version": "native_duplicate_occurrence_audit.v1",
        "source_pins": {
            str(summary_path.resolve()): digest(summary_path),
            str(Path(__file__).resolve()): digest(Path(__file__)),
            str(Path(__file__).with_name("parser_probe.py")): digest(Path(__file__).with_name("parser_probe.py")),
        },
        "duplicate_policy": "retain all occurrences; omit conflicting field from reference projection; identical occurrences explicitly recorded",
        "records": records,
        "summary": {
            "fixed_sources": len(records),
            "reference_count_reproduction_pass": sum(row["reference_counts_equal_to_previous_projection"] for row in records),
            "direct_reference_slots": sum(row["reference_metrics_without_last_value_overwrite"]["direct_modelspace_geometry_reference_edges"] for row in records),
            "distinct_dimension_geometry_edges": sum(row["reference_metrics_without_last_value_overwrite"]["distinct_direct_modelspace_dimension_geometry_edges"] for row in records),
        },
        "original_response_modified": False, "model_input_modified": False,
        "source_pairing_gate_relaxed": False, "route_labels_read": False, "val_test_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    result = run(args.summary)
    if args.replay:
        if read_json(args.output) != result:
            raise ValueError("Independent duplicate occurrence audit differs")
        print(json.dumps({"exact_result_equal": True, "result_sha256": digest(args.output)}))
    else:
        write_new_json(args.output, result)
        print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
