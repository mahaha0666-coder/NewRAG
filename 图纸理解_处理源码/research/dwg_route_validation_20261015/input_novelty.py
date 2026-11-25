"""Compare native MTEXT to actual frozen Train input; never consumes route labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from ezdxf.tools.text import plain_mtext

from research.dwg_dataset_audit_20261015.audit import handle
from src.tools.drawing_extraction.workflow import _load_libredwg_json
from src.tools.research.artifacts import digest, read_json, write_new_json


REPO = Path(__file__).resolve().parents[2]
SOURCES = REPO / "outputs/dwg_dataset_audit_20261015/sample_probe_001"
TRAIN_INPUTS = REPO / "outputs/repeated_pdf_scope_20261006/r6_train_001/features/training_inputs.jsonl"
PROCESS_WORDS = ("焊", "钻", "孔", "配作", "精加工", "机加工", "铣", "镗", "车削", "拉丝", "抛光", "喷塑")
FORBIDDEN = {"operations", "route", "routes", "label", "labels", "gt", "gt_route", "candidates", "predictions"}


def compact(text: str) -> str:
    """Ignore whitespace only; codes, decimal points and engineering signs stay exact."""
    return re.sub(r"\s+", "", text)


def paragraphs(raw: str) -> list[str]:
    return [line.strip() for line in plain_mtext(raw).splitlines() if line.strip()]


def compare_literal(text: str, row: dict[str, Any]) -> dict[str, Any]:
    if FORBIDDEN.intersection(row) or row.get("split") != "train":
        raise ValueError("Only label-free Train input is permitted")
    base = compact(str(row.get("base_text", "")))
    evidence = compact(str(row.get("evidence_text", "")))
    literal = compact(text)
    if not literal:
        raise ValueError("Empty source text is not evidence")
    return {
        "literal": text,
        "present_in_base_text": literal in base,
        "present_in_evidence_text": literal in evidence,
        "present_in_any_effective_text": literal in base or literal in evidence,
        "absence_is_only_review_candidate": True,
    }


def run() -> dict[str, Any]:
    selected = read_json(SOURCES / "selection.json")
    previous = read_json(SOURCES / "results.json")
    if (len(selected["samples"]) != 14 or selected["route_labels_read"] is not False
            or selected["val_test_content_read"] is not False):
        raise ValueError("Use the unchanged fixed Train source selection")
    records = {row["item_code"]: row for row in previous["records"]}
    rows = [json.loads(line) for line in TRAIN_INPUTS.read_text().splitlines() if line.strip()]
    if len(rows) != 313 or any(FORBIDDEN.intersection(row) or row.get("split") != "train" for row in rows):
        raise ValueError("Frozen label-free 313 Train inputs required")
    lookup = {row["item_code"]: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate Train identity")
    pins = {str(path.relative_to(REPO)): digest(path) for path in
            (SOURCES / "selection.json", SOURCES / "results.json", TRAIN_INPUTS, Path(__file__))}
    cases = []
    for source in selected["samples"]:
        code = source["item_code"]
        original = Path(source["path"])
        if digest(original) != source["source_sha256"]:
            raise ValueError("Original DWG changed")
        pins[str(original.relative_to(REPO))] = source["source_sha256"]
        record = records[code]
        case: dict[str, Any] = {
            "item_code": code, "dwg_source_sha256": source["source_sha256"],
            "in_frozen_route_Train_inputs": code in lookup,
            "pair_version_verified": False, "eligible_for_training": False,
            "lines": [],
        }
        if "native" not in record:
            case["status"] = "native_source_unreadable"
            cases.append(case)
            continue
        native = SOURCES / code / "fresh.native.json"
        if digest(native) != record["native_sha256"]:
            raise ValueError("Fresh native extraction changed")
        pins[str(native.relative_to(REPO))] = record["native_sha256"]
        row = lookup.get(code)
        if row is None:
            case["status"] = "outside_frozen_route_population"
            cases.append(case)
            continue
        case.update(status="diagnostic_only", group_id=row["group_id"], pdf_source_sha256=row["source_sha256"])
        for index, obj in enumerate(_load_libredwg_json(native)["OBJECTS"]):
            if obj.get("entity") != "MTEXT" or obj.get("entmode") != 2:
                continue
            raw = obj.get("text")
            if not isinstance(raw, str):
                continue
            identity = handle(obj.get("handle"))
            for line in paragraphs(raw):
                if not any(word in line for word in PROCESS_WORDS):
                    continue
                case["lines"].append({
                    **compare_literal(line, row), "native_handle": format(identity, "X") if identity else None,
                    "native_ref": f"{native.relative_to(REPO)}#/OBJECTS/{index}/text",
                    "raw_mtext": raw, "native_insertion_point": obj.get("ins_pt"),
                })
        cases.append(case)
    lines = [line for case in cases for line in case["lines"]]
    return {
        "schema_version": "native_mtext_frozen_input_novelty.v1", "source_pins": pins,
        "cases": cases,
        "summary": {
            "fixed_sources": len(cases), "route_population_sources": sum(case["in_frozen_route_Train_inputs"] for case in cases),
            "native_unreadable": sum(case["status"] == "native_source_unreadable" for case in cases),
            "compared_native_paragraphs": len(lines),
            "exact_whitespace_insensitive_presence": sum(line["present_in_any_effective_text"] for line in lines),
            "absent_literal_review_candidates": sum(not line["present_in_any_effective_text"] for line in lines),
        },
        "absence_is_not_verified_omission": True, "ocr_calls": 0, "GPU_used": False,
        "route_labels_read": False, "val_test_read": False, "route_gain_measured": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    result = run()
    if args.replay:
        expected = read_json(args.output)
        if result != expected:
            raise ValueError("Independent native/input comparison differs")
        print(json.dumps({"exact_result_equal": True, "result_sha256": digest(args.output)}))
    else:
        write_new_json(args.output, result)
        print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
