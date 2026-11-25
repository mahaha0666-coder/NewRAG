"""Source-bound native DWG evidence, without manufacturing or route inference.

Only decoded, direct modelspace references become structural edges. CAD units,
physical ownership and PDF registration remain separate unresolved facts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from research.dwg_route_validation_20261015.duplicate_audit import lossless_projection
from research.dwg_route_validation_20261015.parser_probe import (
    GEOMETRY_TYPES, handle_value, hex_handle, object_kind, summarize_native,
)
from src.tools.research.artifacts import digest, read_json, write_new_json

REPO = Path(__file__).resolve().parents[2]
COORDINATES = (
    "text_midpt", "def_pt", "xline1_pt", "xline2_pt", "center", "start", "end",
    "radius", "start_angle", "end_angle", "extrusion", "elevation",
    "text_rotation", "dim_rotation", "ins_scale", "ins_rotation",
)


def build_ir(parsed: dict[str, Any], *, item_code: str, source_sha256: str,
             response_sha256: str) -> dict[str, Any]:
    """Retain exact source values and references; do not synthesize dimension text."""
    for value in (source_sha256, response_sha256):
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("Original source and response SHA256 are required")
    _, detail = summarize_native(parsed)
    dimensions, geometry = [], []
    evidence_by_handle = {}
    for position, obj in enumerate(parsed["OBJECTS"]):
        kind = object_kind(obj) or ""
        identity = hex_handle(handle_value(obj.get("handle")))
        if identity is None:
            continue
        evidence = {
            "source_sha256": source_sha256, "response_sha256": response_sha256,
            "json_pointer": f"#/OBJECTS/{position}", "native_handle": identity,
        }
        evidence_by_handle[identity] = evidence
        if obj.get("entmode") != 2:
            continue
        coordinates = {key: deepcopy(obj[key]) for key in COORDINATES if key in obj}
        node = {
            "id": identity, "native_kind": kind, "cad_coordinates": coordinates,
            "native_entmode": obj["entmode"], "source_ref": evidence,
            "physical_owner": "unknown", "confidence": None,
            "native_visibility": obj.get("invisible"),
        }
        if kind.startswith("DIMENSION_"):
            literal = obj.get("user_text", "")
            if not isinstance(literal, str):
                raise ValueError("Native dimension user_text must remain a string")
            dimensions.append(dict(node, raw_text=literal,
                                   raw_measurement=obj.get("act_measurement"),
                                   unit="unknown", unit_source=None))
        elif kind in GEOMETRY_TYPES:
            geometry.append(node)
    dimensions.sort(key=lambda node: node["id"])
    geometry.sort(key=lambda node: node["id"])
    dim_ids = {node["id"] for node in dimensions}
    geo_ids = {node["id"] for node in geometry}
    grouped = {}
    unresolved = []
    for assoc in detail["dimassoc"]:
        for ref in assoc["references"]:
            evidence = dict(ref, source_sha256=source_sha256,
                            response_sha256=response_sha256,
                            dimassoc_handle=assoc["dimassoc_handle"],
                            dimension_source=evidence_by_handle.get(assoc["dimension_handle"]),
                            geometry_source=evidence_by_handle.get(ref["resolved_handle"]))
            pair = (assoc["dimension_handle"], ref["resolved_handle"])
            if (not ref["confirmed_direct_modelspace_geometry_reference"]
                    or pair[0] not in dim_ids or pair[1] not in geo_ids):
                unresolved.append(evidence)
                continue
            grouped.setdefault(pair, []).append(evidence)
    relations = [
        {"dimension_id": pair[0], "geometry_id": pair[1],
         "status": "verified_direct_modelspace", "dictionary_chain_verified": True,
         "dimension_entmode": 2, "target_entmode": 2,
         "native_xref_path_length": 1, "reference_field": "xrefs",
         "physical_scope": "unknown", "relationship_confidence": None,
         "source_refs": refs}
        for pair, refs in sorted(grouped.items())
    ]
    # Ordering follows original native object/field order. Raw formatting remains
    # part of the evidence, rather than being decoded into guessed PMI fields.
    text_evidence = [dict(row, source_sha256=source_sha256,
                          response_sha256=response_sha256,
                          source_ref=evidence_by_handle.get(row["handle"]))
                     for row in detail["texts"]]
    return {
        "schema_version": "native_dwg_evidence_ir.v1", "item_code": item_code,
        "source_kind": "native_dwg", "source_sha256": source_sha256,
        "response_sha256": response_sha256, "units": "unknown", "scope": "unknown",
        "raw_insunits": parsed.get("HEADER", {}).get("INSUNITS"),
        "insunits_is_not_confirmed_dimension_unit": True,
        "eligible_for_route_training": False,
        "source_pairing": {"same_version": "unknown", "pdf_registration": "unknown"},
        "original_text": "\n".join(row["value"] for row in text_evidence),
        "text_evidence": text_evidence, "dimensions": dimensions,
        "geometry_nodes": geometry, "relations": relations,
        "unresolved_references": unresolved,
        "quality": {"dimensions": len(dimensions), "geometry_nodes": len(geometry),
                    "distinct_verified_direct_edges": len(relations),
                    "unresolved_reference_slots": len(unresolved),
                    "geometry_kinds": dict(sorted(Counter(node["native_kind"] for node in geometry).items()))},
        "capability_limits": ["not_physical_part_ownership", "not_hole_detection",
                              "not_process_evidence", "not_registered_to_pdf"],
    }


def run(summary_path: Path, output: Path, *, replay: bool = False) -> dict[str, Any]:
    summary = read_json(summary_path)
    records = []
    if len(summary["cases"]) != 14:
        raise ValueError("The complete fixed14 source set is required")
    if not replay:
        output.mkdir(parents=True, exist_ok=False)
    for case in summary["cases"]:
        code = case["item_code"]
        response = REPO / case["execution"]["stdout_path"]
        original = REPO / "data/frozen/route_drawing_ready_20260909/assets/dwg" / f"{code}.dwg"
        if digest(response) != case["execution"]["stdout_sha256"]:
            raise ValueError(f"Native response changed: {code}")
        source_hash = digest(original)
        expected_hash = case.get("source_sha256_before")
        if expected_hash is None:
            raise ValueError("A pinned original hash is required")
        if source_hash != expected_hash or source_hash != case.get("source_sha256_after"):
            raise ValueError(f"Original DWG changed: {code}")
        parsed, events = lossless_projection(response.read_bytes())
        payload = build_ir(parsed, item_code=code, source_sha256=source_hash,
                           response_sha256=digest(response))
        destination = output / f"{code}.json"
        if replay:
            if read_json(destination) != payload:
                raise ValueError(f"Independent IR replay differs: {code}")
        else:
            write_new_json(destination, payload)
        records.append({"item_code": code, "path": str(destination.resolve()),
                        "sha256": digest(destination), "quality": payload["quality"],
                        "duplicate_field_groups_preserved": len(events)})
    result = {"schema_version": "fixed14_native_ir_receipt.v1",
              "input_summary_sha256": digest(summary_path),
              "adapter_sha256": digest(Path(__file__)), "records": records,
              "route_labels_read": False, "val_test_read": False,
              "route_training_admitted": False,
              "distinct_direct_edges": sum(row["quality"]["distinct_verified_direct_edges"] for row in records)}
    if replay:
        if read_json(output / "summary.json") != result:
            raise ValueError("Independent IR receipt differs")
    else:
        write_new_json(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    result = run(args.summary, args.output, replay=args.replay)
    print(json.dumps({"fixed_sources": len(result["records"]),
                      "distinct_direct_edges": result["distinct_direct_edges"],
                      "route_training_admitted": False, "replay": args.replay}))


if __name__ == "__main__":
    main()
