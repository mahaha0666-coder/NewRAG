"""Read literal display text through a dimension's anonymous native block.

Only explicit resolved handles and directly owned visible TEXT/MTEXT members
are accepted. Measurement and style values never synthesize display text.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from typing import Any

from research.dwg_route_validation_20261015.duplicate_audit import SIDECAR, lossless_projection
from research.dwg_route_validation_20261015.parser_probe import (
    REPO, digest, handle_value, hex_handle, native_index, object_kind, write_new_json,
)

COORDINATES = ("ins_pt", "insertion_pt", "alignment_pt", "extrusion", "x_axis_dir",
               "rotation", "elevation", "text_midpt", "text_rotation", "ins_scale", "ins_rotation")
CODE_FILES = ("native_dimension_display.py", "test_native_dimension_display.py",
              "duplicate_audit.py", "parser_probe.py", "strict_json_audit.py", "input_novelty.py")


def strict_handle(value: Any, *, reference: bool, origin: int | None = None,
                  allow_null: bool = False) -> int | None:
    """Require native object triples or resolved reference quadruples."""
    if not isinstance(value, list) or len(value) != (4 if reference else 3):
        return None
    if any(type(slot) is not int or slot < 0 for slot in value):
        return None
    code, size, encoded = value[:3]
    if size > 8 or encoded >= 1 << (size * 8):
        return None
    if not reference:
        return handle_value(value) if code == 0 else None
    resolved = value[3]
    if code in {2, 3, 4, 5}:
        consistent = encoded == resolved
    elif code in {6, 8}:
        consistent = (origin is not None and size == 0 and encoded == 0
                      and resolved == origin + (1 if code == 6 else -1))
    elif code in {10, 12}:
        consistent = origin is not None and resolved == origin + (encoded if code == 10 else -encoded)
    else:
        consistent = False
    if not consistent or (resolved == 0 and not allow_null):
        return None
    return 0 if resolved == 0 else handle_value(value)


def conflicts(obj: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    repeated = obj.get(SIDECAR, {})
    return [field for field in fields if field in repeated and field not in obj]


def coordinate_evidence(obj: dict[str, Any], pointer: str) -> dict[str, Any]:
    return {field: {"raw_value": obj[field], "source_pointer": f"{pointer}/{field}"}
            for field in COORDINATES if field in obj}


def extract_dimension_displays(raw: bytes, dimension_handles: list[str]) -> dict[str, Any]:
    parsed, duplicates = lossless_projection(raw)
    objects = parsed["OBJECTS"]
    if any(not isinstance(obj, dict) for obj in objects):
        raise ValueError("Native OBJECTS contains a non-object")
    valid = [obj for obj in objects if strict_handle(obj.get("handle"), reference=False) is not None]
    # native_index rejects repeated object identities, including identical ones.
    index = native_index(valid)
    pointers = {strict_handle(obj.get("handle"), reference=False): f"#/OBJECTS/{slot}"
                for slot, obj in enumerate(objects)
                if strict_handle(obj.get("handle"), reference=False) is not None}
    records = []
    for handle in dimension_handles:
        value = int(handle, 16)
        dimension = index.get(value)
        row: dict[str, Any] = {
            "dimension_handle": hex_handle(value), "status": "unknown", "reasons": [],
            "display_text": None, "visible_text_members": [], "block_members": [],
            "unit": "unknown", "physical_owner": "unknown", "manufacturing_scope": "unknown",
            "engineering_value_confirmed": False, "measurement_used_for_display": False,
            "block_members_used_as_geometry_targets": False,
        }
        records.append(row)
        if dimension is None:
            row["reasons"].append("dimension_missing_or_invalid_handle")
            continue
        dp = pointers[value]
        row.update({"dimension_kind": object_kind(dimension), "dimension_source_pointer": dp,
                    "dimension_native_coordinates": coordinate_evidence(dimension, dp),
                    "raw_measurement": {"raw_value": dimension.get("act_measurement"),
                                        "source_pointer": f"{dp}/act_measurement"},
                    "raw_user_text": dimension.get("user_text"),
                    "dimension_style_raw_reference": dimension.get("dimstyle"),
                    "dimension_block_raw_reference": dimension.get("block")})
        if (not (object_kind(dimension) or "").startswith("DIMENSION_")
                or type(dimension.get("entmode")) is not int or dimension["entmode"] != 2
                or type(dimension.get("invisible")) is not int or dimension["invisible"] != 0
                or conflicts(dimension, ("handle", "entity", "object", "dxfname", "block", "entmode", "invisible"))):
            row["reasons"].append("dimension_type_visibility_or_reference_unknown")
            continue
        bh = strict_handle(dimension.get("block"), reference=True, origin=value)
        block = index.get(bh)
        row["block_handle"] = hex_handle(bh)
        if block is None:
            row["reasons"].append("block_missing_or_invalid_reference")
            continue
        bp = pointers[bh]
        row.update({"block_source_pointer": bp, "block_name": block.get("name"),
                    "block_anonymous": block.get("anonymous"),
                    "block_xref_raw_fields": {field: block.get(field) for field in
                        ("is_xref_ref", "is_xref_dep", "blkisxref", "xrefoverlaid", "xref_loaded", "xref", "xref_pname")}})
        flags = ("is_xref_dep", "blkisxref", "xrefoverlaid", "xref_loaded")
        block_fields = ("handle", "object", "entity", "dxfname", "name", "anonymous", "entities", "xref", "xref_pname") + flags
        if (object_kind(block) != "BLOCK_HEADER" or type(block.get("anonymous")) is not int
                or block["anonymous"] != 1 or not isinstance(block.get("name"), str)
                or not block["name"].startswith("*") or conflicts(block, block_fields)
                or any(type(block.get(flag)) is not int or block[flag] != 0 for flag in flags)
                or block.get("xref_pname") != ""
                or strict_handle(block.get("xref"), reference=True, origin=bh, allow_null=True) != 0):
            row["reasons"].append("anonymous_non_xref_block_not_verified")
            continue
        members = block.get("entities")
        if not isinstance(members, list):
            row["reasons"].append("block_members_missing_or_conflicting")
            continue
        seen: set[int] = set()
        for slot, ref in enumerate(members):
            mh = strict_handle(ref, reference=True, origin=bh)
            member = index.get(mh)
            evidence = {"raw_reference": ref, "reference_pointer": f"{bp}/entities/{slot}",
                        "member_handle": hex_handle(mh), "status": "unknown"}
            row["block_members"].append(evidence)
            if member is None or mh in seen:
                row["reasons"].append("member_missing_invalid_or_repeated")
                continue
            seen.add(mh)
            mp = pointers[mh]
            kind = object_kind(member)
            evidence.update({"source_pointer": mp, "kind": kind,
                             "owner_raw_reference": member.get("ownerhandle"),
                             "entmode": member.get("entmode"), "invisible": member.get("invisible")})
            required = ("handle", "ownerhandle", "entmode", "invisible", "entity", "object", "dxfname")
            if (conflicts(member, required) or strict_handle(member.get("ownerhandle"), reference=True, origin=mh) != bh
                    or type(member.get("entmode")) is not int or member["entmode"] != 0
                    or type(member.get("invisible")) is not int or member["invisible"] != 0
                    or not kind or member.get("entity") == "UNKNOWN_ENT" or member.get("object") == "UNKNOWN_OBJ"):
                row["reasons"].append("member_ownership_visibility_or_kind_unknown")
                continue
            evidence["status"] = "direct_owned_visible_member"
            if kind not in {"TEXT", "MTEXT"}:
                continue
            fields = {field: member[field] for field in ("text", "text_value") if field in member}
            text = {**evidence, "raw_text_fields": fields,
                    "native_coordinates": coordinate_evidence(member, mp),
                    "coordinate_conflicts": conflicts(member, COORDINATES),
                    "style_raw_reference": member.get("style"),
                    "text_field_occurrences": member.get(SIDECAR, {})}
            row["visible_text_members"].append(text)
            if (conflicts(member, ("text", "text_value")) or not fields
                    or any(not isinstance(raw_text, str) or raw_text == "" for raw_text in fields.values())
                    or len(set(fields.values())) != 1):
                text["raw_text"] = None
                row["reasons"].append("text_missing_or_conflicting")
            else:
                text["raw_text"] = next(iter(fields.values()))
                text["text_source_pointers"] = [f"{mp}/{field}" for field in fields]
        if len(row["visible_text_members"]) != 1:
            row["reasons"].append("zero_or_multiple_visible_text_members")
        row["reasons"] = sorted(set(row["reasons"]))
        if not row["reasons"]:
            row["status"] = "literal_native_block_text"
            row["display_text"] = row["visible_text_members"][0]["raw_text"]
            row["unit"] = "degree" if "°" in row["display_text"] else "unknown"
    return {"dimension_records": records, "duplicate_field_groups": len(duplicates),
            "conflicting_duplicate_field_groups": sum(not event["identical"] for event in duplicates),
            "invalid_object_handles": len(objects) - len(valid)}


def freeze(audit_path: Path, output: Path) -> None:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    cases, edges = audit["cases"], audit["rows"]
    if len(cases) != 3 or len(edges) != 11:
        raise ValueError("Expected fixed 3 fit sources and 11 distinct edges")
    pairs = {(row["item_code"], row["dimension_handle"], row["target_handle"]) for row in edges}
    dimensions = {(row["item_code"], row["dimension_handle"]) for row in edges}
    if len(pairs) != 11 or len(dimensions) != 7:
        raise ValueError("Expected 11 distinct pairs deduplicating to 7 dimensions")
    pins = {str(audit_path.resolve()): digest(audit_path)}
    sources = []
    for case in cases:
        source = REPO / case["native_response_path"]
        source_hash = digest(source)
        if source_hash != audit["source_pins"][case["native_response_path"]]:
            raise ValueError("Original native response changed")
        pins[str(source)] = source_hash
        sources.append({"item_code": case["item_code"], "group_id": case["group_id"],
                        "fit_line": case["fit_line"], "native_response_path": str(source),
                        "native_response_sha256": source_hash,
                        "native_dwg_sha256": next(row["native_dwg_sha256"] for row in edges
                                                  if row["item_code"] == case["item_code"]),
                        "dimension_handles": sorted(handle for code, handle in dimensions if code == case["item_code"])})
    code_dir = Path(__file__).parent
    for name in CODE_FILES:
        path = code_dir / name
        pins[str(path)] = digest(path)
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / "frozen_code"
    snapshots.mkdir()
    for name in CODE_FILES:
        shutil.copyfile(code_dir / name, snapshots / name)
    write_new_json(output / "freeze.json", {
        "schema_version": "native_dimension_display_freeze.v1", "source_pins": pins,
        "sources": sources, "dimension_count": 7, "distinct_edge_count": 11,
        "code_snapshots": {name: digest(snapshots / name) for name in CODE_FILES},
        "scope": "fixed3 Train fit original native responses only; no route labels or Val/Test",
    })


def run(output: Path, *, replay: bool = False) -> dict[str, Any]:
    frozen_path = output / "freeze.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    for path, expected in frozen["source_pins"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen input or code changed: {path}")
    cases = []
    for source in frozen["sources"]:
        try:
            extracted = extract_dimension_displays(Path(source["native_response_path"]).read_bytes(), source["dimension_handles"])
        except ValueError as error:
            extracted = {"dimension_records": [{"dimension_handle": handle, "status": "unknown",
                          "display_text": None, "reasons": [f"source_rejected: {error}"]}
                         for handle in source["dimension_handles"]]}
        cases.append({**source, **extracted})
    states = Counter(row["status"] for case in cases for row in case["dimension_records"])
    result = {"schema_version": "native_dimension_display.v1", "freeze_sha256": digest(frozen_path),
              "source_count": len(cases), "dimension_count": sum(states.values()),
              "status_counts": dict(sorted(states.items())), "cases": cases,
              "display_basis": "one directly owned visible TEXT/MTEXT member of explicit anonymous non-xref dimension block",
              "multiple_visible_text_policy": "retain all members; scalar display remains unknown; no order or concatenation guessed",
              "style_formatting_performed": False, "raw_measurement_formatted_as_display": False,
              "pdf_read": False, "route_labels_read": False, "val_test_read": False,
              "ocr_calls": 0, "new_reader_calls": 0, "gpu_used": False,
              "source_gate_promoted": False, "critical_fact_promoted": False,
              "route_training_admitted": False, "manufacturing_hard_constraint_count": 0}
    for path, expected in frozen["source_pins"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Input changed during extraction: {path}")
    product = output / "extractproducts.json"
    if replay:
        if json.loads(product.read_text(encoding="utf-8")) != result:
            raise ValueError("Independent extractproducts replay differs")
        write_new_json(output / "independent_replay.json", {"extractproducts_exact_equal": True,
                       "extractproducts_sha256": digest(product), "freeze_sha256": digest(frozen_path)})
    else:
        write_new_json(product, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        if args.audit is None or args.replay:
            parser.error("--freeze requires --audit and excludes --replay")
        freeze(args.audit, args.output_dir)
        print(json.dumps({"frozen": True, "dimension_count": 7}))
    else:
        result = run(args.output_dir, replay=args.replay)
        print(json.dumps({"source_count": result["source_count"], "dimension_count": result["dimension_count"],
                          "status_counts": result["status_counts"], "replay": args.replay}))


if __name__ == "__main__":
    main()
