"""Fresh DWG metadata and Train-only native-entity census, without route answers."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from src.tools.drawing_extraction.workflow import _load_libredwg_json
from src.tools.research.artifacts import digest, read_json, write_new_json


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs/dwg_dataset_audit_20261015"
FROZEN = REPO / "data/frozen/route_drawing_ready_20260909/assets"
TRAIN = REPO / "outputs/full_train_holdout_20261005/run002/inputs/train.jsonl"
REPEATED = REPO / "outputs/repeated_pdf_scope_20261006/r6_train_001/training_inputs.jsonl"
DWGREAD = Path("/home/userroot/miniconda3/envs/dwg-free/bin/dwgread")
DWG2DXF = Path("/home/userroot/miniconda3/envs/dwg-free/bin/dwg2dxf")
RAW_FOLDERS = [REPO / "data/raw/drawings/dwg_batch_1", REPO / "data/raw/third_batch/extracted/dwg"]
STANDARD_HEADERS = {"AC1015", "AC1018", "AC1027"}


def input_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if (not rows or len({row["item_code"] for row in rows}) != len(rows)
            or any(row.get("split") != "train" or {"route", "operations", "gt_route", "target"} & row.keys()
                   for row in rows)):
        raise ValueError("Only label-free frozen Train input identities are permitted")
    return rows


def inventory() -> dict:
    train, repeated = input_rows(TRAIN), input_rows(REPEATED)
    train_codes, repeated_codes = ({row["item_code"] for row in rows} for rows in (train, repeated))
    if not repeated_codes <= train_codes:
        raise ValueError("Repeated-route identity scope differs from original Train")
    grouped = defaultdict(list)
    for folder in [*RAW_FOLDERS, FROZEN / "dwg"]:
        for path in sorted(folder.glob("*.dwg")):
            with path.open("rb") as stream:
                header = stream.read(6)
            grouped[digest(path)].append({"path": str(path.resolve()), "item_code": path.stem,
                "bytes": path.stat().st_size, "header_hex": header.hex(),
                "header_ascii": header.decode("ascii", errors="replace"),
                "canonical": path.parent == FROZEN / "dwg"})
    codes = defaultdict(set)
    pdfs = {path.stem: str(path.resolve()) for path in (FROZEN / "pdf").glob("*.pdf")}
    records = []
    for sha, copies in sorted(grouped.items()):
        names = {copy["item_code"] for copy in copies}
        if len(names) != 1:
            raise ValueError("One byte-identical DWG has multiple item names; resolve before sampling")
        item = next(copy for copy in copies if copy["canonical"]) if any(copy["canonical"] for copy in copies) else copies[0]
        code = item["item_code"]
        codes[code].add(sha)
        records.append({**item, "source_sha256": sha, "copy_paths": [copy["path"] for copy in copies],
            "standard_header": item["header_ascii"] in STANDARD_HEADERS,
            "same_stem_pdf": pdfs.get(code), "pair_version_verified": False,
            "in_train_pdf_scope": code in train_codes, "in_repeated_train_scope": code in repeated_codes})
    if any(len(values) > 1 for values in codes.values()):
        raise ValueError("Same item code has different DWG bytes; no automatic version selection")
    canonical = [record for record in records if record["canonical"]]
    train_records = [record for record in records if record["in_train_pdf_scope"]]
    summary = {"files_in_three_source_folders": sum(len(values) for values in grouped.values()),
        "unique_dwg_contents": len(records), "unique_item_codes": len(codes),
        "duplicate_copies": sum(len(values) - 1 for values in grouped.values()),
        "same_code_content_conflicts": 0,
        "unique_content_headers": dict(Counter(record["header_ascii"] for record in records)),
        "canonical_count": len(canonical), "canonical_bytes": sum(record["bytes"] for record in canonical),
        "canonical_same_stem_pdf_count": sum(bool(record["same_stem_pdf"]) for record in canonical),
        "raw_only_count": len(records) - len(canonical), "pdf_frozen_count": len(pdfs),
        "label_free_train_pdf_scope_count": len(train), "repeated_train_scope_count": len(repeated),
        "train_dwg_count": len(train_records), "train_canonical_dwg_count": sum(r["canonical"] for r in train_records),
        "train_standard_dwg_count": sum(r["standard_header"] for r in train_records),
        "repeated_train_dwg_count": sum(r["in_repeated_train_scope"] for r in records),
        "nonstandard_items": [r["item_code"] for r in records if not r["standard_header"]],
        "nonstandard_train_items": [r["item_code"] for r in train_records if not r["standard_header"]]}
    result = {"schema_version": "dwg_dataset_inventory.v1", "summary": summary, "records": records,
        "pins": {str(path): digest(path) for path in (TRAIN, REPEATED, DWGREAD, DWG2DXF, Path(__file__))},
        "dwgread_version": subprocess.run([str(DWGREAD), "--version"], check=True, capture_output=True,
                                          text=True).stdout.strip(),
        "ezdxf_version": version("ezdxf"), "route_labels_read": False, "val_test_content_read": False,
        "pairing_is_filename_candidate_not_version_confirmation": True}
    ROOT.mkdir(parents=True, exist_ok=True)
    write_new_json(ROOT / "inventory.json", result)
    return summary


def handle(value: object) -> int | None:
    """LibreDWG JSON appends the resolved absolute handle to each reference."""
    if isinstance(value, list) and len(value) >= 2 and isinstance(value[-1], int) and value[-1] > 0:
        return value[-1]
    return None


def summarize_native(payload: dict) -> dict:
    objects = payload.get("OBJECTS")
    if not isinstance(objects, list) or not objects:
        raise ValueError("Decoded native JSON has no object database")
    types = Counter(obj.get("entity") or obj.get("object") or "missing_type" for obj in objects)
    entities = [obj for obj in objects if obj.get("entity")]
    model = [obj for obj in entities if obj.get("entmode") == 2]
    dimensions = [obj for obj in model if str(obj.get("entity", "")).startswith("DIMENSION_")]
    lookup = {handle(obj.get("handle")): obj for obj in objects if handle(obj.get("handle")) is not None}
    dim_records = []
    for obj in dimensions:
        block = lookup.get(handle(obj.get("block")))
        components = [lookup.get(handle(ref)) for ref in block.get("entities", [])] if block else []
        component_types = Counter(c.get("entity", "missing") for c in components if c)
        dim_records.append({"handle": handle(obj.get("handle")), "type": obj["entity"],
            "measurement": obj.get("act_measurement"), "user_text": obj.get("user_text"),
            "definition_fields": sorted(key for key in obj if key.endswith("_pt") or key == "text_midpt"),
            "graphics_block_handle": handle(obj.get("block")), "graphics_block_resolved": block is not None,
            "graphics_component_count": sum(component_types.values()), "graphics_component_types": dict(component_types),
            "reactor_count": len(obj.get("reactors", [])),
            "reactor_types": [lookup.get(handle(ref), {}).get("object", "unresolved") for ref in obj.get("reactors", [])]})
    leaders = [obj for obj in model if obj.get("entity") == "LEADER"]
    leader_refs = [handle(obj.get(key)) for obj in leaders for key in ("annotation", "annotation_handle") if obj.get(key)]
    blocks = [obj for obj in objects if obj.get("object") == "BLOCK_HEADER"]
    header = payload.get("HEADER", {})
    return {"object_count": len(objects), "all_object_types": dict(types),
        "direct_modelspace_entity_count": len(model), "direct_modelspace_types": dict(Counter(obj["entity"] for obj in model)),
        "direct_paperspace_types": dict(Counter(obj["entity"] for obj in entities if obj.get("entmode") == 1)),
        "direct_modelspace_dimensions": dim_records, "direct_modelspace_dimension_count": len(dimensions),
        "dimension_graphics_blocks_resolved": sum(r["graphics_block_resolved"] for r in dim_records),
        "dimension_graphics_component_count": sum(r["graphics_component_count"] for r in dim_records),
        "dimension_direct_association_objects": types.get("DIMASSOC", 0),
        "leader_annotation_reference_count": len(leader_refs),
        "leader_annotation_reference_resolved": sum(ref in lookup for ref in leader_refs),
        "unknown_objects": types.get("UNKNOWN_OBJ", 0), "unknown_entities": types.get("UNKNOWN_ENT", 0),
        "insunits": header.get("INSUNITS"), "measurement_header": header.get("MEASUREMENT"),
        "layout_count": types.get("LAYOUT", 0), "block_count": len(blocks),
        "external_reference_blocks": sum(bool(obj.get("blkisxref")) for obj in blocks),
        "linetype_names": sorted(str(obj.get("name", "")) for obj in objects if obj.get("object") == "LTYPE"),
        "text_decoding": payload.get("_text_decoding", {"replacement_count": 0}),
        "entity_role_or_part_owner_claimed": False, "unused_block_geometry_not_part_count": True}


def _parse_one(record: dict) -> dict:
    started = time.monotonic()
    source = Path(record["path"])
    result = {"item_code": record["item_code"], "source_sha256": record["source_sha256"],
        "canonical": record["canonical"], "status": "failed"}
    if digest(source) != record["source_sha256"]:
        raise ValueError("DWG source changed before the fresh read")
    try:
        with tempfile.TemporaryDirectory(prefix="jn_dwg_audit_") as name:
            folder = Path(name)
            native = folder / "native.json"
            process = subprocess.run([str(DWGREAD), "-O", "JSON", "-o", str(native), str(source)],
                stdin=subprocess.DEVNULL, capture_output=True, timeout=30,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1"})
            result.update(returncode=process.returncode, stderr=process.stderr.decode("utf-8", errors="replace")[:4000])
            if native.is_file():
                result.update(native_json_sha256=digest(native), **summarize_native(_load_libredwg_json(native)))
                result["status"] = "parsed" if process.returncode == 0 else "parsed_with_tool_error"
            else:
                result["error"] = "native_json_not_written"
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["seconds"] = time.monotonic() - started
    if digest(source) != record["source_sha256"]:
        raise ValueError("DWG source changed during the fresh read")
    return result


def native_census() -> dict:
    inventory_value = read_json(ROOT / "inventory.json")
    for path, expected in inventory_value["pins"].items():
        if digest(Path(path)) != expected:
            raise ValueError("Audit inputs, tools or code changed")
    selected = [r for r in inventory_value["records"] if r["in_train_pdf_scope"] and r["standard_header"]]
    if (ROOT / "native_train.json").exists():
        raise FileExistsError("Fresh native census is write-once")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_parse_one, selected))
    accepted = [r for r in results if r["status"] in {"parsed", "parsed_with_tool_error"}]
    counts = Counter()
    for row in accepted:
        counts.update(row["direct_modelspace_types"])
    summary = {"scope": "all_standard_header_DWG_matching_frozen_Train_PDF_scope_only",
        "attempted": len(results), "statuses": dict(Counter(r["status"] for r in results)),
        "direct_modelspace_types": dict(counts),
        "with_direct_native_dimensions": sum(r["direct_modelspace_dimension_count"] > 0 for r in accepted),
        "native_dimension_count": sum(r["direct_modelspace_dimension_count"] for r in accepted),
        "dimension_graphics_blocks_resolved": sum(r["dimension_graphics_blocks_resolved"] for r in accepted),
        "dimension_graphics_component_count": sum(r["dimension_graphics_component_count"] for r in accepted),
        "with_direct_model_INSERT": sum(r["direct_modelspace_types"].get("INSERT", 0) > 0 for r in accepted),
        "with_unknown_entities": sum(r["unknown_entities"] > 0 for r in accepted),
        "with_unknown_objects": sum(r["unknown_objects"] > 0 for r in accepted),
        "with_xref_blocks": sum(r["external_reference_blocks"] > 0 for r in accepted),
        "insunits_distribution": dict(Counter(str(r["insunits"]) for r in accepted)),
        "with_text_decode_replacements": sum(r["text_decoding"].get("replacement_count", 0) > 0 for r in accepted),
        "wall_seconds": time.monotonic() - started, "workers": 2,
        "readable_JSON_not_full_entity_accuracy": True, "same_filename_not_version_acceptance": True,
        "route_labels_read": False, "val_test_content_read": False, "ocr_calls": 0, "gpu_used": False}
    write_new_json(ROOT / "native_train.json", {"summary": summary, "records": results,
        "inventory_sha256": digest(ROOT / "inventory.json")})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("inventory", "native"))
    args = parser.parse_args()
    print(json.dumps({"inventory": inventory, "native": native_census}[args.stage](), ensure_ascii=False))


if __name__ == "__main__":
    main()
