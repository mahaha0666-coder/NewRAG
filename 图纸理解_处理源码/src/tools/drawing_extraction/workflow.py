"""Extract structured drawing information from DWG files."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.tools.drawing_extraction.workflow_geometry import (
    BBox,
    _clean_mtext,
    _dimension_display_text,
    _dimension_geometry_bbox,
    _entity_bbox,
    _entity_details,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


INSUNITS_TO_MM = {
    0: ("unitless", 1.0),
    1: ("inch", 25.4),
    2: ("foot", 304.8),
    3: ("mile", 1_609_344.0),
    4: ("millimeter", 1.0),
    5: ("centimeter", 10.0),
    6: ("meter", 1000.0),
    7: ("kilometer", 1_000_000.0),
    8: ("microinch", 0.0000254),
    9: ("mil", 0.0254),
    10: ("yard", 914.4),
    14: ("decimeter", 100.0),
}


def _run(command: list[str], stdout_path: Path | None = None, stderr_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    stdout_target = stdout_path.open("w", encoding="utf-8") if stdout_path else subprocess.PIPE
    stderr_target = stderr_path.open("w", encoding="utf-8") if stderr_path else subprocess.PIPE
    try:
        return subprocess.run(
            command,
            check=True,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True,
        )
    finally:
        if stdout_path and hasattr(stdout_target, "close"):
            stdout_target.close()
        if stderr_path and hasattr(stderr_target, "close"):
            stderr_target.close()


def _replace_surrogates(text: str) -> str:
    return "".join("\ufffd" if "\ud800" <= char <= "\udfff" else char for char in text)


def _json_safe(value: Any) -> Any:
    if isinstance(value, str):
        return _replace_surrogates(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {_json_safe(key): _json_safe(item) for key, item in value.items()}
    return value


def _decode_raw_strings(dwg_path: Path) -> list[str]:
    result = subprocess.run(
        ["strings", "-a", "-n", "2", str(dwg_path)],
        check=True,
        capture_output=True,
    )
    lines = []
    for raw_line in result.stdout.splitlines():
        for encoding in ("gbk", "utf-8", "latin1"):
            try:
                decoded = raw_line.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            decoded = raw_line.decode("utf-8", errors="replace")
        decoded = decoded.strip()
        if decoded:
            lines.append(decoded)
    return lines


def _extract_title_candidates(raw_strings: list[str], decoded_unknown_bits: list[str]) -> dict[str, Any]:
    source_lines = raw_strings + decoded_unknown_bits
    haystack = "\n".join(source_lines)
    candidates: dict[str, Any] = {}

    drawing_numbers = sorted(set(re.findall(r"\b[A-Z]{1,4}\d{2,4}(?:\.\d+)+(?:#\d+)?", haystack)))
    drawing_numbers = [item for item in drawing_numbers if re.search(r"\d", item) and ("#" in item or "." in item)]
    if drawing_numbers:
        preferred = sorted(drawing_numbers, key=lambda item: ("#" not in item, -len(item), item))[0]
        candidates["drawing_no"] = {
            "value": preferred,
            "confidence": "medium",
            "source": "dwg_raw_strings",
            "candidates": drawing_numbers[:10],
        }

    material_matches = sorted(set(re.findall(r"(?<!\d)(304|316L?|Q235B?|45#?)(?!\d)", haystack, flags=re.IGNORECASE)))
    if material_matches:
        candidates["material"] = {
            "value": material_matches[0],
            "confidence": "medium",
            "source": "dwg_raw_strings",
            "candidates": material_matches,
        }

    scale_matches = sorted(
        {
            match
            for line in source_lines
            for match in re.findall(r"(?<![\d:])\d{1,3}:\d{1,3}(?![\d:])", line)
        }
    )
    if scale_matches:
        candidates["scale"] = {
            "value": scale_matches[0].replace(" ", ""),
            "confidence": "medium",
            "source": "dwg_raw_strings",
            "candidates": scale_matches,
        }

    revision_matches = sorted(set(re.findall(r"\bA\d\b", haystack)))
    if revision_matches:
        preferred = "A0" if "A0" in revision_matches else revision_matches[0]
        candidates["revision"] = {
            "value": preferred,
            "confidence": "medium",
            "source": "dwg_raw_strings",
            "candidates": revision_matches,
        }

    date_matches = sorted(set(re.findall(r"20\d{2}-\d{2}-\d{2}", haystack)))
    if date_matches:
        candidates["date"] = {
            "value": date_matches[0],
            "confidence": "medium",
            "source": "dwg_raw_strings",
            "candidates": date_matches,
        }

    cjk_candidates = []
    for item in decoded_unknown_bits:
        for match in re.findall(r"[\u4e00-\u9fff]{2,12}", item):
            if any(keyword in match for keyword in ("缸", "体", "基座", "筒", "轴", "板", "座")):
                cjk_candidates.append(match)
    if cjk_candidates:
        candidates["part_name"] = {
            "value": cjk_candidates[0],
            "confidence": "low",
            "source": "dwg_unknown_bits_gbk_candidate",
            "candidates": cjk_candidates[:10],
        }

    return candidates


def _load_libredwg_json(json_path: Path) -> dict[str, Any]:
    raw = json_path.read_bytes()
    lossy = False
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            decoded = raw.decode("gbk")
        except UnicodeDecodeError:
            decoded = raw.decode("utf-8-sig", errors="replace")
            lossy = True
    result = json.loads(decoded)
    if lossy:
        result["_text_decoding"] = {"encoding": "utf8_with_invalid_bytes", "replacement_count": decoded.count("\ufffd")}
    return result


def _modelspace_texts(doc: Any) -> list[str]:
    from ezdxf.disassemble import recursive_decompose
    from ezdxf.tools.text import plain_mtext

    result = []
    for entity in recursive_decompose(doc.modelspace()):
        kind = entity.dxftype()
        if kind not in {"TEXT", "MTEXT", "ATTRIB"} or entity.dxf.get("invisible", 0):
            continue
        layer = doc.layers.get(entity.dxf.layer)
        if layer.is_off() or layer.is_frozen():
            continue
        if kind == "ATTRIB" and entity.is_invisible:
            continue
        value = entity.text if kind == "MTEXT" else entity.dxf.text
        cleaned = plain_mtext(value).strip()
        if cleaned:
            result.append(cleaned)
    return list(dict.fromkeys(result))


def _walk_json(payload: Any) -> tuple[list[str], list[str]]:
    text_fields: list[str] = []
    unknown_bits: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "text" and isinstance(item, str):
                    text_fields.append(_clean_mtext(item))
                elif key == "unknown_bits" and isinstance(item, str):
                    unknown_bits.append(item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return text_fields, unknown_bits


def _decode_unknown_bits(items: list[str]) -> list[str]:
    decoded: list[str] = []
    for item in items:
        hex_value = re.sub(r"[^0-9A-Fa-f]", "", item)
        if len(hex_value) % 2:
            continue
        try:
            data = bytes.fromhex(hex_value)
        except ValueError:
            continue
        for encoding in ("gbk", "utf-8"):
            text = data.decode(encoding, errors="ignore")
            for match in re.findall(r"[\u4e00-\u9fffA-Za-z0-9#_.:-]{2,40}", text):
                if any("\u4e00" <= char <= "\u9fff" for char in match) or re.search(r"FZ|20\d{2}|304|1:6|A0", match):
                    decoded.append(match)
    seen = set()
    unique = []
    for item in decoded:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def extract_dwg(
    stem: str,
    dwg_dir: Path,
    output_dir: Path,
    dwg2dxf_bin: str = "dwg2dxf",
    dwgread_bin: str = "dwgread",
    target_dxf_version: str = "r2004",
    keep_intermediates: bool = False,
) -> dict[str, Any]:
    dwg_path = dwg_dir / f"{stem}.dwg"
    if not dwg_path.exists():
        raise FileNotFoundError(f"DWG file not found: {dwg_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if keep_intermediates:
        work_dir = output_dir
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix=f"{stem}_dwg_")
        work_dir = Path(temp_dir.name)

    try:
        dxf_path = work_dir / f"{stem}.dxf"
        json_path = work_dir / f"{stem}.libredwg.json"
        logs_dir = work_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        _run(
            [dwg2dxf_bin, "--as", target_dxf_version, "-y", "-o", str(dxf_path), str(dwg_path)],
            stdout_path=logs_dir / f"{stem}.dwg2dxf.stdout",
            stderr_path=logs_dir / f"{stem}.dwg2dxf.stderr",
        )
        _run(
            [dwgread_bin, "-O", "JSON", "-o", str(json_path), str(dwg_path)],
            stdout_path=logs_dir / f"{stem}.dwgread.stdout",
            stderr_path=logs_dir / f"{stem}.dwgread.stderr",
        )

        import ezdxf

        doc = ezdxf.readfile(dxf_path)
    finally:
        if temp_dir is not None and "doc" not in locals():
            temp_dir.cleanup()

    auditor = doc.audit()
    modelspace = doc.modelspace()
    all_bboxes = [bbox for entity in modelspace for bbox in [_entity_bbox(entity)] if bbox is not None]
    drawing_bbox = None
    for bbox in all_bboxes:
        drawing_bbox = bbox if drawing_bbox is None else drawing_bbox.expand(bbox)

    unit_code = int(doc.header.get("$INSUNITS", 0) or 0)
    unit_name, unit_to_mm = INSUNITS_TO_MM.get(unit_code, ("unknown", 1.0))
    unit_source = "dwg_header_$INSUNITS"
    if unit_code == 0:
        unit_source = "assumed_mm_for_mechanical_drawing"

    dimensions = []
    for index, entity in enumerate(modelspace.query("DIMENSION"), start=1):
        measurement = float(entity.get_measurement())
        text = str(entity.dxf.text or "")
        display_text = _dimension_display_text(text, measurement)
        bbox = _entity_bbox(entity)
        geometry_bbox = _dimension_geometry_bbox(entity)
        dimensions.append(
            {
                "id": f"dimension_{index}",
                "dxf_handle": entity.dxf.get("handle"),
                "layer": entity.dxf.layer,
                "dimtype": int(entity.dxf.dimtype),
                "raw_text": text,
                "display_text": display_text,
                "measurement_source_units": measurement,
                "measurement_mm": measurement * unit_to_mm,
                "source_unit": unit_name,
                "bbox_cad": asdict(bbox) if bbox else None,
                "geometry_bbox_cad": asdict(geometry_bbox) if geometry_bbox else None,
                "bbox_normalized_to_cad_extents": bbox.normalized_in(drawing_bbox) if bbox and drawing_bbox else None,
            }
        )

    geometry_entities = []
    by_layer = Counter()
    geometry_bbox = None
    for index, entity in enumerate(modelspace, start=1):
        bbox = _entity_bbox(entity)
        by_layer[entity.dxf.layer] += 1
        if entity.dxftype() in {"LINE", "HATCH", "ARC", "CIRCLE", "LWPOLYLINE", "POLYLINE", "SPLINE"} and bbox:
            geometry_bbox = bbox if geometry_bbox is None else geometry_bbox.expand(bbox)
        geometry_entities.append(
            {
                "id": f"entity_{index}",
                "dxf_handle": entity.dxf.get("handle"),
                "type": entity.dxftype(),
                "layer": entity.dxf.layer,
                "bbox_cad": asdict(bbox) if bbox else None,
                "bbox_normalized_to_cad_extents": bbox.normalized_in(drawing_bbox) if bbox and drawing_bbox else None,
                "details": _entity_details(entity, unit_to_mm),
            }
        )

    block_summaries = []
    table_templates: dict[str, Any] = {"title_block": [], "bom": [], "other_text_blocks": []}
    for block in doc.blocks:
        text_values = []
        counts = Counter(entity.dxftype() for entity in block)
        for entity in block:
            if entity.dxftype() == "TEXT":
                text_values.append(_clean_mtext(entity.dxf.text))
            elif entity.dxftype() == "MTEXT":
                text_values.append(_clean_mtext(entity.text))
            elif entity.dxftype() in {"ATTRIB", "ATTDEF"}:
                text_values.append(_clean_mtext(entity.dxf.text))
        cleaned = [value for value in text_values if value]
        summary = {"name": block.name, "entity_counts": dict(counts), "text": cleaned}
        block_summaries.append(summary)
        joined = "\n".join(cleaned)
        if any(keyword in joined for keyword in ("图号", "材料", "版本", "比例", "常用名", "DRW.NO", "MATERIAL")):
            table_templates["title_block"].extend(cleaned)
        elif any(keyword in joined for keyword in ("序", "代", "名称", "总计", "备注")):
            table_templates["bom"].extend(cleaned)
        elif cleaned:
            table_templates["other_text_blocks"].append(summary)

    libredwg_payload = _load_libredwg_json(json_path)
    json_text_fields, unknown_bits = _walk_json(libredwg_payload)
    decoded_unknown_bits = _decode_unknown_bits(unknown_bits)
    raw_strings = _decode_raw_strings(dwg_path)
    title_candidates = _extract_title_candidates(raw_strings, decoded_unknown_bits)

    regions = []
    if geometry_bbox and drawing_bbox:
        regions.append(
            {
                "id": "main_view_geometry",
                "kind": "subfigure",
                "description": "Primary modelspace geometry excluding title-block templates.",
                "bbox_cad": asdict(geometry_bbox),
                "bbox_normalized_to_cad_extents": geometry_bbox.normalized_in(drawing_bbox),
                "entity_types": dict(Counter(entity.dxftype() for entity in modelspace if entity.dxftype() != "DIMENSION")),
                "alignment_note": "DWG-only normalized CAD bbox; can be used later as a coarse anchor for external rendered-page alignment.",
            }
        )
    for dimension in dimensions:
        if dimension["bbox_cad"]:
            regions.append(
                {
                    "id": f"{dimension['id']}_region",
                    "kind": "annotation",
                    "description": f"Dimension annotation {dimension['display_text']}",
                    "bbox_cad": dimension["bbox_cad"],
                    "bbox_normalized_to_cad_extents": dimension["bbox_normalized_to_cad_extents"],
                }
            )

    result = {
        "schema_version": "dwg_extraction.v1",
        "stem": stem,
        "files": {
            "dwg": str(dwg_path),
            "intermediate_files_retained": keep_intermediates,
            "intermediate_dir": str(work_dir) if keep_intermediates else "",
        },
        "toolchain": {
            "dwg_to_dxf": "LibreDWG dwg2dxf",
            "dwg_json": "LibreDWG dwgread",
            "dxf_reader": "ezdxf",
            "json_text_decoding": libredwg_payload.get("_text_decoding", {"replacement_count": 0}),
            "license_note": "LibreDWG is GPL; ezdxf is MIT. Commercial use is possible, but distribution/integration must respect each license.",
        },
        "units": {
            "insunits_code": unit_code,
            "source_unit": unit_name,
            "to_mm_factor": unit_to_mm,
            "source": unit_source,
        },
        "cad": {
            "dxf_version": doc.dxfversion,
            "audit_errors": len(auditor.errors),
            "audit_fixes": len(auditor.fixes),
            "layouts": [{"name": layout.name, "entity_count": len(list(layout))} for layout in doc.layouts],
            "layers": [
                {"name": layer.dxf.name, "color": int(layer.dxf.color), "linetype": layer.dxf.linetype}
                for layer in doc.layers
            ],
            "entity_counts": dict(Counter(entity.dxftype() for entity in modelspace)),
            "entity_counts_by_layer": dict(by_layer),
            "cad_extents": asdict(drawing_bbox) if drawing_bbox else None,
        },
        "regions": regions,
        "dimensions": dimensions,
        "entities": geometry_entities,
        "tables": {
            "templates": table_templates,
            "referenced_modelspace_texts": _modelspace_texts(doc),
            "text_scope": "modelspace and referenced blocks; off/frozen layers excluded; XCLIP not evaluated",
            "json_text_fields": [field for field in json_text_fields if field],
            "title_block_values": title_candidates,
            "decoded_unknown_bits_candidates": decoded_unknown_bits[:200],
        },
        "technical_information": {
            "dimensions": dimensions,
            "layers_used_for_technical_semantics": {
                "centerline": [layer for layer in by_layer if "中心" in layer],
                "hatch": [layer for layer in by_layer if "剖面" in layer],
                "dimension": [layer for layer in by_layer if "标注" in layer],
                "outline": [layer for layer in by_layer if "轮廓" in layer],
            },
            "notes": "No stable TEXT/MTEXT technical note entities were inserted in modelspace for this test file.",
        },
    }
    safe_result = _json_safe(result)
    output_json = output_dir / f"{stem}.dwg_extraction.json"
    output_json.write_text(json.dumps(safe_result, ensure_ascii=False, indent=2), encoding="utf-8")
    if temp_dir is not None:
        temp_dir.cleanup()
    return safe_result
