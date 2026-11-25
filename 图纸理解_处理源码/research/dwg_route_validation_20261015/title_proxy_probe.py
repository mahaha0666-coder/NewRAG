"""Decode source-pinned TDMTITLE preview commands, without PDF or opaque scans.

By default, the original fixed three cases receive graphical text decoding;
--all-fixed14 decodes the complete fixed Train selection. All 14 fixed sources
retain a field inventory, including failures. Stored preview text
is native display evidence; custom object parameters and field roles stay
separate from the recovered strings.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any

import ezdxf
from ezdxf.proxygraphic import ProxyGraphic, ProxyGraphicTypes


REPO = Path(__file__).resolve().parents[2]
NATIVE_ROOT = REPO / "outputs/dwg_route_validation_20261015/parser_comparison_001/fixed14_parser_probe_001"
OUTPUT_ROOT = REPO / "outputs/dwg_route_validation_20261015/title_proxy_probe_002"
ALL_OUTPUT_ROOT = REPO / "outputs/dwg_route_validation_20261015/title_proxy_probe_003"
FIXED_CODES = ("12830010725", "12500575523", "12501700017")
PROTOCOL_SHA256 = "f0c5b95f9063de43d69f54ac4c2b390a9c116f7b9c48629a5be34342b1c6b678"
CODEPAGE_ENCODINGS = {39: "gbk"}  # Pinned LibreDWG src/codepages.h:82, ANSI_936.
TEXT_OPCODES = {10: (96, False), 11: (72, False), 36: (96, True), 38: (72, True)}
# Assistant's blind review of strings recovered from these previews, not PDF.
# The role assignment is explicitly a candidate; the exact bytes remain primary.
BLIND_ROLE_CANDIDATES = {
    "12830010725": {"title_name": "轴", "drawing_no": "LHS1200.8001#3", "revision": "A2", "common_name": "搅拌轴"},
    "12500575523": {"title_name": "门", "drawing_no": "QD1000.5057#99", "revision": "A0", "common_name": "大视镜左门体焊接316L"},
    "12501700017": {"title_name": "框架部件", "drawing_no": "HZD2000.5170#3", "revision": "A11", "common_name": "HZD2000框架"},
}


@dataclass(frozen=True)
class Frame:
    index: int
    offset: int
    size: int
    opcode: int

    @property
    def name(self) -> str:
        try:
            return ProxyGraphicTypes(self.opcode).name
        except ValueError:
            return f"UNKNOWN_TYPE_{self.opcode}"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def handle_hex(value: list[int]) -> str:
    if not value or value[-1] <= 0:
        raise ValueError("Missing native object handle")
    return f"{value[-1]:X}"


def decode_hex_preview(value: str, declared_size: int) -> bytes:
    if not isinstance(value, str) or len(value) % 2 or not re.fullmatch(r"[0-9A-Fa-f]*", value):
        raise ValueError("Preview must be uninterrupted hexadecimal")
    data = bytes.fromhex(value)
    if len(data) != declared_size:
        raise ValueError("Native preview_size disagrees with decoded bytes")
    return data


def validate_frames(data: bytes) -> list[Frame]:
    """Use only the documented ezdxf stream origin; never search for a start."""
    if len(data) < 8 or struct.unpack_from("<I", data)[0] != len(data):
        raise ValueError("Preview header length does not match the complete stream")
    frames: list[Frame] = []
    offset = 8
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError(f"Truncated command header at {offset}")
        size, opcode = struct.unpack_from("<II", data, offset)
        if size < 8 or offset + size > len(data):
            raise ValueError(f"Invalid command extent at {offset}: size={size}")
        frames.append(Frame(len(frames), offset, size, opcode))
        offset += size
    return frames


def strict_text_record(data: bytes, frame: Frame, encoding: str) -> dict[str, Any]:
    relative, unicode_text = TEXT_OPCODES[frame.opcode]
    body_start, end = frame.offset + 8, frame.offset + frame.size
    text_start = body_start + relative
    if text_start >= end:
        raise ValueError("Text command is shorter than its fixed fields")
    stride = 2 if unicode_text else 1
    terminator = b"\x00\x00" if unicode_text else b"\x00"
    text_end = None
    for position in range(text_start, end - stride + 1, stride):
        if data[position:position + stride] == terminator:
            text_end = position
            break
    if text_end is None:
        raise ValueError("Text command has no aligned terminator")
    text_bytes = data[text_start:text_end]
    chosen_encoding = "utf-16-le" if unicode_text else encoding
    raw_text = text_bytes.decode(chosen_encoding, errors="strict")
    points = struct.unpack_from("<9d", data, body_start)
    if not all(math.isfinite(number) for number in points):
        raise ValueError("Nonfinite native text coordinates")
    return {
        "raw_text": raw_text,
        "raw_text_hex": text_bytes.hex().upper(),
        "strict_encoding": chosen_encoding,
        "text_byte_start": text_start,
        "text_byte_end": text_end,
        "terminator_byte_offset": text_end,
        "raw_start_point": list(points[:3]),
        "raw_normal": list(points[3:6]),
        "raw_text_direction": list(points[6:9]),
        "raw_frame_hex": data[frame.offset:end].hex().upper(),
    }


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite decoded entity attribute")
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if hasattr(value, "__iter__"):
        return [json_value(item) for item in value]
    raise TypeError(f"Unsupported entity attribute {type(value).__name__}")


def decode_preview(data: bytes, *, encoding: str, version: str) -> dict[str, Any]:
    frames = validate_frames(data)
    graphic = ProxyGraphic(data, dxfversion=version)
    graphic.encoding = encoding
    entities: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    unsupported = []
    matrix_records = []
    maximum_matrix_depth = 0
    for frame in frames:
        raw_text = strict_text_record(data, frame, encoding) if frame.opcode in TEXT_OPCODES else None
        method = getattr(graphic, frame.name.lower(), None)
        if method is None:
            unsupported.append({"frame_index": frame.index, "opcode": frame.opcode, "name": frame.name})
            continue
        before_matrices = [list(matrix) for matrix in graphic.matrices]
        result = method(data[frame.offset + 8:frame.offset + frame.size])
        maximum_matrix_depth = max(maximum_matrix_depth, len(graphic.matrices))
        if frame.opcode in (29, 30, 31):
            matrix_records.append({"frame_index": frame.index, "byte_offset": frame.offset,
                                   "opcode": frame.opcode, "name": frame.name,
                                   "raw_frame_hex": data[frame.offset:frame.offset + frame.size].hex().upper(),
                                   "stack_before": before_matrices,
                                   "stack_after": [list(matrix) for matrix in graphic.matrices],
                                   "coordinate_validation_status": "unknown; library transform is retained but not independently validated"})
        emitted = result if isinstance(result, tuple) else (result,) if result is not None else ()
        for entity in emitted:
            if graphic.matrices:
                entity.transform(graphic.matrices[-1])
            attributes = json_value(entity.dxfattribs())
            row = {"frame_index": frame.index, "byte_offset": frame.offset, "byte_size": frame.size,
                   "opcode": frame.opcode, "opcode_name": frame.name, "type": entity.dxftype(),
                   "attributes": attributes,
                   "matrix_stack": [list(matrix) for matrix in graphic.matrices]}
            if entity.dxftype() == "POLYLINE":
                row["vertices"] = [json_value(vertex.dxfattribs()) for vertex in entity.vertices]
            entities.append(row)
            if raw_text is not None:
                text_row = dict(row, **raw_text, decoded_text=entity.dxf.text,
                                plain_text=entity.plain_text(),
                                library_text_identical_to_raw=entity.dxf.text == raw_text["raw_text"],
                                wcs_insert=json_value(entity.ocs().to_wcs(entity.dxf.insert)),
                                layer_name_mapping="unknown; library fallback is not native layer identity")
                texts.append(text_row)
        if result:
            graphic.fill = False
    border_lines = []
    for row in entities:
        vertices = row.get("vertices", [])
        if row["type"] != "POLYLINE" or len(vertices) != 2:
            continue
        start, end = vertices[0]["location"], vertices[1]["location"]
        if abs(start[0] - end[0]) <= 1e-9 or abs(start[1] - end[1]) <= 1e-9:
            border_lines.append({"frame_index": row["frame_index"], "byte_offset": row["byte_offset"],
                                 "start": start, "end": end, "status": "axis_aligned_line_candidate",
                                 "basis": "decoded two-vertex native POLYLINE; no cell ownership inferred"})
    return {
        "header_raw_hex": data[:8].hex().upper(),
        "header_second_uint_uninterpreted": struct.unpack_from("<I", data, 4)[0],
        "header_second_uint_equals_frame_count": struct.unpack_from("<I", data, 4)[0] == len(frames),
        "frames": [{"index": f.index, "offset": f.offset, "size": f.size,
                    "opcode": f.opcode, "name": f.name} for f in frames],
        "opcode_counts": dict(Counter(frame.name for frame in frames)),
        "entity_counts": dict(Counter(row["type"] for row in entities)),
        "texts": texts, "entities": entities, "unsupported_commands": unsupported,
        "matrix_records": matrix_records,
        "border_line_candidates": border_lines,
        "label_text_candidates": [row for row in texts if any(label in row["raw_text"]
            for label in ("图号", "版本", "常用名", "编码", "材料", "重量", "比例", "数量"))],
        "coordinate_limits": {
            "push_matrix2_present": any(frame.opcode == 30 for frame in frames),
            "matrix_commands_present": any(frame.opcode in (29, 30, 31) for frame in frames),
            "maximum_matrix_depth": maximum_matrix_depth,
            "matrix_stack_balanced_at_end": not graphic.matrices,
            "non_z_text_normals": any(row["raw_normal"] != [0.0, 0.0, 1.0] for row in texts),
            "rotated_text_frames": [row["frame_index"] for row in texts
                if abs(row["raw_text_direction"][1]) > 1e-9 or row["raw_text_direction"][0] < 0
                or abs(row["raw_text_direction"][2]) > 1e-9],
            "coordinate_validation_status": "unknown_transform_or_ocs_validation" if (
                any(frame.opcode in (29, 30, 31) for frame in frames)
                or any(row["raw_normal"] != [0.0, 0.0, 1.0] for row in texts))
                else "native_graphic_xy_coordinates; text rotation retained; cells not assigned",
            "preview_coordinates_are_stored_native_graphic_coordinates": True,
            "underlying_custom_object_parameters_decoded": False,
        },
    }


def literal_field_candidates(texts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Retain every matching literal; role review cannot suppress conflicts."""
    patterns = {"drawing_no": r"[A-Z][A-Z0-9]*\d(?:\.\d+)+(?:#\d+)?",
                "revision": r"A\d{1,3}", "item_code": r"\d{11}"}
    return {field: [{"value": row["raw_text"], "frame_index": row["frame_index"],
                     "byte_offset": row["byte_offset"], "cad_insert": row["attributes"]["insert"]}
                    for row in texts if re.fullmatch(pattern, row["raw_text"].strip())]
            for field, pattern in patterns.items()}


def role_candidates(code: str, texts: list[dict[str, Any]]) -> dict[str, Any]:
    if code not in BLIND_ROLE_CANDIDATES:
        return {field: {"status": "unknown", "value": None,
                        "basis": "no blind assistant field-role review for this fixed source", "evidence": []}
                for field in ("title_name", "drawing_no", "revision", "common_name")}
    result = {}
    for field, value in BLIND_ROLE_CANDIDATES[code].items():
        matches = [row for row in texts if row["raw_text"] == value]
        result[field] = {
            "status": "candidate_field_role" if len(matches) == 1 else "unknown",
            "value": value if len(matches) == 1 else None,
            "exact_native_string_recovered": len(matches) == 1,
            "basis": "assistant blind review of TDMTITLE graphic strings; not a decoded custom parameter",
            "evidence": [{"frame_index": row["frame_index"], "byte_offset": row["byte_offset"],
                          "raw_text_hex": row["raw_text_hex"], "cad_insert": row["attributes"]["insert"]}
                         for row in matches],
        }
    return result


def write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run(native_root: Path, output: Path, *, all_fixed14: bool = False) -> dict[str, Any]:
    protocol_path = native_root / "PROTOCOL.json"
    if digest(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("Fixed 14 native source protocol changed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if len(protocol["sources"]) != 14 or protocol.get("gt_or_val_test_read") is not False:
        raise ValueError("Expected the fixed 14 Train source protocol")
    decode_codes = tuple(source["item_code"] for source in protocol["sources"]) if all_fixed14 else FIXED_CODES
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    output.mkdir(parents=True)
    inventories, decoded_cases = [], []
    for source in protocol["sources"]:
        code = source["item_code"]
        source_path = Path(source["path"])
        if digest(source_path) != source["source_sha256"]:
            raise ValueError(f"Original DWG hash changed: {code}")
        response_path = native_root / "cases" / code / "patched.response.json"
        execution_path = response_path.with_name("patched.execution.json")
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        raw = response_path.read_bytes()
        response_hash = hashlib.sha256(raw).hexdigest()
        if response_hash != execution["stdout_sha256"]:
            raise ValueError(f"Raw response hash changed: {code}")
        inventory = {"item_code": code, "source_path": str(source_path), "source_sha256": source["source_sha256"],
                     "response_path": str(response_path.resolve()), "response_sha256": response_hash,
                     "title_objects": [], "status": "unknown", "literal_field_candidates": {}}
        inventories.append(inventory)
        try:
            native = json.loads(raw.decode("utf-8-sig", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            inventory["reason"] = f"{type(error).__name__}: {error}"
            continue
        classes = {row["number"]: row for row in native["CLASSES"]}
        for object_index, obj in enumerate(native["OBJECTS"]):
            class_row = classes.get(obj.get("type"), {})
            if "TDMTITLE" not in class_row.get("dxfname", ""):
                continue
            metadata = {"class": class_row, "object_index": obj.get("index"),
                        "object_array_index": object_index, "handle": handle_hex(obj["handle"]),
                        "storage_type": obj.get("entity") or obj.get("object"),
                        "entmode": obj.get("entmode"),
                        "field_names": sorted(obj), "native_pointer": f"#/OBJECTS/{object_index}",
                        "native_fields_except_opaque_payloads": {key: value for key, value in obj.items()
                            if key not in ("unknown_bits", "preview", "proxy_data")},
                        "preview_exists": obj.get("preview_exists"), "preview_is_proxy": obj.get("preview_is_proxy"),
                        "preview_size": obj.get("preview_size"), "proxy_data_size": obj.get("proxy_data_size"),
                        "unknown_bits_decoded_or_scanned": False}
            inventory["title_objects"].append(metadata)
            if code not in decode_codes or class_row.get("dxfname") != "TDMTITLE":
                continue
            if not obj.get("preview_exists") or "preview" not in obj:
                metadata["decode_status"] = "unknown_no_preview"
                continue
            encoding = CODEPAGE_ENCODINGS.get(native["FILEHEADER"].get("codepage"))
            if encoding is None:
                metadata["decode_status"] = "unknown_unsupported_source_codepage"
                continue
            try:
                data = decode_hex_preview(obj["preview"], obj["preview_size"])
                decoded = decode_preview(data, encoding=encoding, version=native["FILEHEADER"]["version"])
            except (ValueError, TypeError, IndexError, OverflowError, struct.error) as error:
                metadata["decode_status"] = "unknown_decode_error"
                metadata["decode_error"] = f"{type(error).__name__}: {error}"
                continue
            case_dir = output / "cases" / code / metadata["handle"]
            case_dir.mkdir(parents=True)
            binary_path = case_dir / "preview.bin"
            with binary_path.open("xb") as stream:
                stream.write(data)
            decoded.update({"source": inventory.copy(), "title_object": metadata.copy(),
                            "native_graphic_pointer": metadata["native_pointer"] + "/preview",
                            "preview_path": str(binary_path.resolve()), "preview_sha256": digest(binary_path),
                            "source_codepage": native["FILEHEADER"]["codepage"], "strict_encoding": encoding,
                            "role_candidates": role_candidates(code, decoded["texts"]),
                            "literal_field_candidates": literal_field_candidates(decoded["texts"])})
            for field, candidates in decoded["literal_field_candidates"].items():
                inventory["literal_field_candidates"].setdefault(field, []).extend(
                    dict(candidate, title_object_handle=metadata["handle"]) for candidate in candidates)
            decoded_path = case_dir / "decoded.json"
            write_json(decoded_path, decoded)
            metadata["decode_status"] = "native_graphic_text_recovered"
            decoded_cases.append({"item_code": code, "handle": metadata["handle"],
                                  "decoded_path": str(decoded_path.resolve()), "decoded_sha256": digest(decoded_path),
                                  "preview_sha256": digest(binary_path), "frame_count": len(decoded["frames"]),
                                  "text_count": len(decoded["texts"]), "role_candidates": decoded["role_candidates"],
                                  "literal_field_candidates": decoded["literal_field_candidates"],
                                  "border_line_candidate_count": len(decoded["border_line_candidates"]),
                                  "coordinate_limits": decoded["coordinate_limits"]})
        inventory["status"] = "field_inventory_only" if code not in decode_codes else (
            "all_fixed14_probed" if all_fixed14 else "fixed_three_probed")
        inventory["modelspace_tdmtitle_count"] = sum(row["class"]["dxfname"] == "TDMTITLE"
            and row["entmode"] == 2 for row in inventory["title_objects"])
        inventory["field_conflicts"] = {field: len({row["value"] for row in candidates}) > 1
            for field, candidates in inventory["literal_field_candidates"].items()}
    result = {"schema_version": "tdmtitle_native_preview_probe.v1", "protocol_path": str(protocol_path.resolve()),
              "protocol_sha256": digest(protocol_path), "source_count": 14, "decode_fixed_codes": list(decode_codes),
              "all_fixed14_mode": all_fixed14,
              "script_sha256": digest(Path(__file__)), "ezdxf_version": ezdxf.__version__,
              "inventories": inventories, "decoded_cases": decoded_cases,
              "pdf_read": False, "ocr_used": False, "route_gt_or_val_test_read": False,
              "unknown_bits_scanned": False, "parser_recompiled": False}
    write_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, default=NATIVE_ROOT)
    parser.add_argument("--all-fixed14", action="store_true", help="Decode all 14 original fixed Train previews")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (ALL_OUTPUT_ROOT if args.all_fixed14 else OUTPUT_ROOT)
    result = run(args.native_root.resolve(), output.resolve(), all_fixed14=args.all_fixed14)
    print(json.dumps({"inventory_count": len(result["inventories"]),
                      "decoded_cases": result["decoded_cases"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
