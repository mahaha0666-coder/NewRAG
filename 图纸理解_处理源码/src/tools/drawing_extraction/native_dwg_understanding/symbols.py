"""Expose original plugin previews without inventing symbol parameters or owners."""

from __future__ import annotations

import math
import re
import struct

from ezdxf.tools.text import plain_mtext

from .geometry import GeometryIndex, bbox

SYMBOL_TYPES = {
    "XTWELDSYMBOL": ("welding", "焊接符号"),
    "XTCHAMFER": ("chamfer", "倒角标注"),
    "XTSHAPEPOSITION": ("gdt", "形位公差"),
    "XTBENCHMARK": ("datum", "基准标注"),
    "XTCUTSYMBOL": ("section", "剖切标识"),
    "XTDOUBLETEXT": ("note", "引出说明"),
    "XTROUGH": ("roughness", "表面粗糙度"),
    "TH_DimRoughA": ("roughness", "表面粗糙度"),
    "TH_DIMROUGH": ("roughness", "表面粗糙度"),
    "ZWMSURFSYM": ("roughness", "表面粗糙度"),
    "YkyCucao": ("roughness", "表面粗糙度候选"),
    "TH_DimWeld": ("welding", "焊接符号"),
    "ZWMWELDING": ("welding", "焊接符号"),
    "TH_TOLERANCEENT": ("gdt", "形位公差"),
    "YkyXingWei": ("gdt", "形位公差候选"),
    "TH_DIMBASEA": ("datum", "基准标注"),
    "YkyJiZhun": ("datum", "基准标注候选"),
    "TH_DIMCHAMFERUA": ("chamfer", "倒角标注"),
    "ZWMCHAMFER": ("chamfer", "倒角标注"),
    "ZWMSECTIONSYM": ("section", "剖切标识"),
    "ZWMSECTIONSYMLABEL": ("section", "剖切标识"),
    "ZWMVIEWDIRECTION": ("section", "向视标识"),
    "ZWMVIEWDIRECTIONLABEL": ("section", "向视标识"),
}
CONTEXT_OBJECTS = frozenset(
    {
        "TDMFRAME",
        "TDMTITLE",
        "TDMCELL",
        "TH_WaterMark",
        "WIPEOUT",
        "ZWMBALLOON",
        "TH_XUHAOENTITY2DXF",
    }
)


def recover_polygons(raw: dict, observations: list[dict]) -> dict[str, list[dict]]:
    """Recover original POLYGON vertices lost by the old preview serialization.

    No symbol shape is drawn from a template. Each triangle/polygon comes from
    its exact binary command, with a byte hash and frame/source reference.
    """
    from research.dwg_route_validation_20261015.parser_probe import (
        handle_value,
        hex_handle,
    )
    from research.dwg_route_validation_20261015.title_proxy_probe import (
        decode_hex_preview,
    )

    import hashlib

    result = {}
    for observation in observations:
        preview = observation.get("decoded_preview", {})
        frames = [
            r
            for r in preview.get("entities", [])
            if r["type"] == "HATCH"
            and r.get("opcode") == 7
            and not r.get("matrix_stack")
        ]
        if not frames:
            continue
        ref = observation["source_ref"]
        obj = raw["OBJECTS"][int(ref["json_pointer"].rsplit("/", 1)[1])]
        if hex_handle(handle_value(obj.get("handle"))) != ref["native_handle"]:
            raise ValueError("Native polygon object identity differs")
        data = decode_hex_preview(obj["preview"], obj["preview_size"])
        paths = []
        for row in frames:
            offset, size = row["byte_offset"], row["byte_size"]
            command = data[offset : offset + size]
            if len(command) != size or struct.unpack_from("<II", command) != (size, 7):
                raise ValueError("Native polygon frame differs")
            count = struct.unpack_from("<I", command, 8)[0]
            if count < 3 or len(command) != 12 + count * 24:
                raise ValueError("Unsupported native polygon payload")
            points = [
                list(struct.unpack_from("<3d", command, 12 + ordinal * 24))
                for ordinal in range(count)
            ]
            if not all(math.isfinite(v) for point in points for v in point):
                raise ValueError("Non-finite native polygon")
            paths.append(
                {
                    "points": points,
                    "frame_index": row["frame_index"],
                    "native_kind": "POLYGON",
                    "filled": True,
                    "display_only": True,
                    "source_ref": ref,
                    "byte_offset": offset,
                    "byte_size": size,
                    "frame_sha256": hashlib.sha256(command).hexdigest(),
                }
            )
        result[ref["native_handle"]] = paths
    return result


def literal_text(value: str) -> str:
    """Display AutoCAD formatting; the unchanged raw string stays in every field."""
    return (
        plain_mtext(value)
        .replace("%%c", "Ø")
        .replace("%%C", "Ø")
        .replace("%%p", "±")
        .replace("%%P", "±")
        .replace("%%d", "°")
        .replace("%%D", "°")
    )


def _preview_paths(preview: dict) -> list[dict]:
    result = []
    for row in preview.get("entities", []):
        # A matrix stack needs a separately verified transform. Never ignore it.
        if row.get("matrix_stack"):
            continue
        points = [r["location"] for r in row.get("vertices", [])]
        attributes = row.get("attributes", {})
        if row.get("type") in {"CIRCLE", "ARC"} and attributes.get("radius", 0) > 0:
            normal = attributes.get("extrusion", [0.0, 0.0, 1.0])
            if normal != [0.0, 0.0, 1.0]:
                continue
            center, radius = attributes["center"], attributes["radius"]
            # ezdxf ARC attributes use degrees. This is display tessellation only.
            start = (
                math.radians(attributes.get("start_angle", 0))
                if row["type"] == "ARC"
                else 0.0
            )
            sweep = (
                ((math.radians(attributes["end_angle"]) - start) % math.tau)
                if row["type"] == "ARC"
                else math.tau
            )
            points = [
                [
                    center[0] + radius * math.cos(start + i * sweep / 48),
                    center[1] + radius * math.sin(start + i * sweep / 48),
                    center[2],
                ]
                for i in range(49)
            ]
        if points:
            result.append(
                {
                    "points": points,
                    "frame_index": row["frame_index"],
                    "native_kind": row["type"],
                    "display_only": True,
                }
            )
    return result


def build_symbols(
    data: dict, index: GeometryIndex, native_paths: dict | None = None
) -> list[dict]:
    symbols = []
    for row in data.get("native_symbols", []):
        native_kind = row["native_kind"]
        if native_kind in CONTEXT_OBJECTS:
            continue
        category, label = SYMBOL_TYPES.get(
            native_kind, ("unclassified", "其他原生标注")
        )
        preview = row.get("decoded_preview", {})
        paths = _preview_paths(preview) + (native_paths or {}).get(
            row["source_ref"]["native_handle"], []
        )
        texts = []
        for text in preview.get("texts", []):
            if text.get("matrix_stack"):
                continue
            raw = text.get("raw_text", "")
            point = text.get("wcs_insert") or text.get("raw_start_point")
            texts.append(
                {
                    "raw_text": raw,
                    "text": literal_text(raw),
                    "point": point,
                    "height": text.get("attributes", {}).get("height"),
                    "frame_index": text["frame_index"],
                    "confidence": None,
                }
            )
        points = [p for path in paths for p in path["points"]] + [
            t["point"] for t in texts if t["point"]
        ]
        if not points and not texts:
            continue  # Preserved separately as unavailable; never an empty engineering card.
        contacts = []
        for path in paths:
            if path["native_kind"] != "POLYLINE" or len(path["points"]) < 2:
                continue
            if math.dist(path["points"][0], path["points"][-1]) < 1e-7:
                continue  # A GD&T frame is not an arrow or leader tip.
            for slot, point in ((0, path["points"][0]), (-1, path["points"][-1])):
                ids = index.contacts(point)
                if ids:
                    contacts.append(
                        {
                            "point": point,
                            "geometry_ids": ids,
                            "frame_index": path["frame_index"],
                            "endpoint_slot": slot,
                            "basis": "finite_native_contact",
                            "semantic_arrowtip_confirmed": False,
                        }
                    )
        literals = [t["text"] for t in texts if t["text"]]
        parameters = []
        for text in texts:
            value = text["text"].strip()
            if category == "chamfer" and re.fullmatch(r"[Cc]\s*\d+(?:\.\d+)?", value):
                parameters.append(
                    {
                        "kind": "chamfer_literal",
                        "value": value,
                        "unit": None,
                        "frame_index": text["frame_index"],
                        "angle_inferred": False,
                    }
                )
            elif category == "gdt" and re.fullmatch(r"[Ø⌀]?\s*\d+(?:\.\d+)?", value):
                parameters.append(
                    {
                        "kind": "tolerance_literal",
                        "value": value,
                        "unit": None,
                        "frame_index": text["frame_index"],
                        "characteristic_type": None,
                    }
                )
            elif category == "welding" and re.fullmatch(r"[nN]\s*=\s*\d+", value):
                parameters.append(
                    {
                        "kind": "repetition_literal",
                        "value": value,
                        "frame_index": text["frame_index"],
                        "operation_count_inferred": False,
                    }
                )
        symbols.append(
            {
                "id": "symbol:" + row["source_ref"]["native_handle"],
                "native_kind": native_kind,
                "category": category,
                "label": label,
                "source_ref": row["source_ref"],
                "texts": texts,
                "raw_text": " / ".join(literals),
                "paths": paths,
                "bbox": bbox(points),
                "geometry_contacts": contacts,
                "geometry_ids": sorted(
                    {i for r in contacts for i in r["geometry_ids"]}
                ),
                "parameters": parameters,
                "part_candidates": [],
                "confidence": None,
                "relation_confidence": None,
                "physical_owner_confirmed": False,
                "coordinate_status": "native_preview_to_WCS_not_independently_verified",
                "unsupported_command_count": len(
                    preview.get("unsupported_commands", [])
                ),
                "matrix_commands_present": bool(preview.get("matrix_records")),
                "unrendered_graphic_entities": sum(
                    r["type"] != "TEXT"
                    and r["frame_index"] not in {p["frame_index"] for p in paths}
                    for r in preview.get("entities", [])
                ),
                "hard_constraint_eligible": False,
            }
        )
    return symbols
