"""Render native DWG geometry and stored custom-object graphics to vector PDF.

This adapter repairs converter text ordering from the native response. It never
uses the companion PDF, OCR, route labels, or synthetic geometry as an input.
"""

from __future__ import annotations

from collections import Counter
from copy import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time

import ezdxf
from ezdxf import bbox
from ezdxf.addons.drawing import Frontend, RenderContext, layout, pymupdf
from ezdxf.addons.drawing.config import BackgroundPolicy, ColorPolicy, Configuration
from ezdxf.math import BoundingBox2d
from ezdxf.fonts import fonts
from ezdxf.proxygraphic import ProxyGraphic
from ezdxf.tools.text import plain_mtext
import pymupdf as fitz

from research.dwg_route_validation_20261015.duplicate_audit import lossless_projection
from research.dwg_route_validation_20261015.native_layer_roles import _field_evidence
from research.dwg_route_validation_20261015.title_proxy_probe import (
    CODEPAGE_ENCODINGS, decode_hex_preview,
    validate_frames,
)
from research.dwg_pdf_visualization_20261031.line_colors import ColoredFrontend, NativeLineColorizer
from research.dwg_pdf_visualization_20261031.common import digest, work_dir

UNICODE_ESCAPE = re.compile(r"\\U\+([0-9a-fA-F]{4})")
FORMAT_CODE = re.compile(r"\\[A-Za-z][^;{}]*;")


def native_handle(obj: dict) -> str:
    return f"{obj['handle'][-1]:X}"


def decoded_unicode(text: str) -> str:
    return UNICODE_ESCAPE.sub(lambda match: chr(int(match[1], 16)), text)


def preview_display_text(raw: str) -> str:
    """Strip display control codes without combining or inventing fragments.

    The preview contains both original rich TEXT and separately drawn fragments.
    Keep only the prefix of rich TEXT; retain its separately positioned fragments.
    Raw strings and byte positions remain in the native evidence, unchanged.
    """
    prefix = raw.split("{\\", 1)[0]
    prefix = FORMAT_CODE.sub("", prefix).replace("\\P", " ")
    return decoded_unicode(prefix.replace("{", "").replace("}", "")).strip()


def entity_bounds(entities: list) -> list[float] | None:
    extent = bbox.extents(entities, fast=False)
    if not extent.has_data:
        return None
    return [extent.extmin.x, extent.extmin.y, extent.extmax.x, extent.extmax.y]


def display_wrap(text: str, height: float, width: float) -> str:
    """Insert display-only paragraph breaks using the substituted font metrics."""
    if width == 0:
        return text  # Native MTEXT auto width has no wrapping boundary.
    if width < 0:
        raise ValueError("MTEXT width cannot be negative")
    font = fonts.make_font("NotoSansCJK-Regular.ttc", height, width_factor=0.7)
    result = []
    for paragraph in plain_mtext(text).splitlines():
        line = ""
        for char in paragraph:
            if line and font.text_width(line + char) > width * 0.96:
                result.append(line)
                line = ""
            line += char
        result.append(line)
    return "\\P".join(result)


def drawing_bounds(entities: list, explicit: tuple[float, float, float, float] | None) -> list[float]:
    """Use an explicit legacy crop or all available modelspace extents."""
    values = list(explicit) if explicit is not None else entity_bounds(entities)
    if (values is None or len(values) != 4 or not all(math.isfinite(v) for v in values)
            or values[2] <= values[0] or values[3] <= values[1]):
        raise ValueError("A finite, nonempty drawing extent is required")
    return values


def restore_mtext_background(obj: dict, entity, pointer: str) -> list[dict]:
    """Restore supported mask fields from the native object, never a fallback guess."""
    repairs = []
    for source_field, display_field in (("bg_fill_flag", "bg_fill"),
                                         ("bg_fill_scale", "box_fill_scale"),
                                         ("bg_fill_trans", "bg_fill_transparency")):
        evidence = _field_evidence(obj, source_field, pointer)
        if evidence["conflicting"]:
            raise ValueError(f"Conflicting native MTEXT background: {pointer}/{source_field}")
        if not evidence["present_in_projection"]:
            continue
        value = evidence["raw_value"]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or (source_field == "bg_fill_scale" and not 1 <= value <= 5)
                or (source_field != "bg_fill_scale" and (type(value) is not int or value < 0))
                or (source_field == "bg_fill_flag" and value & ~19)):
            raise ValueError(f"Invalid native MTEXT background: {pointer}/{source_field}")
        converted = entity.dxf.get(display_field)
        if converted != value:
            repairs.append({"handle": entity.dxf.handle, "native_field": source_field,
                            "display_field": display_field, "converted": converted,
                            "native": value, "source_pointer": evidence["source_pointer"]})
            entity.dxf.set(display_field, value)
    if entity.dxf.get("bg_fill", 0) and not 1 <= entity.dxf.get("box_fill_scale", 1.5) <= 5:
        raise ValueError("Invalid converted MTEXT mask has no usable native replacement")
    return repairs


def pixel_audit(page) -> dict:
    """Reject solid black/white output even when a PDF has many vector paths."""
    pix = page.get_pixmap(matrix=fitz.Matrix(0.35, 0.35), colorspace=fitz.csGRAY, alpha=False)
    histogram = Counter(pix.samples)
    non_white = sum(count for value, count in histogram.items() if value < 245)
    fraction = non_white / (pix.width * pix.height)
    if min(histogram) >= 250 or max(histogram) <= 5 or not 1e-5 < fraction < 0.98:
        raise ValueError("Degenerate black/blank PDF raster; export requires review")
    return {"width": pix.width, "height": pix.height, "grayscale_min": min(histogram),
            "grayscale_max": max(histogram), "non_white_fraction": fraction,
            "passed_nonblank_check": True, "semantic_completeness_confirmed": False}


def black_entity(_entity, properties) -> None:
    properties.color = "#000000"


class EvidenceFrontend(ColoredFrontend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped: list[dict] = []

    def skip_entity(self, entity, msg: str) -> None:
        self.skipped.append({"handle": entity.dxf.handle, "kind": entity.dxftype(), "reason": msg})
        super().skip_entity(entity, msg)


def render(source: Path, response: Path, dxf: Path, output: Path, *, color_lines: bool = False,
           cad_bounds: tuple[float, float, float, float] | None = (0, 0, 4752, 3360),
           filename_stem: str = "12502370454", audit_entities: bool = False) -> dict:
    started = time.perf_counter()
    if not filename_stem or Path(filename_stem).name != filename_stem or filename_stem in {".", ".."}:
        raise ValueError("Expected a plain drawing filename stem")
    source_hash = digest(source)
    response_hash, dxf_hash = digest(response), digest(dxf)
    raw_response = response.read_bytes()
    if response.suffix == ".gz":
        raw_response = gzip.decompress(raw_response)
    parsed, duplicate_events = lossless_projection(raw_response)
    doc = ezdxf.readfile(dxf)
    block_repairs = []
    for obj in parsed["OBJECTS"]:
        reference = obj.get("block_header")
        if obj.get("entity") != "INSERT" or not reference:
            continue
        entity = doc.entitydb.get(native_handle(obj))
        block_record = doc.entitydb.get(f"{reference[-1]:X}")
        if entity is None or block_record is None or block_record.dxftype() != "BLOCK_RECORD":
            continue
        if entity.dxf.name != block_record.dxf.name:
            block_repairs.append({"handle": entity.dxf.handle,
                                  "native_entmode": obj.get("entmode"),
                                  "source_block_handle": f"{reference[-1]:X}",
                                  "converted_name": entity.dxf.name,
                                  "repaired_name": block_record.dxf.name,
                                  "basis": "exact_native_block_header_handle"})
            entity.dxf.name = block_record.dxf.name
    auditor = doc.audit()
    if auditor.has_errors:
        raise ValueError(f"Converted drawing has {len(auditor.errors)} DXF errors")
    msp = doc.modelspace()
    conventional_counts = Counter(entity.dxftype() for entity in msp)
    corrections: list[dict] = []
    background_repairs: list[dict] = []

    # DWG->DXF placed long MTEXT continuation chunks before the initial chunk.
    # Replace all matching text from the independently parsed native object.
    for index, obj in enumerate(parsed["OBJECTS"]):
        if "handle" not in obj:
            continue
        handle = native_handle(obj)
        entity = doc.entitydb.get(handle)
        if entity is None or entity.dxftype() not in {"TEXT", "MTEXT"}:
            continue
        raw = obj.get("text")
        if not isinstance(raw, str):
            continue
        if entity.dxf.text != raw:
            corrections.append({"handle": handle, "kind": "native_text_order_restore",
                                "converted": entity.dxf.text, "native": raw})
            entity.dxf.text = raw
        if audit_entities and entity.dxftype() == "MTEXT":
            background_repairs.extend(restore_mtext_background(obj, entity, f"#/OBJECTS/{index}"))

    classes = {entry["number"]: entry["dxfname"] for entry in parsed["CLASSES"]}
    proxies: list[dict] = []
    for index, obj in enumerate(parsed["OBJECTS"]):
        if (obj.get("entmode") != 2 or not isinstance(obj.get("preview"), str)
                or not obj.get("preview_size")):
            continue
        data = decode_hex_preview(obj["preview"], obj["preview_size"])
        frames = validate_frames(data)
        graphic = ProxyGraphic(data, doc=doc)
        encoding = CODEPAGE_ENCODINGS.get(parsed.get("FILEHEADER", {}).get("codepage"))
        if audit_entities and encoding is None:
            raise ValueError("Cannot render native preview with an unknown declared encoding")
        graphic.encoding = encoding if audit_entities else "gbk"
        entities = list(graphic.virtual_entities())
        original_texts = []
        for entity in entities:
            entity.dxf.layer = "0"
            if entity.dxftype() == "TEXT":
                raw = entity.dxf.text
                clean = preview_display_text(raw)
                original_texts.append({"raw_text": raw, "display_text": clean,
                                       "insert": list(entity.dxf.insert)})
                entity.dxf.text = clean
            msp.add_entity(entity)
        unsupported = Counter(frame.name for frame in frames
                              if getattr(graphic, frame.name.lower(), None) is None)
        # CTB/STB plot-style references are immaterial to an explicit black render.
        unrendered_geometry = set(unsupported) - {"ATTRIBUTE_PLOT_STYLE_NAME"}
        if unrendered_geometry:
            raise ValueError(f"Unrendered proxy geometry: {unrendered_geometry}")
        proxies.append({"native_handle": native_handle(obj),
                        "native_kind": classes.get(obj.get("type"), str(obj.get("type"))),
                        "json_pointer": f"#/OBJECTS/{index}",
                        "preview_sha256": hashlib.sha256(data).hexdigest(),
                        "preview_size": len(data), "commands": len(frames),
                        "ignored_plot_style_commands": dict(unsupported),
                        "unsupported_geometry_commands": [],
                        "emitted_entity_handles": [entity.dxf.handle for entity in entities],
                        "primitive_count": len(entities), "original_texts": original_texts,
                        "cad_bbox": entity_bounds(entities),
                        "coordinate_status": "native_preview_xy_identity_for_display_unverified_WCS",
                        "physical_owner": "unknown"})

    for block in doc.blocks:
        for entity in block:
            if entity.dxftype() in {"TEXT", "MTEXT"}:
                entity.dxf.text = decoded_unicode(entity.dxf.text)
                # Caret-stack syntax needs its separating space after the caret.
                if entity.dxftype() == "MTEXT":
                    entity.dxf.text = entity.dxf.text.replace("\\S 0^-0.1;", "\\S0^ -0.1;")
    for style in doc.styles:
        style.dxf.font = "NotoSansCJK-Regular.ttc"
        style.dxf.bigfont = ""

    for obj in proxies:
        entities = [doc.entitydb[handle] for handle in obj["emitted_entity_handles"]]
        obj["native_preview_bbox_before_font_substitution"] = obj["cad_bbox"]
        obj["cad_bbox"] = entity_bounds(entities)

    # A substituted CJK font has different metrics from the source SHX font.
    # Reflow only the displayed MTEXT to its native rectangle; raw text is kept.
    reflowed: list[dict] = []
    for entity in msp.query("MTEXT"):
        text = entity.dxf.text
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            raw_height = entity.dxf.char_height
            entity.dxf.char_height = raw_height * 0.7
            wrapped = display_wrap(text, entity.dxf.char_height, entity.dxf.width)
            entity.dxf.text = "{\\W0.7;" + wrapped + "}"
            reflowed.append({"handle": entity.dxf.handle, "original_height": raw_height,
                             "display_height": entity.dxf.char_height,
                             "width_factor": 0.7, "reason": "missing_SHX_font_metric_adaptation"})

    # Pin all overlay coordinates to the actual renderer placement matrix.
    resolved_bounds = drawing_bounds(list(msp), cad_bounds)
    bounds = BoundingBox2d([resolved_bounds[:2], resolved_bounds[2:]])
    page = layout.Page(594, 420, units=layout.Units.mm, margins=layout.Margins.all(5))
    settings = layout.Settings(fit_page=True, output_layers=color_lines,
                               min_stroke_width=0.04, max_stroke_width=0.001)
    output_layout = layout.Layout(bounds, flip_y=True)
    resolved_page = output_layout.get_final_page(page, settings)
    placement_settings = copy(settings)
    placement_settings.output_coordinate_space = pymupdf.get_coordinate_output_space(resolved_page)
    matrix = output_layout.get_placement_matrix(resolved_page, placement_settings, top_origin=True)

    backend = pymupdf.PyMuPdfBackend()
    context = RenderContext(doc)
    frontend_type = ColoredFrontend if color_lines else EvidenceFrontend if audit_entities else Frontend
    frontend = frontend_type(context, backend,
                        config=Configuration(color_policy=ColorPolicy.COLOR if color_lines or audit_entities else ColorPolicy.BLACK,
                                             background_policy=BackgroundPolicy.WHITE,
                                             hatching_timeout=10))
    colorizer = NativeLineColorizer(doc, proxies) if color_lines else None
    if colorizer is not None:
        frontend.push_property_override_function(colorizer)
    elif audit_entities:
        # Apply black to entities, not the canvas-colored MTEXT background mask.
        frontend.push_property_override_function(black_entity)
    frontend.draw_layout(msp)
    pdf = backend.get_pdf_bytes(page, settings=settings, render_box=bounds)
    work_dir(output).mkdir(parents=True, exist_ok=True)
    suffix = "线条分类上色" if color_lines else "原生矢量"
    native_path = work_dir(output) / f"{filename_stem}_{suffix}.pdf"
    native_path.write_bytes(pdf)
    with fitz.open(native_path) as rendered:
        page_size = [rendered[0].rect.width, rendered[0].rect.height]
        vector_paths = len(rendered[0].get_drawings())
        bitmap_count = len(rendered[0].get_images())
        pixels = pixel_audit(rendered[0]) if audit_entities else None

    dimensions = []
    for entity in msp.query("DIMENSION"):
        block = doc.blocks.get(entity.dxf.geometry)
        text_entities = [member for member in block
                         if member.dxftype() in {"TEXT", "MTEXT"}]
        dimensions.append({"handle": entity.dxf.handle,
                           "display_literal": [member.dxf.text for member in text_entities],
                           "text_cad_bbox": entity_bounds(text_entities),
                           "dimension_cad_bbox": entity_bounds([entity])})

    manifest = {
        "schema": "dwg_vector_visualization.v1",
        "source": {"path": str(source), "sha256": source_hash},
        "native_response": {"path": str(response), "sha256": response_hash},
        "converter_dxf": {"path": str(dxf), "sha256": dxf_hash},
        "code_sha256": digest(Path(__file__)), "ezdxf_version": ezdxf.__version__,
        "pymupdf_version": fitz.VersionBind,
        "conventional_modelspace_entities": dict(conventional_counts),
        "proxy_objects": proxies, "restored_proxy_count": len(proxies),
        "restored_primitive_count": sum(item["primitive_count"] for item in proxies),
        "duplicate_event_count": len(duplicate_events),
        "dxf_audit_errors": len(auditor.errors), "dxf_audit_fixes": len(auditor.fixes),
        "block_reference_repairs": block_repairs,
        "text_order_corrections": corrections, "display_reflow": reflowed,
        "line_colors": colorizer.report() if colorizer is not None else None,
        "dimensions": dimensions,
        "cad_bounds": resolved_bounds, "cad_to_pdf_matrix44": list(matrix),
        "page_points": page_size, "vector_paths": vector_paths,
        "embedded_bitmap_count": bitmap_count,
        "render_elapsed_seconds": time.perf_counter() - started,
        "limitations": ["SHX字体缺失，显示使用Noto CJK并整理换行，非原CAD像素级打印复刻",
                        "插件对象使用原生预览图元，坐标按原生XY显示，WCS和物理部件归属未认证",
                        "区域放大窗口仅为展示裁剪，不代表自动检出的部件轮廓",
                        "不从图形推定完整制造工艺路线；未知字段和关系不补造"],
        "companion_pdf_used_as_render_input": False,
        "ocr_used": False, "gpu_used": False, "route_labels_read": False,
    }
    if audit_entities:
        extents, failures = {}, []
        for entity in msp:
            try:
                extent = entity_bounds([entity])
                if extent is not None and all(math.isfinite(v) for v in extent):
                    extents[entity.dxf.handle] = extent
                else:
                    failures.append({"handle": entity.dxf.handle, "reason": "no_finite_extent"})
            except (ezdxf.DXFError, ValueError, TypeError, ZeroDivisionError) as error:
                failures.append({"handle": entity.dxf.handle, "reason": type(error).__name__})
        manifest.update({"entity_cad_bounds": extents, "entity_extent_failures": failures,
                         "skipped_entities": getattr(frontend, "skipped", []),
                         "automatic_full_extent": cad_bounds is None,
                         "raw_response_sha256": hashlib.sha256(raw_response).hexdigest(),
                         "final_print_visibility_confirmed": False,
                         "mtext_background_repairs": background_repairs, "pixel_audit": pixels})
    if any(digest(path) != expected for path, expected in
           ((source, source_hash), (response, response_hash), (dxf, dxf_hash))):
        raise ValueError("Drawing input changed during rendering")
    manifest_name = "line_color_manifest.json" if color_lines else "render_manifest.json"
    # Preserve undecodable converter metadata as JSON escapes. The native raw
    # strings and rendered drawing are unchanged; replacing characters here
    # would lose the evidence needed to diagnose the source encoding.
    (work_dir(output) / manifest_name).write_bytes(
        json.dumps(manifest, ensure_ascii=False, indent=2).encode(
            "utf-8", errors="backslashreplace"))
    return manifest
