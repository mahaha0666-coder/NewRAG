"""PDF -> official Paddle evidence -> optional visual entity graph.

All model-derived semantics remain review candidates, never training labels.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from .balloon_binding import bind_bom_balloons
from .document_schema import (
    ATTRIBUTE_FIELDS, COLLECTIONS, METADATA_FIELDS, DrawingDocument, Entity, Location,
    Page, Relationship, SemanticProposal, stable_id,
)
from .document_tables import bom_rows, extract_cells, normalize_part_number, spatial_bom_rows
from .feature_memory import save_document
from .source_page_render import RIGHT_EDGE_RECOVERY, plan_source_page_render


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)


def page_point(x: float, y: float, frame: list, rotation: int) -> list[float]:
    if rotation == 0:
        return [x + frame[0], y + frame[1]]
    if rotation == 90:
        return [y + frame[0], frame[3] - x]
    raise ValueError("Unsupported recognition rotation")


def _clamp_layout_geometry(value: float, lower: float, upper: float,
                           tolerance: float = 16.0) -> tuple[float, bool]:
    """Repair small detector edge jitter without accepting corrupt geometry."""
    if lower <= value <= upper:
        return value, False
    if value < lower and lower - value <= tolerance:
        return lower, True
    if value > upper and value - upper <= tolerance:
        return upper, True
    return value, False


def _normalize_layout_geometry(bbox: list | None, polygon: list | None,
                               width: float, height: float) -> tuple[list | None, list | None, bool]:
    clamped = False
    if bbox:
        normalized = []
        for index, value in enumerate(bbox):
            limit = width if index % 2 == 0 else height
            value, changed = _clamp_layout_geometry(float(value), 0.0, limit)
            normalized.append(value)
            clamped |= changed
        bbox = normalized
        # Paddle occasionally collapses one side of a text polygon to a line.
        # Keep the OCR token, but give it the smallest valid pixel footprint.
        for start, limit in ((0, width), (1, height)):
            if bbox[start] < bbox[start + 2]:
                continue
            center = min(max(bbox[start], 0.0), limit)
            if center >= limit:
                bbox[start], bbox[start + 2] = max(0.0, limit - 1.0), limit
            else:
                bbox[start], bbox[start + 2] = center, min(limit, center + 1.0)
            clamped = True
    if polygon:
        normalized_polygon = []
        for point in polygon:
            x, changed_x = _clamp_layout_geometry(float(point[0]), 0.0, width)
            y, changed_y = _clamp_layout_geometry(float(point[1]), 0.0, height)
            normalized_polygon.append([x, y])
            clamped |= changed_x or changed_y
        polygon = normalized_polygon
    return bbox, polygon, clamped


def local_endpoint(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Only local HTTP model endpoints are allowed for private drawings")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Invalid endpoint")
    return value.rstrip("/")


def render_pdf(source: Path, output: Path, dpi: int = 200) -> DrawingDocument:
    import pymupdf
    if not 72 <= dpi <= 400:
        raise ValueError("DPI must be between 72 and 400")
    sha = digest(source)
    document = DrawingDocument(id=stable_id(sha, "drawing", None),
                               source_path=str(source.resolve()), source_sha256=sha)
    with pymupdf.open(source) as pdf:
        if pdf.needs_pass or not pdf.page_count:
            raise ValueError("Encrypted or empty PDF")
        for number, page in enumerate(pdf, 1):
            # Coordinates map to unrotated, crop-relative PyMuPDF page points.
            plan = plan_source_page_render(page)
            if plan.policy == RIGHT_EDGE_RECOVERY:
                page.set_mediabox(pymupdf.Rect(*plan.render_mediabox))
                document.issues.append({
                    "stage": "source_crop_recovery", "page": number,
                    "status": "review_required", "route_hard_constraint_eligible": False,
                    "source_mediabox": list(plan.source_mediabox),
                    "render_mediabox": list(plan.render_mediabox),
                    "outside_path_counts": plan.outside_path_counts,
                })
            matrix = pymupdf.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output / f"page_{number:04d}.png"
            pix.save(image_path)
            inverse = (~matrix) * page.derotation_matrix
            document.pages.append(Page(
                page_number=number, width=pix.width, height=pix.height,
                image_path=str(image_path.resolve()), image_sha256=digest(image_path),
                pdf_cropbox=list(plan.source_cropbox), pdf_rotation=page.rotation,
                pixel_to_pdf_matrix=list(inverse),
                render_policy=plan.policy,
                source_mediabox=list(plan.source_mediabox),
                render_mediabox=list(plan.render_mediabox),
            ))
    if digest(source) != sha:
        raise ValueError("Original PDF changed during rendering")
    return document


def add_bom_from_region(document: DrawingDocument, region: Entity):
    if region.attributes.get("label") != "table":
        return
    cells = extract_cells(region.raw_text or "")
    for row in bom_rows(cells):
        if row["values"].get("part_number"):
            row["values"]["part_number"] = normalize_part_number(row["values"]["part_number"])
        values = row["values"]
        page_number = region.locations[0].page_number
        core_fields = ("item_number", "part_number", "part_name", "quantity")
        same_item = [
            candidate for candidate in document.bom
            if candidate.locations[0].page_number == page_number
            and str(candidate.attributes.get("item_number")) == str(values.get("item_number"))
        ]
        matching = next((
            candidate for candidate in same_item
            if all(candidate.attributes.get(field) == values.get(field) for field in core_fields)
        ), None)
        if matching is not None:
            for field in ("material", "specification", "remark"):
                current, observed = matching.attributes.get(field), values.get(field)
                if current is None and observed is not None:
                    matching.attributes[field] = observed
                elif current is not None and observed is not None and current != observed:
                    document.issues.append({
                        "stage": "bom_ocr_conflict", "item_number": values.get("item_number"),
                        "fields": [field], "entity_ids": [matching.id],
                    })
            matching.evidence_ids = sorted(set(matching.evidence_ids + [region.id]))
            matching.raw_result_refs = sorted(set(matching.raw_result_refs + region.raw_result_refs))
            for location in region.locations:
                if location not in matching.locations:
                    matching.locations.append(location.model_copy(deep=True))
            matching.attributes["source_observation_count"] = (
                int(matching.attributes.get("source_observation_count", 1)) + 1
            )
            continue
        bom = Entity(
            id=stable_id(document.source_sha256, "bom", [page_number, values]),
            kind="bom", name=values.get("part_name"),
            raw_text=json.dumps(row["raw_cells"], ensure_ascii=False),
            attributes={**values, "source_row_index": row["row_index"],
                        "source_header_row_index": row.get("header_row_index"),
                        "source_parse_mode": row["parse_mode"],
                        "source_observation_count": 1},
            locations=[loc.model_copy(deep=True) for loc in region.locations],
            evidence_ids=[region.id], raw_result_refs=list(region.raw_result_refs),
            method=f"paddle_table_{row['parse_mode']}")
        if bom.id not in {e.id for e in document.bom}:
            document.bom.append(bom)
            if same_item:
                conflicting_fields = [
                    field for field in core_fields[1:]
                    if any(candidate.attributes.get(field) != values.get(field)
                           for candidate in same_item)
                ]
                document.issues.append({
                    "stage": "bom_ocr_conflict", "item_number": values.get("item_number"),
                    "fields": conflicting_fields,
                    "entity_ids": sorted([candidate.id for candidate in same_item] + [bom.id]),
                })


def add_bom_from_spatial_ocr(document: DrawingDocument, page: Page):
    """Recover repeated BOM rows when layout detection missed the table block."""
    items = []
    for entity in document.raw_ocr:
        for location in entity.locations:
            if location.page_number == page.page_number and location.bbox is not None:
                items.append({"text": entity.normalized_text or entity.raw_text or "",
                              "bbox": list(location.bbox), "entity_id": entity.id,
                              "raw_result_ref": entity.raw_result_refs[0]
                              if entity.raw_result_refs else None})
    for row in spatial_bom_rows(items):
        values = row["values"]
        # A code plus two numbers is not a BOM row.  Keep it as unresolved OCR
        # evidence so the targeted table pass can recover the missing name or
        # reject the candidate; never publish a structurally incomplete row.
        if not values.get("part_name") or not values.get("quantity"):
            continue
        matching = next((candidate for candidate in document.bom
                         if candidate.locations[0].page_number == page.page_number
                         and str(candidate.attributes.get("item_number")) == values["item_number"]
                         and normalize_part_number(str(candidate.attributes.get("part_number") or ""))
                         == values["part_number"]), None)
        if matching is not None:
            for field in ("part_name", "material", "specification", "remark"):
                if matching.attributes.get(field) is None and values.get(field) is not None:
                    matching.attributes[field] = values[field]
            matching.evidence_ids = sorted(set(matching.evidence_ids + row["evidence_ids"]))
            matching.raw_result_refs = sorted(set(matching.raw_result_refs + row["raw_result_refs"]))
            continue
        location = Location(page_number=page.page_number, bbox=tuple(row["bbox"]),
                            localization_source="unknown")
        bom = Entity(
            id=stable_id(document.source_sha256, "bom", [page.page_number, values]),
            kind="bom", name=values.get("part_name"),
            raw_text=json.dumps(row["raw_cells"], ensure_ascii=False),
            attributes={**values, "source_row_index": None,
                        "source_header_row_index": None,
                        "source_parse_mode": "ocr_spatial_signature",
                        "source_observation_count": 1},
            locations=[location.model_copy(deep=True)],
            evidence_ids=sorted(set(row["evidence_ids"])),
            raw_result_refs=sorted(set(row["raw_result_refs"])),
            method="paddle_ocr_spatial_signature")
        same_item = [candidate for candidate in document.bom
                     if candidate.locations[0].page_number == page.page_number
                     and str(candidate.attributes.get("item_number")) == values["item_number"]]
        document.bom.append(bom)
        if same_item:
            document.issues.append({
                "stage": "bom_ocr_conflict", "page": page.page_number,
                "item_number": values["item_number"],
                "fields": [field for field in ("part_number", "part_name", "quantity")
                           if any(candidate.attributes.get(field) != values.get(field)
                                  for candidate in same_item)],
                "entity_ids": [candidate.id for candidate in same_item] + [bom.id],
                "resolved": False,
            })


def add_paddle_result(document: DrawingDocument, page: Page, payload: dict, ref: str):
    results = payload.get("results") or []
    status = payload.get("status", "ok" if results else "failed")
    if not results or status == "failed":
        page.paddle_status = "failed"
        document.issues.append({"stage": "paddle", "page": page.page_number,
                                "error": payload.get("error", payload.get("errors", "Empty response")), "raw": ref})
        if not results:
            return
    elif status == "partial":
        page.paddle_status = "partial"
        document.issues.append({"stage": "paddle_partial", "page": page.page_number,
                                "error": payload.get("warnings", "Incomplete completion discarded"),
                                "raw": ref})
    elif status == "ok":
        page.paddle_status = "ok"
    else:
        page.paddle_status = "failed"
        document.issues.append({"stage": "paddle", "page": page.page_number,
                                "error": f"Unknown Paddle status: {status!r}", "raw": ref})
        return
    pass_ids = payload.get("pass_ids", [None] * len(results))
    if len(pass_ids) != len(results):
        raise ValueError("Paddle pass IDs do not match result count")
    seen = {e.id for e in document.entities()}
    for result_index, wrapped in enumerate(results):
        raw = wrapped.get("res", wrapped)
        frame = payload.get("frames", [[0, 0, page.width, page.height]] * len(payload["results"]))[result_index]
        rotation = payload.get("rotations", [0] * len(payload["results"]))[result_index]
        if not (0 <= frame[0] < frame[2] <= page.width and 0 <= frame[1] < frame[3] <= page.height):
            raise ValueError("Invalid crop frame")
        width, height = frame[2] - frame[0], frame[3] - frame[1]
        if rotation == 90:
            width, height = height, width
        elif rotation != 0:
            raise ValueError("Unsupported recognition rotation")
        if raw.get("width") != width or raw.get("height") != height:
            raise ValueError("Paddle output dimensions differ from rendered page")
        records = [("regions", b.get("block_content", ""), b.get("block_bbox"),
                    b.get("block_polygon_points"), {"label": b.get("block_label")},
                    f"{ref}#/results/{result_index}/res/parsing_res_list/{i}")
                   for i, b in enumerate(raw.get("parsing_res_list", []))]
        # Layout spotting may be block-local; only a dedicated full-page pass
        # supplies a documented page coordinate system for text polygons.
        spotting = raw.get("spotting_res") or []
        if raw.get("model_settings", {}).get("use_layout_detection", False):
            spotting = []
        spotting_is_dict = isinstance(spotting, dict)
        if isinstance(spotting, dict):
            spotting = [spotting]
        for group_index, group in enumerate(spotting):
            for i, (text, poly) in enumerate(zip(group.get("rec_texts", []),
                                                group.get("rec_polys", []))):
                xs, ys = zip(*poly)
                pointer = f"{ref}#/results/{result_index}/res/spotting_res"
                if not spotting_is_dict:
                    pointer += f"/{group_index}"
                records.append(("raw_ocr", text, [min(xs), min(ys), max(xs), max(ys)],
                                poly, {"localization": "text_polygon"}, pointer + f"/rec_texts/{i}"))
        records += [("raw_ocr", text, bbox, poly, {**attrs, "localization": "layout_block"}, pointer)
                    for kind, text, bbox, poly, attrs, pointer in records
                    if (kind == "regions" and text
                        and attrs.get("label") in {"text", "paragraph_title", "figure_title",
                                                   "vision_footnote", "aside_text", "number"})]
        for kind, text, bbox, poly, attrs, pointer in records:
            try:
                if bbox:
                    corners = [page_point(x, y, frame, rotation)
                               for x in (bbox[0], bbox[2]) for y in (bbox[1], bbox[3])]
                    xs, ys = zip(*corners)
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                if poly:
                    poly = [page_point(x, y, frame, rotation) for x, y in poly]
                bbox, poly, geometry_clamped = _normalize_layout_geometry(
                    bbox, poly, page.width, page.height)
                attrs = {**attrs, "recognition_rotation": rotation}
                if geometry_clamped:
                    attrs["geometry_edge_clamped"] = True
                if pass_ids[result_index] is not None:
                    attrs["recognition_pass"] = pass_ids[result_index]
                if attrs.get("label") == "spotting":
                    attrs = {**attrs, "localization": "input_window"}
                loc = Location(page_number=page.page_number, bbox=bbox, polygon=poly,
                               localization_source=("unknown" if attrs.get("localization") == "input_window"
                                                    else "paddle_spotting" if attrs.get("localization") == "text_polygon"
                                                    else "paddle_layout"))
                identity = [page.page_number, text, bbox, poly, attrs]
                entity = Entity(id=stable_id(document.source_sha256, kind, identity),
                                kind=kind, raw_text=text,
                                normalized_text=unicodedata.normalize("NFKC", text),
                                locations=[loc], attributes=attrs,
                                raw_result_refs=[pointer], method="PaddleOCR-VL-1.6")
                if entity.id not in seen:
                    getattr(document, kind).append(entity)
                    try:
                        DrawingDocument.model_validate(document.model_dump())
                    except ValueError:
                        getattr(document, kind).pop()
                        raise
                    seen.add(entity.id)
                    if kind == "regions" and attrs.get("label") == "table":
                        add_bom_from_region(document, entity)
            except (ValueError, TypeError) as exc:
                document.issues.append({"stage": "paddle_geometry", "raw": pointer,
                                        "error": str(exc)})


def apply_semantics(document: DrawingDocument, page: Page, proposal: dict, ref: str,
                    verify_pixels: bool = False):
    proposal = SemanticProposal.model_validate(proposal).model_dump()
    pixels = None
    if verify_pixels:
        from PIL import Image
        with Image.open(page.image_path) as source:
            pixels = source.convert("L")
        if pixels.size != (page.width, page.height):
            raise ValueError("Semantic image size differs from page contract")
    known = {e.id for e in document.entities()}
    mapping = {e.id: e.id for e in document.entities() if e.kind not in {"raw_ocr", "regions", "metadata"}}
    entries = [(item, False) for item in proposal.get("entities", [])] + [
        (item, True) for item in proposal.get("metadata", [])]
    for item, metadata in entries:
        try:
            kind = "metadata" if metadata else item["kind"]
            if (metadata and item.get("value") is None and item.get("bbox") is None
                    and not item.get("evidence_ids") and not item.get("raw_text")):
                continue
            if not metadata and kind not in set(COLLECTIONS) - {"raw_ocr", "regions"}:
                raise ValueError("Unknown semantic entity kind")
            if metadata and item["field"] not in METADATA_FIELDS:
                item = {**item, "value": {item["field"]: item.get("value")}, "field": "other_fields"}
            if not metadata and (not item.get("id") or item["id"] in mapping):
                raise ValueError("Missing or duplicate local entity ID")
            evidence = sorted(set(item.get("evidence_ids", [])))
            if not set(evidence) <= known:
                raise ValueError("Unknown evidence ID")
            loc = Location(page_number=page.page_number, bbox=item.get("bbox"),
                           localization_source="visual_estimate")
            if not evidence and loc.bbox is None:
                raise ValueError("No localized visual or OCR evidence")
            if pixels is not None and loc.bbox is not None:
                extrema = pixels.crop(loc.bbox).getextrema()
                if extrema is None or extrema[0] >= 250:
                    raise ValueError("Proposed visual region contains no visible ink")
            attrs = ({"field": item["field"], "value": item.get("value")}
                     if metadata else item.get("attributes", {}))
            if not metadata:
                attrs = {**dict.fromkeys(ATTRIBUTE_FIELDS.get(kind, ())), **attrs}
            if kind == "balloons" and attrs.get("item_number") is None:
                label = (item.get("raw_text") or "").strip()
                if label.isdigit():
                    attrs["item_number"] = label
            if kind == "bom":
                matches = [e for e in document.bom if e.method.startswith("paddle_table_")
                           and e.locations[0].page_number == page.page_number
                           and str(e.attributes["item_number"]) == str(attrs.get("item_number"))]
                if len(matches) == 1:
                    existing = matches[0]
                    existing.attributes.setdefault("visual_candidates", []).append(attrs)
                    existing.raw_result_refs.append(ref)
                    conflicts = [key for key in ATTRIBUTE_FIELDS["bom"]
                                 if attrs.get(key) is not None and str(attrs[key]) != str(existing.attributes.get(key))]
                    if conflicts:
                        document.issues.append({"stage": "bom_conflict", "entity_id": existing.id,
                                                "fields": conflicts, "raw": ref})
                    mapping[item["id"]] = existing.id
                    continue
            identity = [page.page_number, loc.bbox, evidence, attrs, item.get("raw_text"), item.get("name")]
            entity = Entity(id=stable_id(document.source_sha256, kind, identity), kind=kind,
                            name=item.get("name"), raw_text=item.get("raw_text"),
                            normalized_text=item.get("normalized_text"), attributes=attrs,
                            locations=[loc], evidence_ids=evidence, raw_result_refs=[ref],
                            method="visual_semantic_candidate")
            collection = (document.drawing_metadata[item["field"]].candidates if metadata
                          else getattr(document, kind))
            if entity.id not in {e.id for e in document.entities()}:
                collection.append(entity)
                try:
                    DrawingDocument.model_validate(document.model_dump())
                except ValueError:
                    collection.pop()
                    raise
            if not metadata:
                mapping[item["id"]] = entity.id
        except (ValueError, KeyError, TypeError) as exc:
            document.issues.append({"stage": "semantic_entity", "raw": ref,
                                    "error": str(exc), "rejected": item})
    for item in proposal.get("relationships", []):
        try:
            source, target = mapping[item["source_id"]], mapping[item["target_id"]]
            evidence = sorted(set(item.get("evidence_ids", [])))
            source_entity = next(entity for entity in document.entities() if entity.id == source)
            source_location = next(
                (location for location in source_entity.locations
                 if location.page_number == page.page_number and location.bbox is not None),
                None,
            )
            relation = Relationship(
                id=stable_id(document.source_sha256, "relation", [source, target, item["type"]]),
                source_id=source, target_id=target, type=item["type"],
                evidence_ids=evidence, raw_result_refs=[ref],
                page=source_location.page_number if source_location is not None else page.page_number,
                source_bbox=source_location.bbox if source_location is not None else None,
                scope_status=str(item.get("scope_status") or "unassigned"),
                decision_status="supported" if evidence else "unresolved",
                method="focused_vlm_candidate",
                basis=item.get("basis"),
                needs_review=True,
            )
            if relation.id in {r.id for r in document.relationships}:
                continue
            document.relationships.append(relation)
            try:
                DrawingDocument.model_validate(document.model_dump())
            except ValueError:
                document.relationships.pop()
                raise
        except (ValueError, KeyError, TypeError) as exc:
            document.issues.append({"stage": "semantic_relationship", "raw": ref,
                                    "error": str(exc), "rejected": item})
    # Conflicting candidates are retained, never silently resolved by majority.
    for key, field in document.drawing_metadata.items():
        values = {json.dumps(e.attributes["value"], sort_keys=True) for e in field.candidates
                  if e.attributes["value"] is not None}
        field.value = json.loads(next(iter(values))) if len(values) == 1 else None
        decoded = [json.loads(value) for value in values]
        if key == "other_fields" and decoded and all(isinstance(value, dict) for value in decoded):
            candidates = {}
            for value in decoded:
                for name, entry in value.items():
                    candidates.setdefault(name, set()).add(json.dumps(entry, sort_keys=True))
            field.value = {name: json.loads(next(iter(entries))) if len(entries) == 1 else None
                           for name, entries in candidates.items()}
    annotation_views = {}
    for relation in document.relationships:
        if relation.type == "annotation_in":
            annotation_views.setdefault(relation.source_id, set()).add(relation.target_id)
    views = {view.id: view for view in document.views}
    for entity in document.entities():
        targets = annotation_views.get(entity.id, set())
        if targets:
            for location in entity.locations:
                location.view_id = None
        if len(targets) == 1:
            target = next(iter(targets))
            view_pages = {loc.page_number for loc in views[target].locations}
            for location in entity.locations:
                if location.page_number in view_pages:
                    location.view_id = target


def parse_drawing(source: Path, output: Path, paddle_python: Path,
                  paddle_endpoint: str, semantic_endpoint: str | None = None,
                  semantic_model: str | None = None, dpi: int = 200,
                  semantic_max_tokens: int = 1800,
                  semantic_timeout: int = 600, *, feature_memory: Path | None = None,
                  expected_snapshot: str | None = None,
                  paddle_max_tokens: int = 4096,
                  paddle_max_pass_attempts: int = 1,
                  paddle_full_content_spotting: bool = False,
                  paddle_stream_guard: bool = False,
                  region_refinement: bool = False,
                  progress: Callable[[str, dict], None] | None = None) -> DrawingDocument:
    if expected_snapshot is not None and feature_memory is None:
        raise ValueError("expected_snapshot requires feature_memory")
    if not 1 <= paddle_max_pass_attempts <= 5:
        raise ValueError("paddle_max_pass_attempts must be between 1 and 5")
    paddle_stream_guard = paddle_stream_guard or region_refinement
    paddle_endpoint = local_endpoint(paddle_endpoint)
    parallel_endpoint = os.environ.get("CAD_PADDLE_PARALLEL_ENDPOINT") if paddle_full_content_spotting else None
    if parallel_endpoint:
        parallel_endpoint = local_endpoint(parallel_endpoint)
        if parallel_endpoint == paddle_endpoint:
            raise ValueError("Parallel Paddle endpoint must be distinct from the primary endpoint")
    if semantic_endpoint:
        local_endpoint(semantic_endpoint)
        if not semantic_model:
            raise ValueError("semantic_model is required")
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    if progress:
        progress("source_render", {})
    document = render_pdf(source, output, dpi)
    render_seconds = time.monotonic() - started
    document.configuration = {"paddle_model": "PaddleOCR-VL-1.6", "dpi": dpi,
                              "semantic_model": semantic_model, "unwarping": False,
                              "paddle_passes": "full_layout+dynamic_high_risk_and_uncovered_ink_regions",
                              "paddle_max_tokens": paddle_max_tokens,
                              "paddle_generation_policy": (
                                  "region_ledger_v11" if paddle_full_content_spotting else "region_ledger_v6"
                              ),
                              "paddle_max_pass_attempts": paddle_max_pass_attempts,
                              "bom_recovery_policy": "ocr_spatial_signature_v1+independent_conflict_crop_v1",
                              "coordinate_definition": "unrotated crop-relative PDF points"}
    if paddle_stream_guard:
        from .stream_guard import POLICY
        document.configuration["paddle_stream_guard"] = POLICY
    if region_refinement:
        from .pdf_modality_audit import audit_pdf
        native_inventory = audit_pdf(source)
        native_inventory["verified_engineering_fields"] = False
        write_json(output / "native_inventory.json", native_inventory)
        document.configuration["native_inventory_ref"] = "native_inventory.json"
    if len(document.pages) > 1:
        document.issues.append({"stage": "cross_page_identity",
                                "error": "Cross-page physical identity requires review; no automatic name-based merge"})
    manifest = [{"image": p.image_path, "output": str((output / f"paddle_{p.page_number:04d}.json").resolve())}
                for p in document.pages]
    write_json(output / "pages.json", manifest)
    worker = Path(__file__).resolve().parents[3] / "scripts" / "paddle_document_worker.py"
    if progress:
        progress("layout_and_recognition", {"page_count": len(document.pages)})
    recognition_started = time.monotonic()
    with (output / "paddle_worker.log").open("x") as log:
        try:
            command = [str(paddle_python), str(worker), str(output / "pages.json"),
                       "--endpoint", paddle_endpoint,
                       "--max-new-tokens", str(paddle_max_tokens),
                       "--max-pass-attempts", str(paddle_max_pass_attempts)]
            if paddle_full_content_spotting:
                command.append("--full-content-spotting")
            if paddle_stream_guard:
                command.append("--stream-guard")
            if parallel_endpoint:
                command.extend(["--parallel-endpoint", parallel_endpoint])
            run = subprocess.run(command,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 timeout=3600 * len(document.pages), check=False)
            if run.returncode:
                document.issues.append({"stage": "paddle_worker", "exit_code": run.returncode})
        except subprocess.TimeoutExpired:
            document.issues.append({"stage": "paddle_worker", "error": "timeout"})
    for page, record in zip(document.pages, manifest):
        path = Path(record["output"])
        payload = json.loads(path.read_text()) if path.exists() else {"status": "failed", "error": "Worker produced no result"}
        try:
            add_paddle_result(document, page, payload, path.name)
        except (ValueError, TypeError, KeyError) as exc:
            page.paddle_status = "failed"
            document.issues.append({"stage": "paddle_adapter", "error": str(exc), "raw": path.name})
        try:
            add_bom_from_spatial_ocr(document, page)
        except (ValueError, TypeError, KeyError) as exc:
            document.issues.append({"stage": "bom_spatial_fallback",
                                    "page": page.page_number, "error": str(exc)})
    recognition_seconds = time.monotonic() - recognition_started
    if region_refinement:
        from .region_understanding import refine
        refine(document, output, [paddle_endpoint, *([parallel_endpoint] if parallel_endpoint else [])], progress=progress)
    if progress:
        progress("engineering_relations", {})
    relations_started = time.monotonic()
    from .bom_part_verification import verify_bom_part_numbers
    verify_bom_part_numbers(document, output, paddle_python)
    from .technical_requirements import extract_ocr_technical_requirements
    extract_ocr_technical_requirements(document)
    try:
        bind_bom_balloons(document, source)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        document.issues.append({"stage": "balloon_binding", "error": str(exc)})
    if semantic_endpoint:
        from .focused_semantics import extract_focused_semantics
        extract_focused_semantics(document, output, semantic_endpoint, semantic_model,
                                  semantic_max_tokens, semantic_timeout)
    finalize_document(document)
    document.configuration["stage_timings"] = {
        "render_seconds": round(render_seconds, 3),
        "recognition_seconds": round(recognition_seconds, 3),
        "relations_seconds": round(time.monotonic() - relations_started, 3),
        "total_seconds": round(time.monotonic() - started, 3),
    }
    save_document(document, output, memory_path=feature_memory, expected_snapshot=expected_snapshot)
    from .manufacturing_ir import materialize_manufacturing_ir
    materialize_manufacturing_ir(document, output)
    write_json(output / "drawing.schema.json", DrawingDocument.model_json_schema())
    return document


def finalize_document(document: DrawingDocument):
    all_paddle = all(p.paddle_status == "ok" for p in document.pages)
    all_semantic = (all(p.semantic_status == "ok" for p in document.pages) and
                    document.configuration.get("semantic_scope") == "focused_view_local_v1")
    if document.pages and all_paddle and all_semantic and not document.issues:
        document.extraction_status = "extracted_unverified"
    else:
        document.extraction_status = "partial"
    for key in COLLECTIONS:
        document.coverage[key] = "extracted_unverified" if getattr(document, key) else "not_observed_not_proven_absent"
    for key in ("drawing_metadata", "relationships"):
        document.coverage[key] = "assessed_unverified" if all_semantic else "not_fully_assessed"
    tasks = [e for e in document.technical_notes if e.attributes.get("note_type") == "manufacturing_task_candidate"]
    linked = {r.source_id for r in document.relationships if r.type == "affects"}
    document.configuration["manufacturing_audit"] = {
        "task_candidates": len(tasks), "with_candidate_target_links": sum(e.id in linked for e in tasks),
        "unknown_responsibility": sum(e.attributes["responsibility"] == "unknown" for e in tasks),
        "unknown_phase": sum(e.attributes["phase"] == "unknown" for e in tasks),
        "physical_target_accuracy": None, "coverage_complete": False, "eligible_for_training": False,
        "zero_tasks_means": "not_observed_not_proven_unnecessary"}
    DrawingDocument.model_validate(document.model_dump())
