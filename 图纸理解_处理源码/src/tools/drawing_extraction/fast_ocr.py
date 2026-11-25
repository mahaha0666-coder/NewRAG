"""Use fast, localized OCR as drawing evidence without inventing relationships."""

from __future__ import annotations

import json
import math
import subprocess
import unicodedata
from pathlib import Path
from typing import Any

from .balloon_binding import bind_bom_balloons
from .adaptive_regions import adaptive_region_plan
from .document_parser import (
    _normalize_layout_geometry, add_bom_from_spatial_ocr,
    finalize_document, render_pdf, write_json,
)
from .document_schema import DrawingDocument, Entity, Location, Page, stable_id
from .document_tables import PART_NUMBER, localized_bom_rows, normalize_part_number, vertical_bom_rows
from .feature_memory import save_document
from .ocr_geometry import map_region_polygon
from .semantic_gate import build_semantic_tasks
from .technical_requirements import extract_ocr_technical_requirements


def _bbox_iou(first: list | tuple, second: list | tuple) -> float:
    x0 = max(float(first[0]), float(second[0]))
    y0 = max(float(first[1]), float(second[1]))
    x1 = min(float(first[2]), float(second[2]))
    y1 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    return intersection / max(1e-9, first_area + second_area - intersection)


def bom_recovery_regions(document: DrawingDocument, page: Page) -> list[dict[str, Any]]:
    """Target repeated part-number columns with unresolved BOM rows only."""
    resolved = {normalize_part_number(str(row.attributes.get("part_number") or ""))
                for row in document.bom}
    codes = []
    for entity in document.raw_ocr:
        box = next((loc.bbox for loc in entity.locations
                    if loc.page_number == page.page_number and loc.bbox), None)
        number = normalize_part_number(entity.raw_text or "")
        if (box and len(number) >= 6 and number[0].isalpha()
                and PART_NUMBER.fullmatch(number)):
            codes.append((number, box))
    groups: list[list[tuple[str, tuple[float, float, float, float]]]] = []
    for number, box in sorted(codes, key=lambda item: item[1][1]):
        vertical = box[3] - box[1] > 1.5 * (box[2] - box[0])
        primary = (box[1] + box[3]) / 2 if vertical else (box[0] + box[2]) / 2
        for group in groups:
            first_box = group[0][1]
            group_vertical = first_box[3] - first_box[1] > 1.5 * (first_box[2] - first_box[0])
            first_primary = ((first_box[1] + first_box[3]) / 2 if vertical
                             else (first_box[0] + first_box[2]) / 2)
            if vertical == group_vertical and abs(primary - first_primary) < 90:
                group.append((number, box))
                break
        else:
            groups.append([(number, box)])
    regions = []
    for group in groups:
        if len(group) < 2 or all(number in resolved for number, _ in group):
            continue
        vertical = group[0][1][3] - group[0][1][1] > 1.5 * (
            group[0][1][2] - group[0][1][0])
        xs = [value for _, box in group for value in (box[0], box[2])]
        ys = [value for _, box in group for value in (box[1], box[3])]
        frame = ([max(0, int(min(xs) - 90)), max(0, int(min(ys) - 120)),
                  min(page.width, int(max(xs) + 90)),
                  min(page.height, int(max(ys) + 360))] if vertical else
                 [max(0, int(min(xs) - 100)), max(0, int(min(ys) - 70)),
                  min(page.width, int(max(xs) + 500)),
                  min(page.height, int(max(ys) + 70))])
        if frame[2] <= frame[0] or frame[3] <= frame[1]:
            continue
        regions.append({"pass_id": f"bom_roi_{len(regions) + 1}",
                        "frame": frame, "rotation": 90 if vertical else 0,
                        "scale": 1.5,
                        "missing_part_numbers": sorted(set(number for number, _ in group) - resolved)})
    return regions[:4]


def unresolved_bom_candidates(document: DrawingDocument, page: Page) -> list[dict[str, Any]]:
    """Record recognizable part numbers that lack a complete BOM row."""
    resolved = {normalize_part_number(str(row.attributes.get("part_number") or ""))
                for row in document.bom}
    bom_boxes = [location.bbox for row in document.bom for location in row.locations
                 if location.page_number == page.page_number and location.bbox is not None]
    if bom_boxes:
        scope = (min(box[0] for box in bom_boxes) - 360,
                 min(box[1] for box in bom_boxes) - 360,
                 max(box[2] for box in bom_boxes) + 360,
                 max(box[3] for box in bom_boxes) + 360)
    else:
        scope = (0.20 * page.width, 0.35 * page.height,
                 0.98 * page.width, 0.98 * page.height)
    candidates = []
    for entity in document.raw_ocr:
        location = next((location for location in entity.locations
                         if location.page_number == page.page_number and location.bbox is not None), None)
        number = normalize_part_number(entity.raw_text or "")
        center = ((location.bbox[0] + location.bbox[2]) / 2,
                  (location.bbox[1] + location.bbox[3]) / 2) if location and location.bbox else None
        if (location is None or center is None or not (scope[0] <= center[0] <= scope[2]
                and scope[1] <= center[1] <= scope[3])
                or not PART_NUMBER.fullmatch(number)
                or sum(character.isalpha() for character in number) < 2
                or number in resolved):
            continue
        candidate = {"entity_id": entity.id, "part_number": number,
                     "entity_ids": [entity.id],
                     "bbox": list(location.bbox), "confidence": entity.confidence,
                     "raw_result_refs": list(entity.raw_result_refs)}
        candidate_center = ((candidate["bbox"][0] + candidate["bbox"][2]) / 2,
                            (candidate["bbox"][1] + candidate["bbox"][3]) / 2)
        candidate_height = candidate["bbox"][3] - candidate["bbox"][1]
        duplicate = next((existing for existing in candidates
                          if (_bbox_iou(existing["bbox"], candidate["bbox"]) >= 0.65
                              or (existing["part_number"] == number
                                  and abs(((existing["bbox"][0] + existing["bbox"][2]) / 2)
                                          - candidate_center[0])
                                  <= max(160.0, 2.0 * candidate_height)
                                  and abs(((existing["bbox"][1] + existing["bbox"][3]) / 2)
                                          - candidate_center[1])
                                  <= max(160.0, 2.0 * candidate_height)))), None)
        if duplicate is None:
            candidates.append(candidate)
        else:
            duplicate["entity_ids"] = sorted(set(duplicate["entity_ids"] + candidate["entity_ids"]))
            duplicate["raw_result_refs"] = sorted(set(duplicate["raw_result_refs"]
                                                        + candidate["raw_result_refs"]))
            if float(candidate["confidence"] or 0.0) > float(duplicate["confidence"] or 0.0):
                duplicate.update({key: candidate[key] for key in ("entity_id", "part_number", "bbox", "confidence")})
    return candidates


def add_localized_bom_rows(document: DrawingDocument, page: Page,
                           recognition_passes: set[str] | None = None) -> int:
    """Add only fully evidenced rows from a selected ROI, without sequence guessing."""
    items = []
    for entity in document.raw_ocr:
        if (recognition_passes is not None
                and not (recognition_passes & set(entity.attributes.get("recognition_passes") or
                                                 [entity.attributes.get("recognition_pass")]))):
            continue
        for location in entity.locations:
            if location.page_number == page.page_number and location.bbox is not None:
                items.append({"text": entity.raw_text or "", "bbox": list(location.bbox),
                              "score": entity.confidence or 0.0,
                              "entity_id": entity.id,
                              "raw_result_ref": entity.raw_result_refs[0]
                              if entity.raw_result_refs else None})
    added = 0
    rows = [*localized_bom_rows(items), *vertical_bom_rows(items)]
    for row in rows:
        values = row["values"]
        if any(str(existing.attributes.get("part_number") or "")
               == values["part_number"] for existing in document.bom):
            continue
        location = Location(page_number=page.page_number, bbox=tuple(row["bbox"]),
                            localization_source="paddle_spotting")
        document.bom.append(Entity(
            id=stable_id(document.source_sha256, "bom", [page.page_number, values]),
            kind="bom", name=values.get("part_name"),
            raw_text=json.dumps(row["raw_cells"], ensure_ascii=False),
            attributes={**values, "source_parse_mode": "localized_row_v1",
                        "source_observation_count": 1},
            locations=[location], evidence_ids=sorted(set(row["evidence_ids"])),
            raw_result_refs=sorted(set(row["raw_result_refs"])),
            method="paddle_ocr_localized_row_v1",
        ))
        same_item = [existing for existing in document.bom[:-1]
                     if existing.locations and existing.locations[0].page_number == page.page_number
                     and str(existing.attributes.get("item_number") or "")
                     == str(values.get("item_number") or "")]
        if same_item:
            entity_ids = sorted({existing.id for existing in same_item} | {document.bom[-1].id})
            fields = [field for field in ("part_number", "part_name", "quantity")
                      if any(existing.attributes.get(field) != values.get(field)
                             for existing in same_item)]
            key = (page.page_number, str(values.get("item_number") or ""), tuple(entity_ids))
            if fields and not any(issue.get("stage") == "bom_ocr_conflict"
                       and issue.get("page") == key[0]
                       and issue.get("item_number") == key[1]
                       and sorted(issue.get("entity_ids") or []) == entity_ids
                       for issue in document.issues):
                document.issues.append({"stage": "bom_ocr_conflict", "page": page.page_number,
                                        "item_number": values.get("item_number"),
                                        "fields": fields, "entity_ids": entity_ids,
                                        "resolved": False})
        added += 1
    return added


def add_fast_ocr_result(
    document: DrawingDocument, page: Page, payload: dict[str, Any], ref: str,
) -> None:
    """Reject untraceable pixels; preserve each surviving text and polygon."""
    if payload.get("schema_version") not in {"fast_ocr_page.v1", "fast_ocr_page.v2"}:
        raise ValueError("Unsupported fast OCR response")
    if (payload.get("schema_version") == "fast_ocr_page.v1"
            and payload.get("coverage_mode") == "full_page_tiles_v1"
            and any(frame[0] or frame[1] for frame in payload.get("tile_frames") or [])):
        raise ValueError("Legacy full-page tile coordinates require an audited origin replay")
    if (payload.get("schema_version") == "fast_ocr_page.v2"
            and (payload.get("coordinate_space") != "rendered_page_pixels"
                 or payload.get("geometry_version") != "all_passes_origin_rotation_scale_v2")):
        raise ValueError("Fast OCR page-coordinate provenance is missing")
    if (payload.get("image_sha256") != page.image_sha256
            or payload.get("image_size") != [page.width, page.height]):
        raise ValueError("Fast OCR result belongs to another rendered page")
    observations = payload.get("observations")
    if not isinstance(observations, list) or len(observations) > 20000:
        raise ValueError("Fast OCR observations are missing or unbounded")
    page.paddle_status = "ok" if observations else "partial"
    if not observations:
        document.issues.append({"stage": "fast_ocr_empty", "page": page.page_number, "raw": ref})
    for index, item in enumerate(observations):
        pointer = f"{ref}#/observations/{index}"
        if isinstance(item, dict) and item.get("text") == "" and item.get("score") == 0:
            continue
        try:
            text = item["text"]
            score = item["score"]
            polygon = item["polygon"]
            if (not isinstance(text, str) or not text.strip()
                    or not isinstance(score, (int, float)) or isinstance(score, bool)
                    or not math.isfinite(score) or not 0 <= score <= 1
                    or not isinstance(polygon, list) or len(polygon) < 3
                    or any(not isinstance(point, (list, tuple)) or len(point) != 2
                           or any(not isinstance(value, (int, float))
                                  or isinstance(value, bool) or not math.isfinite(value)
                                  for value in point) for point in polygon)):
                raise ValueError("Invalid OCR text, confidence or polygon")
            if payload.get("schema_version") == "fast_ocr_page.v2":
                jobs = {job["pass_id"]: job for job in payload.get("source_passes") or []}
                job = jobs.get(item.get("pass_id"))
                if job is None or job.get("response_status") != "complete" or not item.get("crop_polygon"):
                    raise ValueError("OCR observation lacks a completed source pass")
                expected = map_region_polygon(item["crop_polygon"], job["frame"], job["rotation"], job["scale"])
                if len(expected) != len(polygon) or any(
                    abs(actual - target) > 1e-6
                    for point, wanted in zip(polygon, expected, strict=True)
                    for actual, target in zip(point, wanted, strict=True)
                ):
                    raise ValueError("OCR page polygon differs from its source-pass transform")
            xs, ys = zip(*polygon, strict=True)
            bbox, polygon, clamped = _normalize_layout_geometry(
                [min(xs), min(ys), max(xs), max(ys)], polygon,
                page.width, page.height,
            )
            location = Location(
                page_number=page.page_number, bbox=bbox, polygon=polygon,
                localization_source="paddle_spotting",
            )
            normalized = unicodedata.normalize("NFKC", text)
            duplicate = next((existing for existing in document.raw_ocr
                              if existing.locations and existing.locations[0].page_number == page.page_number
                              and unicodedata.normalize("NFKC", existing.raw_text or "") == normalized
                              and existing.locations[0].bbox is not None
                              and _bbox_iou(existing.locations[0].bbox, bbox) >= 0.65), None)
            if duplicate is not None:
                passes = list(duplicate.attributes.get("recognition_passes") or
                              [duplicate.attributes.get("recognition_pass", "fast_full_page")])
                pass_id = item.get("pass_id", "fast_full_page")
                if pass_id not in passes:
                    passes.append(pass_id)
                duplicate.attributes["recognition_passes"] = passes
                duplicate.raw_result_refs = sorted(set(duplicate.raw_result_refs + [pointer]))
                continue
            entity = Entity(
                id=stable_id(document.source_sha256, "raw_ocr",
                             [page.page_number, index, text, bbox]),
                kind="raw_ocr", raw_text=text,
                normalized_text=normalized,
                confidence=float(score), confidence_source="PP-OCRv6_medium_rec",
                attributes={"localization": "text_polygon",
                            "recognition_pass": item.get("pass_id", "fast_full_page"),
                            "recognition_passes": [item.get("pass_id", "fast_full_page")],
                            **({"geometry_edge_clamped": True} if clamped else {})},
                locations=[location], raw_result_refs=[pointer],
                method="PP-OCRv6_small_det+PP-OCRv6_medium_rec",
            )
            document.raw_ocr.append(entity)
            DrawingDocument.model_validate(document.model_dump())
        except (KeyError, TypeError, ValueError) as error:
            if document.raw_ocr and document.raw_ocr[-1].raw_result_refs == [pointer]:
                document.raw_ocr.pop()
            document.issues.append({"stage": "fast_ocr_geometry", "page": page.page_number,
                                    "raw": pointer, "error": str(error)})
            page.paddle_status = "partial"


def parse_fast_drawing(
    source: Path, output: Path, ocr_python: Path, *, dpi: int = 200,
    limit_side_len: int = 2200, timeout: int = 240, device: str = "cpu",
) -> DrawingDocument:
    """Create a separately versioned candidate graph from fast OCR and PDF vectors."""
    if (not ocr_python.is_file() or not 1200 <= limit_side_len <= 4000
            or timeout < 1 or device not in {"cpu", "gpu:0"}):
        raise ValueError("Invalid fast OCR runtime configuration")
    output.mkdir(parents=True, exist_ok=False)
    document = render_pdf(source, output, dpi)
    document.configuration = {
        "ocr_engine": "PP-OCRv6_small_det+PP-OCRv6_medium_rec",
        "ocr_version": "fast_ocr_page.v2+full_page_tiles_v2", "dpi": dpi,
        "geometry_version": "all_passes_origin_rotation_scale_v2",
        "coverage_policy": "full_page_tiled_then_targeted_recovery",
        "text_det_limit_side_len": limit_side_len,
        "device": device,
        "semantic_scope": "not_requested",
        "coordinate_definition": "unrotated crop-relative PDF points",
        "bom_recovery_policy": "ocr_spatial_signature_v1",
    }
    worker = Path(__file__).resolve().parents[3] / "scripts/fast_ocr_worker.py"
    for page in document.pages:
        name = f"fast_ocr_{page.page_number:04d}.json"
        with (output / f"fast_ocr_{page.page_number:04d}.log").open("x") as log:
            try:
                subprocess.run(
                    [str(ocr_python), str(worker), page.image_path, str(output / name),
                     "--limit-side-len", str(limit_side_len), "--device", device],
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                page.paddle_status = "failed"
                document.issues.append({"stage": "fast_ocr", "page": page.page_number,
                                        "error": type(error).__name__,
                                        "raw": f"fast_ocr_{page.page_number:04d}.log"})
                continue
        try:
            payload = json.loads((output / name).read_text(encoding="utf-8"))
            add_fast_ocr_result(document, page, payload, name)
            add_bom_from_spatial_ocr(document, page)
            regions = bom_recovery_regions(document, page)
            if regions:
                plan = {"image_sha256": page.image_sha256,
                        "image_size": [page.width, page.height], "regions": regions}
                write_json(output / f"fast_ocr_regions_{page.page_number:04d}.plan.json", plan)
                recovery_name = f"fast_ocr_regions_{page.page_number:04d}.json"
                try:
                    with (output / f"fast_ocr_regions_{page.page_number:04d}.log").open("x") as log:
                        subprocess.run(
                            [str(ocr_python), str(worker), page.image_path,
                             str(output / recovery_name), "--limit-side-len", "1600",
                            "--regions-file",
                             str(output / f"fast_ocr_regions_{page.page_number:04d}.plan.json"),
                             "--device", device],
                            stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True,
                        )
                    recovered = json.loads((output / recovery_name).read_text(encoding="utf-8"))
                    add_fast_ocr_result(document, page, recovered, recovery_name)
                    add_bom_from_spatial_ocr(document, page)
                    added_rows = add_localized_bom_rows(
                        document, page,
                        {row["pass_id"] for row in regions})
                    document.configuration.setdefault("bom_roi_recovery", []).append({
                        "page": page.page_number,
                        "plan_ref": f"fast_ocr_regions_{page.page_number:04d}.plan.json",
                        "regions": len(regions), "observations": len(recovered["observations"]),
                        "bom_after": len(document.bom), "localized_rows_added": added_rows,
                        "status": "ok",
                    })
                except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
                    document.issues.append({"stage": "bom_roi_recovery", "page": page.page_number,
                                            "error": type(error).__name__,
                                            "raw": f"fast_ocr_regions_{page.page_number:04d}.log"})
            # A partial BOM is not sufficient evidence: tiled OCR can see
            # part codes while missing a quantity/name anchor at a tile edge.
            # Pay for the high-resolution table passes whenever any localized
            # candidate remains unresolved, not only when the first parser
            # produced zero rows.
            if not document.bom or unresolved_bom_candidates(document, page):
                adaptive = adaptive_region_plan(page.width, page.height)
                plan = {"image_sha256": page.image_sha256,
                        "image_size": [page.width, page.height], "regions": adaptive}
                plan_path = output / f"fast_ocr_adaptive_{page.page_number:04d}.plan.json"
                write_json(plan_path, plan)
                adaptive_name = f"fast_ocr_adaptive_{page.page_number:04d}.json"
                try:
                    with (output / f"fast_ocr_adaptive_{page.page_number:04d}.log").open("x") as log:
                        subprocess.run(
                            [str(ocr_python), str(worker), page.image_path,
                             str(output / adaptive_name), "--limit-side-len", "2400",
                             "--regions-file", str(plan_path), "--device", device],
                            stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True,
                        )
                    adaptive_payload = json.loads((output / adaptive_name).read_text(encoding="utf-8"))
                    add_fast_ocr_result(document, page, adaptive_payload, adaptive_name)
                    add_bom_from_spatial_ocr(document, page)
                    added_rows = add_localized_bom_rows(
                        document, page,
                        {row["pass_id"] for row in adaptive})
                    document.configuration.setdefault("adaptive_ocr", []).append({
                        "page": page.page_number, "regions": len(adaptive),
                        "observations": len(adaptive_payload["observations"]),
                        "bom_after": len(document.bom), "localized_rows_added": added_rows,
                        "status": "ok",
                    })
                except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
                    document.issues.append({"stage": "adaptive_ocr", "page": page.page_number,
                                            "error": type(error).__name__,
                                            "raw": f"fast_ocr_adaptive_{page.page_number:04d}.log"})
        except (OSError, ValueError, KeyError, TypeError) as error:
            page.paddle_status = "failed"
            document.issues.append({"stage": "fast_ocr_adapter", "page": page.page_number,
                                    "error": str(error), "raw": name})
    return finalize_fast_document(document, source, output)


def finalize_fast_document(document: DrawingDocument, source: Path, output: Path) -> DrawingDocument:
    """Rebuild derived relationships identically for inference and raw-OCR replay."""
    extract_ocr_technical_requirements(document)
    document.configuration["unresolved_bom_candidates"] = [
        {"page": page.page_number, "candidates": unresolved_bom_candidates(document, page)}
        for page in document.pages if unresolved_bom_candidates(document, page)
    ]
    try:
        bind_bom_balloons(document, source)
    except (OSError, ValueError, TypeError, KeyError) as error:
        document.issues.append({"stage": "balloon_binding", "error": str(error)})
    document.configuration["semantic_tasks"] = build_semantic_tasks(document)
    finalize_document(document)
    save_document(document, output)
    write_json(output / "drawing.schema.json", DrawingDocument.model_json_schema())
    return document


__all__ = ["add_fast_ocr_result", "add_localized_bom_rows", "bom_recovery_regions",
           "unresolved_bom_candidates",
           "map_region_polygon",
           "parse_fast_drawing", "finalize_fast_document"]
