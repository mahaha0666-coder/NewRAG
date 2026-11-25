"""Bind BOM items to drawing callouts only with strict vector evidence."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from pathlib import Path

import pymupdf

from .document_schema import DrawingDocument, Entity, Relationship, stable_id
from .leader_probe import (
    extract_page_segments, forward_contour_hypotheses, leader_hypothesis_matches,
    overlap, strict_leader_matches,
)


CALLOUT_TOKEN = re.compile(
    r"^(?:(\d{1,3})|\(\s*(\d{1,3})\s*\)|\[\s*(\d{1,3})\s*\]|\{\s*(\d{1,3})\s*\})$")


def _center_inside(box: tuple[float, float, float, float], container) -> bool:
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return container[0] <= center_x <= container[2] and container[1] <= center_y <= container[3]


def _callout_number(text: str | None) -> str | None:
    normalized = unicodedata.normalize("NFKC", text or "").strip()
    match = CALLOUT_TOKEN.fullmatch(normalized)
    return next((group for group in match.groups() if group is not None), None) if match else None


def _callout_candidates(document: DrawingDocument, item_number: str, page_number: int,
                        table_boxes: list[tuple[float, float, float, float]]) -> list[dict]:
    view_boxes = [
        location.bbox
        for entity in [*document.views, *document.regions]
        if entity.kind == "views" or entity.attributes.get("label") == "image"
        for location in entity.locations
        if location.page_number == page_number and location.bbox is not None
    ]
    candidates = []
    for entity in document.raw_ocr:
        if _callout_number(entity.normalized_text or entity.raw_text) != item_number:
            continue
        if entity.attributes.get("localization") != "text_polygon":
            continue
        for location in entity.locations:
            if location.page_number != page_number or location.bbox is None:
                continue
            box = location.bbox
            if any(_center_inside(box, table_box) for table_box in table_boxes):
                continue
            if view_boxes and not any(_center_inside(box, view_box) for view_box in view_boxes):
                continue
            observation = {"box": box, "location": location, "entity": entity}
            duplicate = next((candidate for candidate in candidates
                              if any(overlap(existing["box"], box) > 0.5
                                     for existing in candidate["observations"])), None)
            if duplicate is not None:
                duplicate["observations"].append(observation)
                continue
            candidates.append({"observations": [observation]})
    return candidates


def _same_leader(first: dict, second: dict, tolerance: float) -> bool:
    direct = max(math.dist(first["points"][0], second["points"][0]),
                 math.dist(first["points"][1], second["points"][1]))
    reverse = max(math.dist(first["points"][0], second["points"][1]),
                  math.dist(first["points"][1], second["points"][0]))
    return min(direct, reverse) <= tolerance


def _candidate_topology(candidate: dict, segments: list[dict]) -> list[dict]:
    """Fuse repeated OCR boxes, retaining distinct vector-supported leaders."""
    fused = []
    for observation in candidate["observations"]:
        height = observation["box"][3] - observation["box"][1]
        for match in strict_leader_matches(observation["box"], segments):
            supported = {**match, "observation": observation}
            existing_index = next((index for index, row in enumerate(fused)
                                   if _same_leader(row, supported,
                                                   max(2.0, 0.12 * height))), None)
            if existing_index is None:
                fused.append(supported)
                continue
            existing = fused[existing_index]
            current_gap = existing["gap_pixels"] + min(
                row["gap_pixels"] for row in existing["target_support"])
            supported_gap = supported["gap_pixels"] + min(
                row["gap_pixels"] for row in supported["target_support"])
            if supported_gap < current_gap:
                fused[existing_index] = supported
    return fused


def _compact_match(match: dict) -> dict:
    target = min(match["target_support"], key=lambda row: row["gap_pixels"])
    landing = match["landing_support"][0]
    parallel = match.get("parallel_support")
    return {
        "path_index": match["path_index"],
        "item_index": match["item_index"],
        "leader_points": match["points"],
        "callout_gap_pixels": round(match["gap_pixels"], 3),
        "far_endpoint": match["far_endpoint"],
        "landing_points": landing["points"],
        "target_support_points": target["points"],
        "target_support_ref": {
            "path_index": target["path_index"],
            "item_index": target["item_index"],
        },
        "parallel_support_ref": ({
            "path_index": parallel["path_index"],
            "item_index": parallel["item_index"],
        } if parallel else None),
        "target_gap_pixels": round(target["gap_pixels"], 3),
    }


def _compact_hypothesis(match: dict) -> dict:
    """Keep enough native-vector provenance to build a focused review crop."""
    target = match.get("target_support")
    if isinstance(target, list):
        target = min(target, key=lambda row: row["gap_pixels"])
    return {
        "path_index": match["path_index"],
        "item_index": match["item_index"],
        "leader_points": match["points"],
        "callout_gap_pixels": round(match["gap_pixels"], 3),
        "far_endpoint": match["far_endpoint"],
        "landing_points": [row["points"] for row in match.get("landing_support", [])],
        "target_support_points": target.get("points") if target else None,
        "target_support_ref": ({
            "path_index": target["path_index"],
            "item_index": target["item_index"],
        } if target else None),
        "target_gap_pixels": round(target.get("gap_pixels"), 3)
        if target and target.get("gap_pixels") is not None else None,
        "forward_intersection": target.get("intersection") if target else None,
        "parallel_support_points": (match.get("parallel_support") or {}).get("points"),
        "parallel_support_ref": ({
            "path_index": match["parallel_support"]["path_index"],
            "item_index": match["parallel_support"]["item_index"],
        } if match.get("parallel_support") else None),
        "candidate_only": True,
    }


def _clear_generated_bindings(document: DrawingDocument) -> None:
    generated_ids = {
        entity.id for entity in document.balloons
        if entity.method == "ocr_fusion+pdf_vector_leader_v2"
        and entity.attributes.get("binding_status") == "geometry_supported_candidate"
    }
    if any(rel.type != "balloon_of" and
           (rel.source_id in generated_ids or rel.target_id in generated_ids)
           for rel in document.relationships):
        raise ValueError("Cannot rebuild balloon referenced by another relationship")
    document.relationships = [
        rel for rel in document.relationships
        if not (rel.type == "balloon_of" and rel.source_id in generated_ids)
    ]
    document.balloons = [entity for entity in document.balloons if entity.id not in generated_ids]


def bind_bom_balloons(document: DrawingDocument, pdf_path: Path) -> dict:
    """Create review-only balloon links when text and vector topology agree."""
    summary = {"policy": "ocr_fusion+strict_vector_leader_v2", "linked": [],
               "unresolved": [], "items": []}
    if not document.bom:
        _clear_generated_bindings(document)
        document.configuration["balloon_binding"] = summary
        return summary

    tables_by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    for region in document.regions:
        if region.attributes.get("label") != "table":
            continue
        for location in region.locations:
            if location.bbox is not None:
                tables_by_page.setdefault(location.page_number, []).append(location.bbox)
    for row in document.bom:
        for location in row.locations:
            if (location.bbox is not None
                    and location.bbox not in tables_by_page.get(location.page_number, [])):
                tables_by_page.setdefault(location.page_number, []).append(location.bbox)

    with pdf_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != document.source_sha256:
            raise ValueError("Source PDF does not match the drawing document")

    with pymupdf.open(pdf_path) as pdf:
        segments_by_page = {}
        for page in document.pages:
            if not 1 <= page.page_number <= pdf.page_count:
                raise ValueError(f"Page {page.page_number} is outside the source PDF")
            transform = pymupdf.Matrix(*page.pixel_to_pdf_matrix)
            if transform.invert():
                raise ValueError(f"Page {page.page_number} has a singular pixel transform")
            drawings = pdf[page.page_number - 1].get_drawings()
            segments_by_page[page.page_number] = extract_page_segments(drawings, transform)

    _clear_generated_bindings(document)
    bom_by_number: dict[str, list[Entity]] = {}
    for row in document.bom:
        item_number = str(row.attributes.get("item_number") or "").strip()
        if item_number.isdigit():
            bom_by_number.setdefault(item_number, []).append(row)

    for item_number, rows in sorted(bom_by_number.items(), key=lambda item: int(item[0])):
        if len(rows) != 1:
            summary["unresolved"].append({"item_number": item_number,
                                          "status": "duplicate_bom_item",
                                          "bom_count": len(rows)})
            continue
        row = rows[0]
        pages = {location.page_number for location in row.locations if location.bbox is not None}
        if len(pages) != 1:
            summary["unresolved"].append({"item_number": item_number,
                                          "status": "ambiguous_bom_page"})
            continue
        page_number = next(iter(pages))
        candidates = _callout_candidates(
            document, item_number, page_number, tables_by_page.get(page_number, []))
        if not candidates:
            summary["unresolved"].append({
                "item_number": item_number,
                "status": "callout_not_found", "candidate_count": 0,
            })
            continue

        supported = []
        ambiguous_count = 0
        candidate_hypotheses = []
        for candidate in candidates:
            matches = _candidate_topology(candidate, segments_by_page[page_number])
            if len(matches) == 1:
                supported.append((candidate, matches[0]))
            elif len(matches) > 1:
                ambiguous_count += 1
            if matches:
                continue
            hypotheses = []
            for observation in candidate["observations"]:
                box = observation["box"]
                hypotheses.extend(leader_hypothesis_matches(box, segments_by_page[page_number]))
                hypotheses.extend(forward_contour_hypotheses(box, segments_by_page[page_number]))
            compact = []
            seen = set()
            for hypothesis in hypotheses:
                key = (hypothesis["path_index"], hypothesis["item_index"],
                       tuple(round(value, 2) for value in hypothesis["far_endpoint"]))
                if key in seen:
                    continue
                seen.add(key)
                compact.append(_compact_hypothesis(hypothesis))
            if compact:
                candidate_hypotheses.append({
                    "candidate_entity_ids": sorted({
                        observation["entity"].id for observation in candidate["observations"]
                    }),
                    "candidate_bbox": list(candidate["observations"][0]["box"]),
                    "geometry": compact[:4],
                })
        summary["items"].append({
            "item_number": item_number,
            "ocr_candidate_clusters": len(candidates),
            "linked_occurrences": len(supported),
            "rejected_without_geometry": len(candidates) - len(supported) - ambiguous_count,
            "ambiguous_geometry_clusters": ambiguous_count,
        })
        if not supported:
            summary["unresolved"].append({
                "item_number": item_number,
                "status": "ambiguous_leader" if ambiguous_count else "no_leader_evidence",
                "candidate_count": len(candidates),
                "ambiguous_geometry_clusters": ambiguous_count,
                "candidate_entity_ids": sorted({
                    observation["entity"].id
                    for candidate in candidates
                    for observation in candidate["observations"]
                }),
                "candidate_hypotheses": candidate_hypotheses,
            })
            continue
        if ambiguous_count:
            summary["unresolved"].append({
                "item_number": item_number, "status": "partially_linked_ambiguous_leader",
                "ambiguous_geometry_clusters": ambiguous_count,
                "candidate_entity_ids": sorted({
                    observation["entity"].id
                    for candidate in candidates
                    for observation in candidate["observations"]
                }),
                "candidate_hypotheses": candidate_hypotheses,
            })

        for occurrence, (candidate, match) in enumerate(supported, 1):
            observations = candidate["observations"]
            sources = list({observation["entity"].id: observation["entity"]
                            for observation in observations}.values())
            evidence_ids = list(dict.fromkeys([
                *(entity.id for entity in sources), *row.evidence_ids,
            ]))
            geometry = _compact_match(match)
            recognition_passes = sorted({
                str(entity.attributes["recognition_pass"])
                for entity in sources if entity.attributes.get("recognition_pass") is not None
            })
            balloon = Entity(
                id=stable_id(document.source_sha256, "balloons",
                             [page_number, item_number, match["path_index"],
                              match["item_index"], row.id]),
                kind="balloons", raw_text=item_number, normalized_text=item_number,
                attributes={"item_number": item_number,
                            "binding_status": "geometry_supported_candidate",
                            "ocr_observation_count": len(observations),
                            "recognition_passes": recognition_passes,
                            "geometry": geometry},
                locations=[match["observation"]["location"].model_copy(deep=True)],
                evidence_ids=evidence_ids,
                raw_result_refs=list(dict.fromkeys(
                    reference for entity in sources for reference in entity.raw_result_refs)),
                method="ocr_fusion+pdf_vector_leader_v2",
            )
            if balloon.id not in {entity.id for entity in document.balloons}:
                document.balloons.append(balloon)
            relation = Relationship(
                id=stable_id(document.source_sha256, "relation",
                             [balloon.id, row.id, "balloon_of"]),
                source_id=balloon.id, target_id=row.id, type="balloon_of",
                evidence_ids=evidence_ids,
                raw_result_refs=list(dict.fromkeys([*balloon.raw_result_refs,
                                                    *row.raw_result_refs])),
                page=page_number,
                source_bbox=tuple(match["observation"]["box"]),
                scope_status="current_part",
                decision_status="supported",
                method="strict_pdf_vector_leader",
                needs_review=True,
            )
            if relation.id not in {existing.id for existing in document.relationships}:
                document.relationships.append(relation)
            summary["linked"].append({
                "item_number": item_number, "occurrence": occurrence,
                "balloon_id": balloon.id, "bom_id": row.id,
                "page_number": page_number,
                "status": "geometry_supported_candidate",
            })

    document.configuration["balloon_binding"] = summary
    DrawingDocument.model_validate(document.model_dump())
    return summary


__all__ = ["bind_bom_balloons"]
