"""Bounded VLM task selection for unresolved drawing evidence.

The gate is deliberately evidence-first: it never asks a generative model to
re-read every view. A task is emitted only when deterministic OCR/CAD checks
leave a concrete, localized ambiguity to adjudicate.
"""

from __future__ import annotations

import re
from typing import Any

from .document_schema import DrawingDocument
from .semantic_routing import route_task


_GEOMETRY_ANCHORS = (
    ("thread", re.compile(r"(?:^|[^A-Z])M\s*\d+(?:[Xx×*]\s*\d+)?|螺纹|攻丝", re.I), 86),
    ("surface_finish", re.compile(r"\bR[AZ]\s*\d|粗糙度", re.I), 85),
    ("diameter", re.compile(r"[ΦφØ∅]\s*\d|直径|沉孔|锪孔", re.I), 84),
    ("feature", re.compile(r"(?:^|\s)(?:SR|R)\s*\d|倒角|圆角|槽宽|槽深|孔深", re.I), 82),
)


def _located_boxes(document: DrawingDocument, entity_ids: set[str]) -> dict[int, list[tuple[float, ...]]]:
    boxes: dict[int, list[tuple[float, ...]]] = {}
    entities = {entity.id: entity for entity in document.entities()}
    for entity_id in entity_ids:
        entity = entities.get(entity_id)
        if entity is None:
            continue
        for location in entity.locations:
            if location.bbox is not None:
                boxes.setdefault(location.page_number, []).append(tuple(location.bbox))
    return boxes


def _inside_box(box: tuple[float, ...], candidates: list[tuple[float, ...]]) -> bool:
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return any(candidate[0] <= center[0] <= candidate[2]
               and candidate[1] <= center[1] <= candidate[3] for candidate in candidates)


def geometry_attribution_tasks(
    document: DrawingDocument, *, limit: int = 3,
) -> list[dict[str, Any]]:
    """Select a few route-relevant OCR annotations outside BOM/note blocks."""

    excluded_entities = [*document.bom, *document.technical_notes]
    excluded_ids = {
        evidence_id
        for entity in excluded_entities
        for evidence_id in entity.evidence_ids
    }
    excluded_ids.update(entity.id for entity in excluded_entities)
    excluded_ids.update(
        region.id for region in document.regions
        if region.attributes.get("label") in {"table", "title", "footer"}
    )
    excluded_boxes = _located_boxes(document, excluded_ids)
    candidates = []
    for observation in document.raw_ocr:
        if observation.id in excluded_ids:
            continue
        text = str(observation.normalized_text or observation.raw_text or "").strip()
        if not text or len(text) > 40 or re.match(r"^\s*\d+[.、\s]", text):
            continue
        matched = next((row for row in _GEOMETRY_ANCHORS if row[1].search(text)), None)
        if matched is None:
            continue
        anchor_type, _, priority = matched
        for location in observation.locations:
            if location.bbox is None or _inside_box(tuple(location.bbox), excluded_boxes.get(
                location.page_number, [],
            )):
                continue
            candidates.append({
                "kind": "geometry_attribution",
                "priority": priority,
                "page": location.page_number,
                "entity_ids": [observation.id],
                "anchor_type": anchor_type,
                "anchor_text": text[:160],
                "reason": (
                    "attribute one localized route-relevant annotation to visible adjacent geometry"
                ),
                "bbox": tuple(location.bbox),
            })
    candidates.sort(key=lambda row: (
        -row["priority"], row["page"], row["bbox"][1], row["bbox"][0], row["entity_ids"][0],
    ))
    selected = []
    seen_types = set()
    for candidate in candidates:
        key = (candidate["page"], candidate["anchor_type"])
        if key in seen_types:
            continue
        seen_types.add(key)
        candidate.pop("bbox")
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def build_semantic_tasks(
    document: DrawingDocument, *, max_tasks: int = 6,
) -> list[dict[str, Any]]:
    """Return localized VLM tasks ordered by engineering risk."""
    if not 1 <= max_tasks <= 12:
        raise ValueError("max_tasks must be between 1 and 12")
    tasks: list[dict[str, Any]] = []
    for issue in document.issues:
        stage = str(issue.get("stage") or "")
        if stage in {"bom_ocr_conflict", "bom_conflict"}:
            tasks.append({
                "kind": "bom_conflict",
                "priority": 100,
                "page": issue.get("page"),
                "item_number": issue.get("item_number"),
                "entity_ids": list(issue.get("entity_ids") or []),
                "reason": "compare localized BOM crop against conflicting OCR candidates",
            })
        elif stage in {"balloon_binding", "semantic_relationship"}:
            tasks.append({
                "kind": "leader_or_relation",
                "priority": 90,
                "page": issue.get("page"),
                "entity_ids": list(issue.get("entity_ids") or []),
                "reason": "adjudicate a deterministic candidate with visible leader/geometry evidence",
            })
    # A strict vector pass records unresolved balloon candidates in the
    # configuration rather than creating a weak relationship. Turn those
    # records into localized high-risk tasks with both the BOM row and the
    # matching figure-side OCR IDs as evidence endpoints.
    binding = document.configuration.get("balloon_binding") or {}
    for unresolved in binding.get("unresolved") or []:
        if unresolved.get("status") == "duplicate_bom_item":
            # Resolve the table identity first; a leader cannot be assigned to
            # one of two competing BOM rows without creating a false link.
            continue
        item_number = str(unresolved.get("item_number") or "").strip()
        rows = [row for row in document.bom
                if str(row.attributes.get("item_number") or "").strip() == item_number]
        for row in rows:
            page = next((location.page_number for location in row.locations
                         if location.bbox is not None), None)
            if page is None:
                continue
            # balloon_binding has already removed table, border and unsupported
            # vector candidates. Reuse exactly those IDs instead of scanning
            # every same-number OCR token (which includes dimensions).
            candidate_entity_ids = unresolved.get("candidate_entity_ids")
            if candidate_entity_ids is None:
                # Without the binding pass's figure-side filtering, a same-number
                # OCR scan admits dimensions and the table item itself.
                continue
            hypotheses = list(unresolved.get("candidate_hypotheses") or [])
            hypothesis_ids = {
                entity_id for packet in hypotheses
                for entity_id in packet.get("candidate_entity_ids") or []
            }
            selected_candidate_ids = sorted(hypothesis_ids or set(candidate_entity_ids))
            # The strict response contract adjudicates at most three candidates.
            # Wider ambiguity remains unresolved instead of silently omitting rivals.
            if not selected_candidate_ids or len(selected_candidate_ids) > 3:
                continue
            entity_ids = [row.id, *selected_candidate_ids]
            tasks.append({
                "kind": "leader_or_relation", "priority": 90,
                "page": page, "item_number": item_number,
                "entity_ids": sorted(set(entity_ids)),
                "leader_hypotheses": hypotheses,
                "candidate_selection_basis": (
                    "native_vector_hypothesis" if hypothesis_ids else "localized_ocr_candidates"
                ),
                "reason": f"inspect unresolved vector leader for BOM item {item_number}: {unresolved.get('status')}",
            })
    for page_group in document.configuration.get("unresolved_bom_candidates") or []:
        for candidate in page_group.get("candidates") or []:
            tasks.append({
                "kind": "bom_row_gap", "priority": 60,
                "page": page_group.get("page"),
                "entity_ids": list(candidate.get("entity_ids")
                                    or [candidate.get("entity_id")]),
                "part_number": candidate.get("part_number"),
                "reason": "inspect the localized table area for missing sequence/name/quantity anchors",
            })
    for page in document.pages:
        if page.paddle_status != "ok":
            tasks.append({
                "kind": "ocr_gap",
                "priority": 80,
                "page": page.page_number,
                "reason": "recover a failed or partial page only from localized OCR gaps",
            })
    for note in document.technical_notes:
        if note.attributes.get("scope_status") == "unassigned":
            tasks.append({
                "kind": "technical_scope",
                "priority": 70,
                "page": note.locations[0].page_number if note.locations else None,
                "entity_ids": [note.id],
                "reason": "compare note location and leader context for ownership scope",
            })
    tasks.extend(geometry_attribution_tasks(document))
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for task in sorted(tasks, key=lambda row: (-row["priority"], str(row.get("page")), row["kind"])):
        key = (task["kind"], task.get("page"), task.get("item_number"),
               tuple(task.get("entity_ids") or []))
        unique.setdefault(key, task)
    selected = list(unique.values())[:max_tasks]
    return [{**task, "model_tier": route_task(task), "status": "review_candidate",
             "auto_apply": False} for task in selected]


__all__ = ["build_semantic_tasks", "geometry_attribution_tasks"]
