"""Attach native-PDF geometry candidates to verified leader endpoints.

The candidates describe local visible strokes or faces.  They never establish
physical part ownership and therefore remain unresolved graph relationships.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Literal

import pymupdf

from .document_schema import DrawingDocument, Entity, Location, Relationship, stable_id
from .local_geometry import (
    collinear_runs,
    dense_parallel_ids,
    enclosed_faces,
    extract_strokes,
    face_exclusions,
    group_record,
    stroke_groups,
    visible_extent_gate,
)


def _positive_bbox(
    bbox: list[float], page_width: int, page_height: int,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    if x1 <= x0:
        x0, x1 = max(0.0, x0 - 1.0), min(float(page_width), x1 + 1.0)
    if y1 <= y0:
        y0, y1 = max(0.0, y0 - 1.0), min(float(page_height), y1 + 1.0)
    if not (0 <= x0 < x1 <= page_width and 0 <= y0 < y1 <= page_height):
        raise ValueError("Endpoint geometry candidate lies outside the rendered page")
    return x0, y0, x1, y1


def _area_polygon(value: object) -> list[tuple[float, float]] | None:
    """Keep only area polygons; two-point stroke paths remain in attributes."""
    if not isinstance(value, list) or len(value) < 3:
        return None
    if any(not isinstance(point, (list, tuple)) or len(point) != 2 for point in value):
        return None
    return [(float(point[0]), float(point[1])) for point in value]


def _candidate_records(
    paths: list[dict[str, Any]], page: Any, query: list[float], excluded_paths: set[int],
    window_half_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = [
        max(0, math.floor(query[0]) - window_half_size),
        max(0, math.floor(query[1]) - window_half_size),
        min(page.width, math.floor(query[0]) + window_half_size),
        min(page.height, math.floor(query[1]) + window_half_size),
    ]
    transform = pymupdf.Matrix(*page.pixel_to_pdf_matrix)
    if transform.invert():
        raise ValueError("Singular page pixel transform")
    strokes, stats = extract_strokes(paths, transform, frame, excluded_paths)
    groups = [group_record(group, query, "connected") for group in stroke_groups(strokes)]
    runs = [group_record(group, query, "gapped_collinear")
            for group in collinear_runs(strokes)]
    parallel = dense_parallel_ids(strokes)
    ignored, protected = face_exclusions(groups, runs, parallel)
    faces = enclosed_faces(strokes, frame, query, ignored)
    candidates = [
        *sorted(groups, key=lambda row: (row["distance_to_query_pixels"], row["id"]))[:6],
        *sorted(runs, key=lambda row: (row["distance_to_query_pixels"], row["id"]))[:3],
        *faces[:4],
    ]
    by_id = {stroke["id"]: stroke for stroke in strokes}
    for candidate in candidates:
        candidate["visible_extent_gate"] = visible_extent_gate(candidate)
        stroke_ids = (candidate.get("stroke_ids")
                      or candidate.get("boundary_touching_stroke_ids") or [])
        candidate["stroke_paths"] = [
            {"stroke_id": stroke_id, "points": by_id[stroke_id]["points"]}
            for stroke_id in stroke_ids
        ]
        candidate["stroke_sources"] = [
            source
            for stroke_id in stroke_ids
            for source in by_id[stroke_id]["sources"]
        ]
    return candidates, {
        "frame": frame,
        "query_xy": query,
        "excluded_leader_paths": sorted(excluded_paths),
        "stroke_count": len(strokes),
        "group_count": len(groups),
        "run_count": len(runs),
        "face_count": len(faces),
        "duplicates_merged": stats["duplicates_merged"],
        "unsupported_items": stats["unsupported_items"],
        "face_excluded_stroke_ids": sorted(ignored),
        "face_protected_connected_stroke_ids": sorted(protected),
    }


def select_endpoint_geometry_candidates(
    candidates: list[dict[str, Any]],
    target_support_refs: list[dict[str, int]],
    landing_tolerance_pixels: float,
    *,
    candidate_limit: int = 4,
) -> tuple[list[dict[str, Any]], str]:
    """Prefer candidates containing the native support stroke without promoting ownership."""
    target_keys = {
        (row["path_index"], row["item_index"])
        for row in target_support_refs
        if isinstance(row.get("path_index"), int)
        and isinstance(row.get("item_index"), int)
    }
    qualified = []
    kind_priority = {"connected": 0, "enclosed_face": 1, "gapped_collinear": 2}
    for candidate in candidates:
        if float(candidate["distance_to_query_pixels"]) > landing_tolerance_pixels:
            continue
        source_keys = {
            (source["path_index"], source["item_index"])
            for source in candidate.get("stroke_sources", [])
        }
        candidate["native_target_support_match"] = bool(source_keys & target_keys)
        candidate["native_target_support_refs"] = [
            {"path_index": path_index, "item_index": item_index}
            for path_index, item_index in sorted(source_keys & target_keys)
        ]
        candidate["source_membership_key"] = [list(key) for key in sorted(source_keys)]
        qualified.append(candidate)
    anchored = [candidate for candidate in qualified
                if candidate["native_target_support_match"]]
    pool = anchored or qualified
    pool.sort(key=lambda row: (
        kind_priority.get(row["kind"], 9),
        bool(row.get("clipped_by_window")),
        float(row["distance_to_query_pixels"]),
        row["id"],
    ))
    deduped = []
    seen_membership = set()
    for candidate in pool:
        membership = tuple(tuple(value) for value in candidate["source_membership_key"])
        if membership and membership in seen_membership:
            continue
        if membership:
            seen_membership.add(membership)
        deduped.append(candidate)
        if len(deduped) == candidate_limit:
            break
    policy = "native_target_support_anchored" if anchored else "distance_only_fallback"
    return deduped, policy


def extract_endpoint_geometry(
    document: DrawingDocument,
    endpoint: Entity,
    leader_hypotheses: list[dict[str, Any]],
    candidate_id: str,
    *,
    window_half_size: int = 240,
    landing_tolerance_pixels: float = 24.0,
    selection_mode: Literal["support_anchor", "distance_only"] = "support_anchor",
) -> dict[str, Any]:
    """Extract bounded local geometry around one native leader endpoint."""
    if endpoint.kind != "leader_endpoints" or not endpoint.locations:
        raise ValueError("Endpoint geometry requires a located leader endpoint")
    query = endpoint.attributes.get("endpoint_point")
    if (not isinstance(query, list) or len(query) != 2
            or any(not isinstance(value, (int, float)) for value in query)):
        raise ValueError("Leader endpoint lacks a native endpoint point")
    page_number = endpoint.locations[0].page_number
    page = next((row for row in document.pages if row.page_number == page_number), None)
    if page is None or not (0 <= query[0] < page.width and 0 <= query[1] < page.height):
        raise ValueError("Leader endpoint is outside its rendered page")

    relevant = [
        geometry
        for packet in leader_hypotheses
        if candidate_id in set(packet.get("candidate_entity_ids") or [])
        for geometry in packet.get("geometry") or []
    ]
    excluded_paths = {
        int(geometry["path_index"])
        for geometry in relevant
        if isinstance(geometry.get("path_index"), int)
    }
    base = {
        "schema_version": "endpoint_geometry_candidates.v1",
        "endpoint_id": endpoint.id,
        "page_number": page_number,
        "query_xy": [float(query[0]), float(query[1])],
        "window_half_size": window_half_size,
        "landing_tolerance_pixels": landing_tolerance_pixels,
        "candidate_id": candidate_id,
        "physical_part_ownership_verified": False,
    }
    if not excluded_paths:
        return {**base, "status": "missing_leader_path_provenance", "candidates": []}

    pdf_path = Path(document.source_path)
    if not pdf_path.is_file() or pdf_path.suffix.casefold() != ".pdf":
        return {**base, "status": "source_pdf_unavailable", "candidates": []}
    if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != document.source_sha256:
        raise ValueError("Endpoint geometry source PDF hash mismatch")
    with pymupdf.open(pdf_path) as pdf:
        paths = pdf[page_number - 1].get_drawings()
    candidates, diagnostics = _candidate_records(
        paths, page, [float(query[0]), float(query[1])], excluded_paths,
        window_half_size,
    )
    available_target_support_refs = [
        geometry["target_support_ref"]
        for geometry in relevant
        if isinstance(geometry.get("target_support_ref"), dict)
    ]
    target_support_refs = (
        available_target_support_refs if selection_mode == "support_anchor" else []
    )
    qualified, selection_policy = select_endpoint_geometry_candidates(
        candidates,
        target_support_refs,
        landing_tolerance_pixels,
    )
    return {
        **base,
        **diagnostics,
        "status": "candidates_found" if qualified else "no_local_geometry_at_endpoint",
        "candidates": qualified,
        "all_candidate_count": len(candidates),
        "selection_policy": selection_policy,
        "selection_mode": selection_mode,
        "native_target_support_ref_count": len(available_target_support_refs),
        "native_target_support_ref_used_count": len(target_support_refs),
    }


def persist_endpoint_geometry(
    document: DrawingDocument, endpoint: Entity, analysis: dict[str, Any], ref: str,
) -> int:
    """Persist local candidates and unresolved endpoint-to-geometry edges."""
    candidates = list(analysis.get("candidates") or [])
    endpoint.attributes["geometry_candidate_audit"] = {
        key: value for key, value in analysis.items() if key != "candidates"
    }
    if not candidates:
        return 0
    page_number = endpoint.locations[0].page_number
    page = next(row for row in document.pages if row.page_number == page_number)
    known_objects = {entity.id for entity in document.geometry_objects}
    known_relations = {relation.id for relation in document.relationships}
    added = 0
    for candidate in candidates:
        gate = candidate["visible_extent_gate"]
        object_id = stable_id(
            document.source_sha256,
            "geometry_objects",
            [endpoint.id, candidate["id"], candidate.get("stroke_ids", [])],
        )
        raw_refs = list(dict.fromkeys([
            *endpoint.raw_result_refs,
            ref,
            *(
                f"pdf-vector://page/{page_number}/path/{source['path_index']}"
                f"/item/{source['item_index']}"
                for source in candidate.get("stroke_sources", [])
            ),
        ]))
        geometry = Entity(
            id=object_id,
            kind="geometry_objects",
            attributes={
                "geometry_type": candidate["kind"],
                "visible_extent_status": gate["visible_extent_status"],
                "physical_part_id": None,
                "candidate_id": candidate["id"],
                "stroke_ids": candidate.get("stroke_ids", []),
                "stroke_sources": candidate.get("stroke_sources", []),
                "distance_to_endpoint_pixels": candidate["distance_to_query_pixels"],
                "native_target_support_match": candidate.get(
                    "native_target_support_match", False,
                ),
                "native_target_support_refs": candidate.get(
                    "native_target_support_refs", [],
                ),
                "source_membership_key": candidate.get("source_membership_key", []),
                "complete_part_contour_established": False,
                "gate_reasons": gate["gate_reasons"],
            },
            locations=[Location(
                page_number=page_number,
                bbox=_positive_bbox(candidate["bbox"], page.width, page.height),
                polygon=_area_polygon(candidate.get("polygon")),
                localization_source="pdf_vector",
            )],
            evidence_ids=list(endpoint.evidence_ids),
            raw_result_refs=raw_refs,
            method="native_pdf_endpoint_geometry_v1",
            uncertain=True,
            needs_review=True,
        )
        if object_id not in known_objects:
            document.geometry_objects.append(geometry)
            known_objects.add(object_id)
            added += 1
        relationship = Relationship(
            id=stable_id(document.source_sha256, "relation",
                         [endpoint.id, object_id, "landing_on"]),
            source_id=endpoint.id,
            target_id=object_id,
            type="landing_on",
            evidence_ids=list(endpoint.evidence_ids),
            raw_result_refs=raw_refs,
            page=page_number,
            source_bbox=endpoint.locations[0].bbox,
            scope_status="unassigned",
            decision_status="unresolved",
            method="native_pdf_endpoint_geometry_v1",
            basis="native_geometry",
            uncertain=True,
            needs_review=True,
        )
        if relationship.id not in known_relations:
            document.relationships.append(relationship)
            known_relations.add(relationship.id)
    endpoint.attributes["target_geometry_status"] = "candidate_set_unresolved"
    document.schema_version = "1.2.0"
    DrawingDocument.model_validate(document.model_dump())
    return added


def attach_endpoint_geometry(
    document: DrawingDocument,
    endpoint: Entity,
    leader_hypotheses: list[dict[str, Any]],
    candidate_id: str,
    ref: str,
) -> dict[str, Any]:
    """Extract and persist review-only local geometry for one leader endpoint."""
    source_candidate = next(
        (entity for entity in document.entities() if entity.id == candidate_id), None,
    )
    source_location = next(
        (location for location in (source_candidate.locations if source_candidate else [])
         if location.page_number == endpoint.locations[0].page_number
         and location.bbox is not None),
        None,
    )
    text_height = (
        source_location.bbox[3] - source_location.bbox[1]
        if source_location is not None and source_location.bbox is not None else 1.0
    )
    analysis = extract_endpoint_geometry(
        document,
        endpoint,
        leader_hypotheses,
        candidate_id,
        landing_tolerance_pixels=max(24.0, 1.25 * text_height),
    )
    added = persist_endpoint_geometry(document, endpoint, analysis, ref)
    return {**analysis, "persisted_candidate_count": added}


__all__ = [
    "attach_endpoint_geometry",
    "extract_endpoint_geometry",
    "persist_endpoint_geometry",
    "select_endpoint_geometry_candidates",
]
