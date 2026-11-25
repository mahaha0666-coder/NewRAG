"""Find the first native geometry contact along dimension extension lines."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import pymupdf

from .dimension_geometry import parallel_to_extension, source_keys, stroke_source_keys
from .document_schema import DrawingDocument
from .local_geometry import extract_strokes
from .manufacturing_ir import ManufacturingIR
from .native_view_geometry import positive_page_bbox


def _extension_ray(
    extension: dict[str, Any], dimension: dict[str, Any],
) -> tuple[list[float], float, float] | None:
    if dimension["orientation"] == "horizontal":
        anchor = [float(extension["first"][0]), float(dimension["first"][1])]
        offsets = [point[1] - anchor[1] for point in (extension["first"], extension["second"])]
    else:
        anchor = [float(dimension["first"][0]), float(extension["first"][1])]
        offsets = [point[0] - anchor[0] for point in (extension["first"], extension["second"])]
    remote_offset = max(offsets, key=abs)
    if math.isclose(remote_offset, 0.0, abs_tol=1e-6):
        return None
    return anchor, math.copysign(1.0, remote_offset), abs(remote_offset)


def _segment_axis_contact(
    first: list[float],
    second: list[float],
    anchor: list[float],
    direction: float,
    extension_orientation: str,
) -> tuple[float, float, list[float]]:
    if extension_orientation == "vertical":
        fixed_axis, along_axis = 0, 1
    else:
        fixed_axis, along_axis = 1, 0
    fixed_value = anchor[fixed_axis]
    delta = second[fixed_axis] - first[fixed_axis]
    if abs(delta) > 1e-9 and (
        min(first[fixed_axis], second[fixed_axis]) <= fixed_value
        <= max(first[fixed_axis], second[fixed_axis])
    ):
        fraction = (fixed_value - first[fixed_axis]) / delta
        actual = [
            first[axis] + fraction * (second[axis] - first[axis]) for axis in (0, 1)
        ]
    else:
        actual = min((first, second), key=lambda point: abs(point[fixed_axis] - fixed_value))
    transverse = abs(actual[fixed_axis] - fixed_value)
    along = direction * (actual[along_axis] - anchor[along_axis])
    projected = [float(actual[0]), float(actual[1])]
    projected[fixed_axis] = fixed_value
    return along, transverse, projected


def find_first_contact_targets(
    extension: dict[str, Any],
    dimension: dict[str, Any],
    strokes: list[dict[str, Any]],
    *,
    transverse_tolerance: float = 6.0,
    layer_tolerance: float = 6.0,
    minimum_forward_distance: float = 2.0,
    limit: int = 4,
) -> tuple[list[float] | None, list[dict[str, Any]]]:
    """Return candidates in the nearest non-parallel intersection layer."""

    ray = _extension_ray(extension, dimension)
    if ray is None:
        return None, []
    anchor, direction, maximum_distance = ray
    extension_orientation = "vertical" if dimension["orientation"] == "horizontal" else "horizontal"
    excluded = source_keys(dimension) | source_keys(extension)
    candidates = []
    for stroke in strokes:
        if excluded & stroke_source_keys(stroke) or parallel_to_extension(stroke, extension):
            continue
        contacts = [
            _segment_axis_contact(first, second, anchor, direction, extension_orientation)
            for first, second in zip(stroke["points"], stroke["points"][1:])
        ]
        eligible = [
            row for row in contacts
            if minimum_forward_distance <= row[0] <= maximum_distance + transverse_tolerance
            and row[1] <= transverse_tolerance
        ]
        if not eligible:
            continue
        along, transverse, contact = min(eligible, key=lambda row: (row[0], row[1]))
        candidates.append({
            "stroke_id": stroke["id"],
            "forward_distance_pixels": round(along, 4),
            "transverse_distance_pixels": round(transverse, 4),
            "contact_point": contact,
            "points": stroke["points"],
            "source_refs": stroke.get("sources") or [],
            "clipped_by_view": bool(stroke.get("clipped_by_window")),
        })
    if not candidates:
        return anchor, []
    candidates.sort(key=lambda row: (
        row["forward_distance_pixels"], row["transverse_distance_pixels"], row["stroke_id"]
    ))
    first_distance = candidates[0]["forward_distance_pixels"]
    return anchor, [
        row for row in candidates
        if row["forward_distance_pixels"] <= first_distance + layer_tolerance
    ][:limit]


def build_dimension_first_contacts(
    document: DrawingDocument,
    ir: ManufacturingIR,
    topology: dict[str, Any],
    assignments: dict[str, Any],
) -> dict[str, Any]:
    source = Path(document.source_path)
    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if source_hash != document.source_sha256 or topology["source_sha256"] != source_hash:
        raise ValueError("Source PDF hash mismatch")

    chain_by_key = {
        (row["id"], row["measurement_id"]): row for row in topology["chains"]
    }
    selected = [
        row for row in assignments["associations"]
        if row["assignment_status"] == "selected_unverified"
    ]
    view_by_id = {row.id: row for row in ir.views if row.page is not None and row.bbox is not None}
    page_by_number = {row.page_number: row for row in document.pages}
    strokes_by_view: dict[str, list[dict[str, Any]]] = {}
    with pymupdf.open(source) as pdf:
        paths_by_page: dict[int, list[dict[str, Any]]] = {}
        for view_id in {row["view_id"] for row in selected}:
            view = view_by_id[view_id]
            page = page_by_number[view.page]
            paths = paths_by_page.setdefault(view.page, pdf[view.page - 1].get_drawings())
            transform = pymupdf.Matrix(*page.pixel_to_pdf_matrix)
            if transform.invert():
                raise ValueError("Singular page transform")
            frame = [
                max(0, int(math.floor(view.bbox[0]))),
                max(0, int(math.floor(view.bbox[1]))),
                min(page.width, int(math.ceil(view.bbox[2]))),
                min(page.height, int(math.ceil(view.bbox[3]))),
            ]
            strokes_by_view[view_id] = extract_strokes(paths, transform, frame, set())[0]

    contacts = []
    for assignment in selected:
        chain = chain_by_key[(assignment["topology_chain_id"], assignment["measurement_id"])]
        dimension = chain["dimension_line"]
        page = page_by_number[chain["page"]]
        for index, extension in enumerate(chain["extension_lines"]):
            anchor, targets = find_first_contact_targets(
                extension,
                dimension,
                strokes_by_view.get(chain["view_id"], []),
            )
            target_rows = []
            for target in targets:
                points = target.pop("points")
                xs, ys = [point[0] for point in points], [point[1] for point in points]
                bbox, raw_bbox = positive_page_bbox(
                    [min(xs), min(ys), max(xs), max(ys)], page.width, page.height,
                )
                target_rows.append({**target, "bbox": bbox, "raw_stroke_bbox": raw_bbox})
            identity = [assignment["id"], index, anchor]
            contacts.append({
                "id": "dimension_first_contact_" + hashlib.sha256(
                    repr(identity).encode()
                ).hexdigest()[:20],
                "assignment_id": assignment["id"],
                "measurement_id": assignment["measurement_id"],
                "measurement_raw_text": assignment["measurement_raw_text"],
                "topology_chain_id": assignment["topology_chain_id"],
                "view_id": chain["view_id"],
                "page": chain["page"],
                "extension_index": index,
                "ray_anchor": anchor,
                "extension_source_refs": extension.get("sources") or [],
                "target_candidates": target_rows,
                "contact_status": (
                    "not_found" if not target_rows else "unique_candidate"
                    if len(target_rows) == 1 else "ambiguous_same_layer"
                ),
                "decision_status": "unresolved",
                "feature_ownership_verified": False,
                "needs_review": True,
            })

    return {
        "schema_version": "native_dimension_first_contact_r02g.v1",
        "source_sha256": source_hash,
        "route_label_access": False,
        "parameters": {
            "transverse_tolerance_pixels": 6.0,
            "first_layer_tolerance_pixels": 6.0,
            "minimum_forward_distance_pixels": 2.0,
            "maximum_target_candidates": 4,
        },
        "contacts": contacts,
        "summary": {
            "selected_assignment_count": len(selected),
            "contact_count": len(contacts),
            "contacts_with_candidates": sum(bool(row["target_candidates"]) for row in contacts),
            "contacts_at_candidate_cap": sum(len(row["target_candidates"]) == 4 for row in contacts),
            "unique_candidate_contacts": sum(
                row["contact_status"] == "unique_candidate" for row in contacts
            ),
            "supported_feature_ownership_count": 0,
        },
        "limitations": [
            "The first contact is a local geometry anchor, not a named or owned feature",
            "Sampled curves and coincident native strokes can yield same-layer ambiguity",
            "No route labels or human relation gold are used",
        ],
    }


__all__ = ["build_dimension_first_contacts", "find_first_contact_targets"]
