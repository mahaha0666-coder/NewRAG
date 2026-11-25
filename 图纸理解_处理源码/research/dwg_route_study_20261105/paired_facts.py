"""Link native observations to a PDF derived from the very same DWG.

Coordinates describe the renderer's representation, not a certified CAD plot,
physical part, or manufacturing scope. The two representations count once.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from research.dwg_route_study_20261105.scene_input import SceneInput
from research.dwg_route_validation_20261015.native_ir import build_ir


@dataclass(frozen=True)
class PageTransform:
    matrix: tuple[float, ...]
    page: tuple[float, float]

    def __post_init__(self) -> None:
        if (len(self.matrix) != 16 or len(self.page) != 2
                or not all(math.isfinite(v) for v in (*self.matrix, *self.page))
                or min(self.page) <= 0):
            raise ValueError("Invalid finite page transform")
        m = self.matrix
        if (any(m[i] != 0 for i in (2, 3, 6, 7, 8, 9, 11)) or m[15] != 1
                or abs(m[0] * m[5] - m[1] * m[4]) < 1e-15):
            raise ValueError("An invertible, planar affine transform is required")

    def forward(self, x: float, y: float) -> list[float]:
        m = self.matrix
        return [x * m[0] + y * m[4] + m[12], x * m[1] + y * m[5] + m[13]]

    def inverse(self, x: float, y: float) -> list[float]:
        m = self.matrix
        dx, dy = x - m[12], y - m[13]
        det = m[0] * m[5] - m[1] * m[4]
        return [(m[5] * dx - m[4] * dy) / det, (-m[1] * dx + m[0] * dy) / det]

    def locate(self, bounds: list[float], basis: str) -> dict:
        if (len(bounds) != 4 or not all(math.isfinite(v) for v in bounds)
                or bounds[2] < bounds[0] or bounds[3] < bounds[1]):
            raise ValueError("Invalid native entity bounds")
        x0, y0, x1, y1 = bounds
        corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        points = [self.forward(*point) for point in corners]
        if any(not (-1e-6 <= p[0] <= self.page[0] + 1e-6
                    and -1e-6 <= p[1] <= self.page[1] + 1e-6) for p in points):
            return {"status": "outside_derived_page", "cad_bounds": bounds,
                    "unclipped_pdf_corners": points, "basis": basis}
        error = max(abs(a - b) for point, mapped in zip(corners, points, strict=True)
                    for a, b in zip(point, self.inverse(*mapped), strict=True))
        return {"status": "located_in_derived_pdf", "page_index": 0,
                "cad_bounds": bounds, "pdf_corners": points,
                "pdf_bounds_pt": [min(p[0] for p in points), min(p[1] for p in points),
                                  max(p[0] for p in points), max(p[1] for p in points)],
                "inverse_max_error_cad": error, "basis": basis,
                "bounds_are_exact_part_or_glyph_outline": False}


def build_bundle(raw: dict, manifest: dict, *, item_code: str, source_sha256: str,
                 response_sha256: str, pdf_sha256: str) -> dict:
    if (manifest["source"]["sha256"] != source_sha256
            or manifest["raw_response_sha256"] != response_sha256):
        raise ValueError("Native observations and derived PDF must have the same source")
    if manifest["companion_pdf_used_as_render_input"] or not manifest["automatic_full_extent"]:
        raise ValueError("Expected a complete extent export of this DWG")
    native = build_ir(raw, item_code=item_code, source_sha256=source_sha256,
                      response_sha256=response_sha256)
    transform = PageTransform(tuple(manifest["cad_to_pdf_matrix44"]), tuple(manifest["page_points"]))
    scene = SceneInput(raw)
    dimension_display = {row["handle"]: row for row in manifest["dimensions"]}
    bounds = manifest["entity_cad_bounds"]
    skipped = {row["handle"] for row in manifest["skipped_entities"]}
    facts = []

    def add(category: str, ref: dict, values: dict, extent: list | None, basis: str) -> None:
        if ref["source_sha256"] != source_sha256 or ref["response_sha256"] != response_sha256:
            raise ValueError("Cross-source observation is not eligible for pairing")
        admission = scene.qualify(ref)
        handle = ref["native_handle"]
        if not admission["admitted"]:
            location = {"status": "not_admitted", "reason": admission["reason"]}
        elif admission["basis"] != "direct_modelspace":
            location = {"status": "unresolved_instance_transform"}
        elif handle in skipped:
            location = {"status": "renderer_skipped_entity"}
        elif extent is None:
            location = {"status": "native_entity_extent_unavailable"}
        else:
            location = transform.locate(extent, basis)
        facts.append({"id": f"{category}:{handle}", "category": category, **values,
                      "source_ref": ref, "same_origin_id": source_sha256,
                      "independent_source_count": 1, "scene_admission": admission,
                      "pdf_location": location, "recognition_confidence": None,
                      "physical_owner": "unknown", "view_scope": "unknown",
                      "relationship_confidence": None, "hard_constraint_eligible": False})

    for node in native["dimensions"]:
        display = dimension_display.get(node["id"], {})
        add("dimension", node["source_ref"],
            {"native_kind": node["native_kind"], "raw_text": node["raw_text"],
             "raw_measurement": node["raw_measurement"], "unit": node["unit"],
             "unit_source": node["unit_source"], "cad_coordinates": node["cad_coordinates"],
             "rendered_dimension_literals": display.get("display_literal", [])},
            display.get("dimension_cad_bbox"), "converted_native_dimension_group_extent")
    for node in native["geometry_nodes"]:
        add("geometry", node["source_ref"],
            {"native_kind": node["native_kind"], "cad_coordinates": node["cad_coordinates"]},
            bounds.get(node["id"]), "converted_native_entity_extent")
    # Dimension overrides already belong to dimension facts. Block text remains
    # visible as source observations, but is never positioned as a world instance.
    for row in native["text_evidence"]:
        if row["field"] == "text" and row["kind"] in {"TEXT", "MTEXT"}:
            add("text", row["source_ref"], {"native_kind": row["kind"], "raw_text": row["value"]},
                bounds.get(row["handle"]), "converted_native_text_entity_extent")
    for proxy in manifest["proxy_objects"]:
        ref = {"source_sha256": source_sha256, "response_sha256": response_sha256,
               "json_pointer": proxy["json_pointer"], "native_handle": proxy["native_handle"]}
        add("stored_graphics", ref,
            {"native_kind": proxy["native_kind"], "original_texts": proxy["original_texts"],
             "preview_sha256": proxy["preview_sha256"], "preview_size": proxy["preview_size"],
             "primitive_count": proxy["primitive_count"],
             "coordinate_status": proxy["coordinate_status"], "native_WCS_confirmed": False},
            proxy["cad_bbox"], "stored_preview_XY_in_derived_pdf_not_certified_WCS")

    if len({row["id"] for row in facts}) != len(facts):
        raise ValueError("Duplicate evidence identity")
    facts.sort(key=lambda row: row["id"])
    admitted = {row["id"] for row in facts if row["scene_admission"]["admitted"]}
    relations, withheld = [], []
    for edge in native["relations"]:
        relation = {**edge, "from": f"dimension:{edge['dimension_id']}",
                    "to": f"geometry:{edge['geometry_id']}"}
        if relation["from"] in admitted and relation["to"] in admitted:
            relations.append(relation)
        else:
            withheld.append({**relation, "reason": "endpoint_not_admitted"})
    located = [row for row in facts if row["pdf_location"]["status"] == "located_in_derived_pdf"]
    return {"schema": "same_origin_dwg_pdf_evidence.v1", "item_code": item_code,
            "source": manifest["source"], "response_sha256": response_sha256,
            "same_origin_id": source_sha256,
            "derived_pdf": {"sha256": pdf_sha256, "derived_from_sha256": source_sha256,
                            "page_points": manifest["page_points"],
                            "cad_to_pdf_matrix44": manifest["cad_to_pdf_matrix44"]},
            "facts": facts, "relations": relations, "withheld_relations": withheld,
            "unresolved_references": native["unresolved_references"],
            "summary": {"facts": len(facts), "admitted": len(admitted), "located": len(located),
                        "categories": dict(Counter(row["category"] for row in facts)),
                        "located_categories": dict(Counter(row["category"] for row in located)),
                        "location_states": dict(Counter(row["pdf_location"]["status"] for row in facts)),
                        "direct_dimension_geometry_relations": len(relations),
                        "withheld_relations": len(withheld),
                        "unresolved_reference_slots": len(native["unresolved_references"]),
                        "inverse_max_error_cad": max((row["pdf_location"]["inverse_max_error_cad"]
                                                      for row in located), default=None)},
            "limits": {"independent_source_count": 1, "final_CAD_plot_equivalence_confirmed": False,
                       "physical_part_ownership_confirmed": False, "manufacturing_scope_confirmed": False,
                       "eligible_for_route_training": False, "ocr_used": False, "gpu_used": False,
                       "route_labels_read": False, "val_test_read": False}}
