"""Versioned, evidence-carrying contract for drawing extraction candidates."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


COLLECTIONS = (
    "assemblies", "parts", "sub_parts", "bom", "views", "features", "balloons",
    "leader_segments", "leader_endpoints", "geometry_objects",
    "dimensions", "tolerances", "gdt", "datums", "surface_finish", "threads",
    "welding_symbols", "technical_notes", "regions", "raw_ocr",
)
METADATA_FIELDS = (
    "drawing_name", "drawing_number", "part_number", "product_name", "material",
    "scale", "weight", "units", "designer", "reviewer", "date", "company",
    "revision", "other_fields",
)
ATTRIBUTE_FIELDS = {
    "assemblies": ("part_number", "role"),
    "parts": ("part_number", "role", "material"),
    "sub_parts": ("part_number", "role", "material"),
    "bom": ("item_number", "part_number", "part_name", "quantity", "material", "specification", "remark"),
    "views": ("view_type", "label", "scale"),
    "features": ("feature_type",),
    "balloons": ("item_number",),
    "leader_segments": ("item_number", "source_candidate_entity_id"),
    "leader_endpoints": ("item_number", "target_geometry_status"),
    "geometry_objects": ("geometry_type", "visible_extent_status", "physical_part_id"),
    "dimensions": ("dimension_type", "value", "unit", "upper_deviation", "lower_deviation"),
    "tolerances": ("tolerance_type", "upper_deviation", "lower_deviation", "standard"),
    "gdt": ("symbol", "frame", "value", "unit"),
    "datums": ("label",),
    "surface_finish": ("finish_type", "value", "unit"),
    "threads": ("designation", "pitch", "depth", "handedness"),
    "welding_symbols": ("symbol", "size", "side"),
    "technical_notes": ("note_type",),
}


def stable_id(source: str, kind: str, identity: JsonValue) -> str:
    canonical = json.dumps([source, kind, identity], sort_keys=True, ensure_ascii=True,
                           separators=(",", ":"), allow_nan=False)
    return kind + "_" + hashlib.sha256(canonical.encode()).hexdigest()[:24]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SemanticProposal(StrictModel):
    entities: list[dict[str, JsonValue]] = Field(default_factory=list)
    metadata: list[dict[str, JsonValue]] = Field(default_factory=list)
    relationships: list[dict[str, JsonValue]] = Field(default_factory=list)


class Location(StrictModel):
    page_number: int = Field(ge=1)
    bbox: tuple[float, float, float, float] | None = None
    polygon: list[tuple[float, float]] | None = None
    coordinate_space: Literal["rendered_page_pixels", "cad_projected_pixels"] = "rendered_page_pixels"
    localization_source: Literal[
        "unknown", "paddle_layout", "paddle_spotting", "visual_estimate",
        "cad_entity", "pdf_vector",
    ] = "unknown"
    region_id: str | None = None
    view_id: str | None = None

    @model_validator(mode="after")
    def valid_geometry(self):
        if self.bbox and not (0 <= self.bbox[0] < self.bbox[2] and
                              0 <= self.bbox[1] < self.bbox[3]):
            raise ValueError("Invalid bbox")
        if self.polygon is not None and len(self.polygon) < 3:
            raise ValueError("Polygon needs at least three points")
        return self


class Entity(StrictModel):
    id: str
    kind: str
    name: str | None = None
    raw_text: str | None = None
    normalized_text: str | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    locations: list[Location] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    raw_result_refs: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_source: str | None = None
    uncertain: bool = True
    needs_review: bool = True
    method: str

    @model_validator(mode="after")
    def standard_attributes(self):
        self.attributes = {**dict.fromkeys(ATTRIBUTE_FIELDS.get(self.kind, ())), **self.attributes}
        if self.kind == "technical_notes" and self.attributes.get("note_type") == "manufacturing_task_candidate":
            self.attributes = ManufacturingTask.model_validate(self.attributes).model_dump(mode="json")
        return self


class ManufacturingTask(StrictModel):
    note_type: Literal["manufacturing_task_candidate"]
    action: str | None = None
    phase: Literal["rough", "finish", "intermediate", "unknown"] = "unknown"
    responsibility: Literal["current_part", "joining", "assembly_processing", "downstream", "unknown"] = "unknown"
    evidence_basis: Literal["explicit_annotation", "visual_hypothesis"]
    evidence_summary: str
    target_description: str | None = None
    precondition: str | None = None
    postcondition: str | None = None
    unresolved: list[str] = Field(default_factory=list)
    eligible_for_training: Literal[False] = False
    physical_target_verified: Literal[False] = False


class MetadataValue(StrictModel):
    value: JsonValue = None
    candidates: list[Entity] = Field(default_factory=list)
    uncertain: bool = True
    needs_review: bool = True


class Relationship(StrictModel):
    id: str
    source_id: str
    target_id: str
    type: Literal["dimension_of", "tolerance_of", "datum_of", "balloon_of",
                  "bom_item_of", "part_of", "annotation_in", "affects",
                  "view_of", "corresponding_view", "feature_of",
                  "leader_of", "ends_at", "landing_on"]
    evidence_ids: list[str]
    raw_result_refs: list[str]
    page: int | None = Field(default=None, ge=1)
    source_bbox: tuple[float, float, float, float] | None = None
    scope_status: Literal[
        "current_part", "child_part", "assembly_context", "page_layout", "unassigned"
    ] = "unassigned"
    decision_status: Literal[
        "observed", "supported", "conflicted", "unresolved", "rejected"
    ] = "observed"
    method: str = "unknown"
    basis: Literal[
        "visible_leader", "visible_enclosure", "visible_section",
        "explicit_wording", "native_geometry", "unresolved",
    ] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    uncertain: bool = True
    needs_review: bool = True

    @model_validator(mode="after")
    def validate_source_bbox(self):
        if self.source_bbox is not None and not (
            0 <= self.source_bbox[0] < self.source_bbox[2]
            and 0 <= self.source_bbox[1] < self.source_bbox[3]
        ):
            raise ValueError("Invalid relationship source_bbox")
        return self


class Page(StrictModel):
    page_number: int = Field(ge=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    image_path: str
    image_sha256: str
    pdf_cropbox: list[float]
    pdf_rotation: int
    pixel_to_pdf_matrix: list[float] = Field(min_length=6, max_length=6)
    render_policy: Literal[
        "original_pdf_visible_v1", "original_pdf_right_edge_recovery_v1",
    ] = "original_pdf_visible_v1"
    source_mediabox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    render_mediabox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    coordinate_space: Literal["rendered_page_pixels", "cad_projected_pixels"] = "rendered_page_pixels"
    cad_to_pixel_matrix: list[float] | None = Field(default=None, min_length=6, max_length=6)
    paddle_status: Literal["pending", "ok", "partial", "failed"] = "pending"
    semantic_status: Literal["not_requested", "ok", "partial", "failed"] = "not_requested"

    @model_validator(mode="after")
    def valid_coordinate_system(self):
        if self.coordinate_space == "cad_projected_pixels" and self.cad_to_pixel_matrix is None:
            raise ValueError("CAD page requires an explicit CAD-to-page transform")
        return self


class DrawingDocument(StrictModel):
    schema_version: Literal["1.0.0", "1.1.0", "1.2.0"] = "1.2.0"
    id: str
    source_path: str
    source_sha256: str
    extraction_status: Literal["partial", "extracted_unverified"] = "partial"
    configuration: dict[str, JsonValue] = Field(default_factory=dict)
    pages: list[Page] = Field(default_factory=list)
    drawing_metadata: dict[str, MetadataValue] = Field(
        default_factory=lambda: {key: MetadataValue() for key in METADATA_FIELDS})
    assemblies: list[Entity] = Field(default_factory=list)
    parts: list[Entity] = Field(default_factory=list)
    sub_parts: list[Entity] = Field(default_factory=list)
    bom: list[Entity] = Field(default_factory=list)
    views: list[Entity] = Field(default_factory=list)
    features: list[Entity] = Field(default_factory=list)
    balloons: list[Entity] = Field(default_factory=list)
    leader_segments: list[Entity] = Field(default_factory=list)
    leader_endpoints: list[Entity] = Field(default_factory=list)
    geometry_objects: list[Entity] = Field(default_factory=list)
    dimensions: list[Entity] = Field(default_factory=list)
    tolerances: list[Entity] = Field(default_factory=list)
    gdt: list[Entity] = Field(default_factory=list)
    datums: list[Entity] = Field(default_factory=list)
    surface_finish: list[Entity] = Field(default_factory=list)
    threads: list[Entity] = Field(default_factory=list)
    welding_symbols: list[Entity] = Field(default_factory=list)
    technical_notes: list[Entity] = Field(default_factory=list)
    regions: list[Entity] = Field(default_factory=list)
    raw_ocr: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    issues: list[dict[str, JsonValue]] = Field(default_factory=list)
    coverage: dict[str, str] = Field(default_factory=lambda: {
        key: "not_assessed" for key in (*COLLECTIONS, "drawing_metadata", "relationships")})

    def entities(self) -> list[Entity]:
        return [entity for key in COLLECTIONS for entity in getattr(self, key)] + [
            e for field in self.drawing_metadata.values() for e in field.candidates]

    @model_validator(mode="after")
    def validate_graph(self):
        if set(self.drawing_metadata) != set(METADATA_FIELDS):
            raise ValueError("Metadata field contract mismatch")
        for key, field in self.drawing_metadata.items():
            if any(e.kind != "metadata" or e.attributes.get("field") != key for e in field.candidates):
                raise ValueError("Metadata candidate does not match field")
        for key in COLLECTIONS:
            if any(entity.kind != key for entity in getattr(self, key)):
                raise ValueError("Entity kind does not match collection")
        entities = self.entities()
        ids = {entity.id for entity in entities}
        all_ids = [self.id] + [e.id for e in entities] + [r.id for r in self.relationships]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Duplicate stable ID")
        pages = {p.page_number: p for p in self.pages}
        if len(pages) != len(self.pages):
            raise ValueError("Duplicate page number")
        kinds = {e.id: e.kind for e in entities}
        by_id = {e.id: e for e in entities}
        evidence_ids = {e.id for e in self.regions + self.raw_ocr}
        for entity in entities:
            if not set(entity.evidence_ids) <= evidence_ids:
                raise ValueError("Dangling entity evidence")
            for loc in entity.locations:
                if loc.page_number not in pages:
                    raise ValueError("Unknown page")
                page = pages[loc.page_number]
                if loc.coordinate_space != page.coordinate_space:
                    raise ValueError("Location and page coordinate spaces differ")
                if loc.bbox and (loc.bbox[2] > page.width or loc.bbox[3] > page.height):
                    raise ValueError("Bbox outside page")
                if loc.polygon and any(not (0 <= x <= page.width and 0 <= y <= page.height)
                                       for x, y in loc.polygon):
                    raise ValueError("Polygon outside page")
                for ref, kind in ((loc.view_id, "views"), (loc.region_id, "regions")):
                    if ref and kinds.get(ref) != kind:
                        raise ValueError("Invalid spatial reference")
        hierarchy: dict[str, list[str]] = {}
        for rel in self.relationships:
            if rel.source_id not in ids or rel.target_id not in ids:
                raise ValueError("Dangling relationship")
            if rel.source_id == rel.target_id or not set(rel.evidence_ids) <= evidence_ids:
                raise ValueError("Invalid relationship evidence or self-reference")
            allowed = {
                "dimension_of": ({"dimensions", "threads"}, {"features", "parts", "sub_parts", "assemblies"}),
                "tolerance_of": ({"tolerances"}, {"dimensions", "threads"}),
                "datum_of": ({"datums"}, {"gdt"}),
                "balloon_of": ({"balloons"}, {"bom"}),
                "bom_item_of": ({"bom"}, {"parts", "sub_parts", "assemblies"}),
                "annotation_in": ({"dimensions", "tolerances", "gdt", "datums", "surface_finish", "threads", "welding_symbols", "technical_notes", "balloons"}, {"views"}),
                "affects": ({"technical_notes", "surface_finish", "welding_symbols", "gdt"}, {"features", "parts", "sub_parts", "assemblies"}),
                "view_of": ({"views"}, {"parts", "sub_parts", "assemblies"}),
                "corresponding_view": ({"views"}, {"views"}),
                "feature_of": ({"features"}, {"parts", "sub_parts", "assemblies"}),
                "leader_of": ({"leader_segments"}, {"balloons"}),
                "ends_at": ({"leader_segments"}, {"leader_endpoints"}),
                "landing_on": ({"leader_endpoints"}, {
                    "geometry_objects", "features", "parts", "sub_parts", "assemblies",
                }),
            }
            if rel.type in allowed:
                source_kinds, target_kinds = allowed[rel.type]
                if kinds[rel.source_id] not in source_kinds or kinds[rel.target_id] not in target_kinds:
                    raise ValueError("Invalid relationship endpoint types")
            if rel.type == "balloon_of":
                balloon, row = by_id[rel.source_id], by_id[rel.target_id]
                number = balloon.attributes.get("item_number") or balloon.normalized_text or balloon.raw_text
                item = row.attributes.get("item_number")
                if number is None or item is None or str(number).strip() != str(item).strip():
                    raise ValueError("Balloon number does not match BOM item")
            if rel.type == "part_of":
                if kinds[rel.source_id] not in {"parts", "sub_parts", "assemblies"} or kinds[rel.target_id] not in {"parts", "sub_parts", "assemblies"}:
                    raise ValueError("Invalid hierarchy endpoint type")
                hierarchy.setdefault(rel.source_id, []).append(rel.target_id)
        visiting, visited = set(), set()
        def visit(node):
            if node in visiting:
                raise ValueError("Hierarchy cycle")
            if node in visited:
                return
            visiting.add(node)
            for parent in hierarchy.get(node, []):
                visit(parent)
            visiting.remove(node)
            visited.add(node)
        for node in hierarchy:
            visit(node)
        return self
