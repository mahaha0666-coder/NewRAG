"""Build a conservative manufacturing intermediate representation.

The IR materializes localized observations and explicit unknowns. It never
reads route labels and never upgrades text proximity to physical ownership.
"""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, JsonValue, model_validator

from .document_schema import DrawingDocument, Entity, StrictModel
from .pdf_modality_audit import audit_pdf
from .requirement_semantics import WELDING_ACTION_PATTERN, parse_requirement_semantics
from .source_watermark_guard import (
    is_source_watermark_fragment,
    source_watermark_glyphs,
    watermark_candidate_size,
)


VerificationStatus = Literal[
    "known_localized",
    "observed_unlocalized",
    "unresolved",
    "unknown",
    "conflicted",
    "source_forbidden",
]

_DIMENSION_TEXT = re.compile(
    r"^(?:(?P<count>\d+)\s*[xX×-]\s*)?(?P<symbol>[⌀ΦφØ∅Rr]|SR)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)"
    r"(?:\s*[±+]\s*\d+(?:\.\d+)?)?(?:\s*[/\\]\s*[-−]?\d+(?:\.\d+)?)?"
    r"(?:\s*(?:mm|°|deg))?(?P<distribution>\s*均布)?$",
    re.IGNORECASE,
)
_THREAD_TEXT = re.compile(
    r"^(?:\d+\s*[xX×-]\s*)?[MG]\s*\d+(?:\.\d+)?"
    r"(?:\s*[xX×]\s*\d+(?:\.\d+)?)?(?:[-－]\w+)?$",
    re.IGNORECASE,
)
_TOLERANCE_TEXT = re.compile(
    r"(?:±\s*\d|[+]\s*\d+(?:\.\d+)?\s*[/\\]\s*[-−]\s*\d|"
    r"ISO\s*2768|GB/?T\s*1804|公差)",
    re.IGNORECASE,
)
_SURFACE_TEXT = re.compile(r"(?:\bRa\s*\d+(?:\.\d+)?|粗糙度)", re.IGNORECASE)
_PADDLE_FORMULA_DIMENSION = re.compile(
    r"^(?P<diameter>\\phi\s*)?(?P<value>\d+(?:\.\d+)?)"
    r"(?:\s*\^\{\s*(?P<upper>[+\-−]\d+(?:\.\d+)?|0)\s*\}"
    r"\s*_\{\s*(?P<lower>[+\-−]\d+(?:\.\d+)?|0)\s*\})?$",
)
_CHAMFER_TEXT = re.compile(r"^C\s*\d+(?:\.\d+)?$", re.IGNORECASE)
_DECIMAL_ONLY_TEXT = re.compile(r"^\d+\.\d+$")
_NEGATIVE_CHAMFER_TEXT = re.compile(r"(?:不可|不得|禁止|不应)\s*倒角")
_CHAMFER_INSTRUCTION_TEXT = re.compile(
    r"(?:条边|铣|加工|图中标注).{0,40}倒角\s*C\s*\d+(?:\.\d+)?",
    re.IGNORECASE | re.DOTALL,
)
_CAPABILITIES = (
    ("joining_welding", WELDING_ACTION_PATTERN),
    ("grind_or_deburr", re.compile(r"磨平|打磨|锐边倒钝|去毛刺|毛刺")),
    ("polishing", re.compile(r"抛光|拉丝")),
    ("coating", re.compile(r"喷塑|喷漆|涂装|镀锌|钝化|防锈")),
    ("heat_treatment", re.compile(r"热处理|淬火|调质|退火|回火")),
    ("inspection", re.compile(r"检验|检测|探伤|100\s*%\s*VT|试验", re.IGNORECASE)),
    ("machining", re.compile(r"车削|铣削|镗|钻|攻丝|磨削|加工")),
)


def _stable_id(*values: object) -> str:
    text = "\x1f".join(str(value) for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _normalized(text: str | None) -> str:
    return "".join(unicodedata.normalize("NFKC", text or "").split()).strip("：:。.;；")


class IRRecord(StrictModel):
    id: str
    kind: str
    raw_text: str | None = None
    normalized_value: JsonValue = None
    page: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None
    source_kind: Literal[
        "paddleocr_vl", "ppocr_v6", "native_pdf_text", "pdf_vector", "dwg_teacher", "derived_audit", "native_dwg"
    ]
    source_ids: list[str] = Field(default_factory=list)
    raw_result_refs: list[str] = Field(default_factory=list)
    verification_status: VerificationStatus
    localization_status: Literal["known_localized", "observed_unlocalized", "unknown"]
    needs_review: bool = True
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class IRRelation(StrictModel):
    id: str
    type: str
    source_id: str
    target_id: str
    evidence_ids: list[str] = Field(default_factory=list)
    page: int | None = Field(default=None, ge=1)
    source_bbox: tuple[float, float, float, float] | None = None
    scope_status: str
    decision_status: str
    method: str
    verification_status: VerificationStatus
    needs_review: bool = True


class ManufacturingIR(StrictModel):
    schema_version: Literal["manufacturing_ir.v1"] = "manufacturing_ir.v1"
    source_path: str
    source_sha256: str
    drawing_document_id: str
    extraction_model: Literal["PaddleOCR-VL-1.6", "PP-OCRv6", "DWG-native"] = "PaddleOCR-VL-1.6"
    route_label_access: Literal[False] = False
    page_coverage: list[dict[str, JsonValue]] = Field(default_factory=list)
    metadata: list[IRRecord] = Field(default_factory=list)
    bom: list[IRRecord] = Field(default_factory=list)
    objects: list[IRRecord] = Field(default_factory=list)
    views: list[IRRecord] = Field(default_factory=list)
    features: list[IRRecord] = Field(default_factory=list)
    annotations: list[IRRecord] = Field(default_factory=list)
    measurements: list[IRRecord] = Field(default_factory=list)
    requirements: list[IRRecord] = Field(default_factory=list)
    capability_needs: list[IRRecord] = Field(default_factory=list)
    relationships: list[IRRelation] = Field(default_factory=list)
    operation_instances: list[IRRecord] = Field(default_factory=list)
    precedence_constraints: list[IRRelation] = Field(default_factory=list)
    field_coverage: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    unresolved_gaps: list[dict[str, JsonValue]] = Field(default_factory=list)
    audit: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_locations(self):
        page_sizes = {
            int(row["page"]): (float(row["width"]), float(row["height"]))
            for row in self.page_coverage
            if row.get("page") is not None and row.get("width") and row.get("height")
        }
        for record in [
            *self.metadata,
            *self.bom,
            *self.objects,
            *self.views,
            *self.features,
            *self.annotations,
            *self.measurements,
            *self.requirements,
            *self.capability_needs,
        ]:
            if record.bbox is None:
                continue
            size = page_sizes.get(record.page or -1)
            if size is None:
                raise ValueError("Localized IR record references an unknown page")
            x0, y0, x1, y1 = record.bbox
            if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
                raise ValueError("IR bbox outside rendered page")
        return self


def _first_location(entity: Entity) -> tuple[int | None, tuple[float, float, float, float] | None]:
    localized = next((location for location in entity.locations if location.bbox is not None), None)
    if localized is not None:
        return localized.page_number, localized.bbox
    if entity.locations:
        return entity.locations[0].page_number, None
    return None, None


def _record(entity: Entity, *, kind: str | None = None,
            status: VerificationStatus | None = None) -> IRRecord:
    page, bbox = _first_location(entity)
    conflicted = bool(entity.attributes.get("ocr_variant_disagreement"))
    resolved_status: VerificationStatus = status or (
        "conflicted" if conflicted else "known_localized" if bbox is not None
        else "observed_unlocalized" if entity.raw_text else "unknown"
    )
    return IRRecord(
        id=entity.id,
        kind=kind or entity.kind,
        raw_text=entity.raw_text,
        normalized_value=entity.attributes,
        page=page,
        bbox=bbox,
        source_kind="dwg_teacher" if entity.method.startswith("cad_") else "paddleocr_vl",
        source_ids=sorted({entity.id, *entity.evidence_ids}),
        raw_result_refs=sorted(set(entity.raw_result_refs)),
        verification_status=resolved_status,
        localization_status=("known_localized" if bbox is not None
                             else "observed_unlocalized" if entity.raw_text else "unknown"),
        needs_review=entity.needs_review,
        attributes={"method": entity.method, "uncertain": entity.uncertain},
    )


def _contains(container: tuple[float, float, float, float],
              candidate: tuple[float, float, float, float]) -> bool:
    center_x = (candidate[0] + candidate[2]) / 2
    center_y = (candidate[1] + candidate[3]) / 2
    return container[0] <= center_x <= container[2] and container[1] <= center_y <= container[3]


def _substantial_box_overlap(first: tuple[float, float, float, float],
                             second: tuple[float, float, float, float]) -> bool:
    intersection = (max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
                    * max(0.0, min(first[3], second[3]) - max(first[1], second[1])))
    smaller = min((first[2] - first[0]) * (first[3] - first[1]),
                  (second[2] - second[0]) * (second[3] - second[1]))
    return smaller > 0 and intersection >= .5 * smaller


class _EmbeddedImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.has_image = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "img":
            self.has_image = True

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def _coarse_embedded_drawing_tables(
    document: DrawingDocument,
) -> set[tuple[int, tuple[float, float, float, float]]]:
    pages = {page.page_number: page for page in document.pages}
    tables = set()
    for region in document.regions:
        if region.attributes.get("label") != "table" or not region.raw_text:
            continue
        parser = _EmbeddedImageParser()
        parser.feed(region.raw_text)
        if not parser.has_image:
            continue
        for location in region.locations:
            page = pages.get(location.page_number)
            box = location.bbox
            if page is None or box is None:
                continue
            width, height = box[2] - box[0], box[3] - box[1]
            if (width * height >= .35 * page.width * page.height
                    and height >= .8 * page.height
                    and box[1] <= .02 * page.height
                    and box[3] >= .98 * page.height):
                tables.add((page.page_number, box))
    return tables


def coarse_embedded_drawing_table_pages(document: DrawingDocument) -> set[int]:
    """Pages where a page-scale OCR table swallowed embedded drawing imagery."""
    return {page for page, _box in _coarse_embedded_drawing_tables(document)}


def _native_match(text: str | None, page: int | None, native_pdf: dict[str, Any]) -> list[str]:
    needle = _normalized(text)
    if not needle or page is None:
        return []
    page_row = next((row for row in native_pdf["pages"] if row["page"] == page), None)
    if page_row is None:
        return []
    return [
        f"native_pdf.json#/pages/{page - 1}/text_spans/{index}"
        for index, span in enumerate(page_row["text_spans"])
        if needle == _normalized(span.get("text"))
    ]


def _paddle_formula_dimension(raw_text: str) -> dict[str, str] | None:
    text = raw_text.strip()
    if text.startswith("$") or text.endswith("$"):
        if not (text.startswith("$") and text.endswith("$") and len(text) > 2):
            return None
        text = text[1:-1].strip()
    if "$" in text:
        return None
    match = _PADDLE_FORMULA_DIMENSION.fullmatch(text)
    if match is None or not (match.group("diameter") or match.group("upper")):
        return None
    parsed = {
        "value": match.group("value"),
        "formula_parse_status": "structured_candidate",
        "scope_status": "unassigned",
    }
    if match.group("diameter"):
        parsed["symbol"] = "diameter"
    if match.group("upper") is not None:
        parsed["upper_deviation"] = match.group("upper").replace("−", "-")
        parsed["lower_deviation"] = match.group("lower").replace("−", "-")
    return parsed


def _measurement_candidates(document: DrawingDocument, view_records: list[IRRecord]) -> list[IRRecord]:
    view_boxes = [(record.page, record.bbox, record.id) for record in view_records if record.bbox]
    coarse_tables = _coarse_embedded_drawing_tables(document)
    page_heights = {page.page_number: page.height for page in document.pages}
    page_sizes = {page.page_number: (page.width, page.height) for page in document.pages}
    watermark_glyphs: dict[int, list[tuple[str, float, float]]] | None = None
    table_boxes = [
        (location.page_number, location.bbox,
         (location.page_number, location.bbox) in coarse_tables)
        for region in document.regions if region.attributes.get("label") == "table"
        for location in region.locations if location.bbox is not None
    ]
    candidates: list[IRRecord] = []
    seen: list[tuple[str, int, tuple[float, float, float, float], IRRecord]] = []
    for observation in document.raw_ocr:
        text = (observation.raw_text or "").strip().replace("$", "").strip()
        formula = _paddle_formula_dimension(observation.raw_text or "")
        chamfer = bool(_CHAMFER_TEXT.fullmatch(text))
        dimension = _DIMENSION_TEXT.fullmatch(text)
        if not (formula or dimension or _THREAD_TEXT.fullmatch(text)
                or _TOLERANCE_TEXT.search(text) or _SURFACE_TEXT.search(text)
                or chamfer):
            continue
        page, bbox = _first_location(observation)
        if page is None or bbox is None:
            continue
        page_size = page_sizes.get(page)
        if page_size and watermark_candidate_size(bbox, page_size):
            if watermark_glyphs is None:
                watermark_glyphs = source_watermark_glyphs(document)
            if is_source_watermark_fragment(
                text, bbox, watermark_glyphs.get(page, []),
            ):
                continue
        containing_tables = [coarse for table_page, table_box, coarse in table_boxes
                             if table_page == page and _contains(table_box, bbox)]
        if (False in containing_tables
                or containing_tables and (bbox[1] + bbox[3]) / 2 >=
                .7 * page_heights[page]):
            continue
        in_coarse_table = bool(containing_tables)
        containing_views = [view_id for view_page, view_box, view_id in view_boxes
                            if view_page == page and view_box is not None and _contains(view_box, bbox)]
        pass_id = str(observation.attributes.get("recognition_pass") or "")
        if formula and not containing_views:
            if observation.attributes.get("localization") != "text_polygon":
                continue
        elif not containing_views and "major_view" not in pass_id:
            page_row = next((row for row in document.pages if row.page_number == page), None)
            has_page_views = any(view_page == page for view_page, _, _ in view_boxes)
            numeric_candidate = bool(re.fullmatch(r"\d{2,4}(?:\.\d+)?(?:\s*[+±]\s*\d+(?:\.\d+)?)?", text))
            geometric_symbol = bool(dimension and dimension.group("symbol") in {
                "⌀", "Φ", "φ", "Ø", "∅", "R", "r", "SR",
            })
            if (has_page_views or page_row is None
                    or (bbox[1] + bbox[3]) / 2 >= .7 * page_row.height
                    or not (numeric_candidate or chamfer or _THREAD_TEXT.fullmatch(text)
                            or in_coarse_table and geometric_symbol)):
                continue
        normalized = _normalized(text)
        duplicate = next((previous_record for previous_text, previous_page,
                          previous_bbox, previous_record in seen
                          if previous_text == normalized and previous_page == page
                          and _substantial_box_overlap(previous_bbox, bbox)), None)
        if duplicate is not None:
            duplicate.source_ids = sorted({*duplicate.source_ids, observation.id,
                                           *observation.evidence_ids})
            duplicate.raw_result_refs = sorted({*duplicate.raw_result_refs,
                                                *observation.raw_result_refs})
            continue
        kind = ("dimension" if formula else
                "thread" if _THREAD_TEXT.fullmatch(text) else
                "surface_finish" if _SURFACE_TEXT.search(text) else
                "tolerance" if _TOLERANCE_TEXT.search(text) else "dimension")
        record = _record(observation, kind=kind, status="unresolved")
        if formula:
            record.normalized_value = {**observation.attributes, **formula}
            record.needs_review = True
        elif dimension and dimension.group("symbol") in {"⌀", "Φ", "φ", "Ø", "∅"}:
            parsed: dict[str, JsonValue] = {
                "dimension_type": "diameter",
                "value": dimension.group("value"),
                "scope_status": "unassigned",
            }
            if dimension.group("count"):
                parsed["count"] = int(dimension.group("count"))
            if dimension.group("distribution"):
                parsed["distribution"] = "equally_spaced"
            record.normalized_value = {**observation.attributes, **parsed}
            record.needs_review = True
        record.attributes.update({
            "candidate_basis": ("dimension_like_text_inside_view" if containing_views
                                else "dimension_like_text_without_view"),
            "candidate_view_ids": containing_views,
            "feature_association_status": "unresolved",
        })
        if formula:
            record.attributes.update({"formula_candidate": True,
                                      "scope_status": "unassigned", "hard_constraint": False})
        if chamfer:
            record.attributes["candidate_subtype"] = "chamfer"
        candidates.append(record)
        seen.append((normalized, page, bbox, record))
    return candidates


def _raw_chamfer_requirements(
    document: DrawingDocument, existing: list[IRRecord],
) -> list[IRRecord]:
    observations = sorted(document.raw_ocr, key=lambda row: len(row.raw_text or ""), reverse=True)
    candidates: list[IRRecord] = []
    for observation in observations:
        text = (observation.raw_text or "").strip()
        negative = bool(_NEGATIVE_CHAMFER_TEXT.search(text))
        if not (negative or _CHAMFER_INSTRUCTION_TEXT.search(text)):
            continue
        page, bbox = _first_location(observation)
        if page is None or bbox is None:
            continue
        if any(
            previous.page == page and previous.bbox is not None
            and _contains(previous.bbox, bbox)
            and _normalized(text) in _normalized(previous.raw_text)
            for previous in [*existing, *candidates]
        ):
            continue
        record = _record(observation, kind="technical_notes", status="unresolved")
        normalized = dict(record.normalized_value) if isinstance(
            record.normalized_value, dict
        ) else {}
        normalized.update({
            "note_type": "chamfer_instruction_candidate",
            "scope_status": "unassigned",
            "polarity": "negative" if negative else "positive",
            "physical_target_binding_status": "unresolved",
            "eligible_for_training": False,
        })
        record.normalized_value = normalized
        record.attributes.update({
            "candidate_basis": "localized_raw_ocr_chamfer_instruction",
            "scope_status": "unassigned",
            "hard_constraint": False,
        })
        candidates.append(record)
    return candidates


def build_manufacturing_ir(
    document: DrawingDocument,
    native_pdf: dict[str, Any],
    paddle_payloads: list[dict[str, Any]],
    *,
    additional_metadata_candidates: list[dict[str, Any]] | None = None,
) -> ManufacturingIR:
    """Build label-free ManufacturingIR from one frozen drawing snapshot."""

    page_coverage = []
    for page, payload, native_page in zip(document.pages, paddle_payloads, native_pdf["pages"]):
        ledger = payload.get("coverage_ledger") or {}
        unresolved_errors = [error for error in payload.get("errors") or [] if not error.get("resolved")]
        page_coverage.append({
            "page": page.page_number,
            "width": page.width,
            "height": page.height,
            "image_sha256": page.image_sha256,
            "pixel_to_pdf_matrix": page.pixel_to_pdf_matrix,
            "render_policy": page.render_policy,
            "source_mediabox": page.source_mediabox,
            "render_mediabox": page.render_mediabox,
            "pdf_modality": native_page["modality"],
            "paddle_status": payload.get("status", "failed"),
            "processing_frame": ledger.get("processing_frame"),
            "processing_area_ratio": ledger.get("processing_area_ratio"),
            "layout_ink_coverage": ledger.get("layout_ink_coverage"),
            "final_ink_coverage": ledger.get("final_ink_coverage"),
            "coverage_gap_count": len(ledger.get("unselected_coverage_gaps") or []),
            "unresolved_error_count": len(unresolved_errors),
            "discarded_incomplete_completion_count": (
                payload.get("completion_summary") or {}
            ).get("discarded_incomplete", 0),
            "all_regions_accounted": bool(
                ledger.get("all_detected_gaps_accounted")
                and not ledger.get("unselected_coverage_gaps")
                and not unresolved_errors
                and not ledger.get("incomplete_targeted_region_count")
            ),
            "coverage_ref": f"paddle_{page.page_number:04d}.json#/coverage_ledger",
        })

    metadata = [
        _record(candidate, kind=f"metadata.{field_name}")
        for field_name, field in document.drawing_metadata.items()
        for candidate in field.candidates
    ]
    for candidate in additional_metadata_candidates or []:
        source_ids = [candidate["region_id"], *candidate["spotting_entity_ids"]]
        raw_refs = [candidate["region_ref"], *candidate["spotting_raw_refs"]]
        metadata.append(IRRecord(
            id="metadata_" + _stable_id(
                document.source_sha256,
                candidate["field"],
                candidate["value"],
                candidate["page"],
                candidate["value_cell"],
            ),
            kind=f"metadata.{candidate['field']}",
            raw_text=candidate["value"],
            normalized_value=candidate["value"],
            page=candidate["page"],
            bbox=candidate["bbox"],
            source_kind="paddleocr_vl",
            source_ids=list(dict.fromkeys(source_ids)),
            raw_result_refs=list(dict.fromkeys(ref for ref in raw_refs if ref)),
            verification_status="unresolved",
            localization_status="known_localized",
            needs_review=True,
            attributes={
                "field": candidate["field"],
                "label_text": candidate["label_text"],
                "label_cell": candidate["label_cell"],
                "value_cell": candidate["value_cell"],
                "candidate_rank": candidate["candidate_rank"],
                "cross_source_count": candidate["cross_source_count"],
                "localization_granularity": candidate["localization_granularity"],
                "method": "paddle_title_table_candidate_v1",
            },
        ))
    bom = [_record(entity) for entity in document.bom]
    objects = [_record(entity) for collection in (
        document.assemblies, document.parts, document.sub_parts, document.geometry_objects
    ) for entity in collection]
    views = []
    for region in document.regions:
        if region.attributes.get("label") != "image":
            continue
        page, bbox = _first_location(region)
        views.append(IRRecord(
            id="view_" + _stable_id(document.source_sha256, region.id),
            kind="view",
            raw_text=region.raw_text,
            normalized_value={"view_type": "unknown"},
            page=page,
            bbox=bbox,
            source_kind="paddleocr_vl",
            source_ids=[region.id],
            raw_result_refs=region.raw_result_refs,
            verification_status="unresolved",
            localization_status="known_localized" if bbox else "observed_unlocalized",
            attributes={"semantic_type_status": "unknown", "source_region_label": "image"},
        ))
    features = [_record(entity, status="unresolved") for entity in [
        *document.features, *document.threads,
    ]]
    annotations = [_record(entity, status="unresolved") for entity in [
        *document.balloons, *document.leader_segments, *document.leader_endpoints,
    ]]
    measurements = [_record(entity) for entity in [
        *document.dimensions, *document.datums,
    ]]
    measurements.extend(_measurement_candidates(document, views))
    requirements = [_record(entity, status=(
        "unresolved" if entity.attributes.get("scope_status", "unassigned") == "unassigned" else None
    )) for entity in [
        *document.technical_notes, *document.welding_symbols, *document.surface_finish,
        *document.gdt, *document.tolerances,
    ] if not (entity.kind == "technical_notes"
              and _DECIMAL_ONLY_TEXT.fullmatch((entity.raw_text or "").strip()))]
    requirements.extend(_raw_chamfer_requirements(document, requirements))
    for record in requirements:
        normalized = dict(record.normalized_value) if isinstance(
            record.normalized_value, dict
        ) else {"source_value": record.normalized_value}
        normalized["manufacturing_semantics"] = parse_requirement_semantics(
            record.raw_text
        )
        record.normalized_value = normalized

    for record in [
        *metadata, *bom, *objects, *views, *features, *annotations, *measurements,
        *requirements,
    ]:
        native_refs = _native_match(record.raw_text, record.page, native_pdf)
        if native_refs:
            record.attributes["native_pdf_exact_text_refs"] = native_refs

    capability_needs = []
    for requirement in requirements:
        for family, pattern in _CAPABILITIES:
            if not pattern.search(requirement.raw_text or ""):
                continue
            capability_needs.append(IRRecord(
                id="capability_" + _stable_id(requirement.id, family),
                kind="capability_need",
                raw_text=requirement.raw_text,
                normalized_value={"capability_family": family},
                page=requirement.page,
                bbox=requirement.bbox,
                source_kind="derived_audit",
                source_ids=[requirement.id],
                raw_result_refs=requirement.raw_result_refs,
                verification_status=("unresolved" if requirement.verification_status in {
                    "unresolved", "unknown", "conflicted"
                } else "known_localized"),
                localization_status=requirement.localization_status,
                attributes={"scope_status": "unassigned", "hard_constraint": False},
            ))

    relations = [IRRelation(
        id=relation.id,
        type=relation.type,
        source_id=relation.source_id,
        target_id=relation.target_id,
        evidence_ids=relation.evidence_ids,
        page=relation.page,
        source_bbox=relation.source_bbox,
        scope_status=relation.scope_status,
        decision_status=relation.decision_status,
        method=relation.method,
        verification_status=("known_localized" if relation.decision_status == "supported"
                             else "conflicted" if relation.decision_status == "conflicted"
                             else "unresolved"),
        needs_review=relation.needs_review,
    ) for relation in document.relationships]

    metadata_fields = {}
    for name in document.drawing_metadata:
        candidates = [record for record in metadata if record.kind == f"metadata.{name}"]
        values = {
            json.dumps(record.normalized_value, ensure_ascii=True, sort_keys=True)
            for record in candidates if record.normalized_value is not None
        }
        metadata_fields[name] = {
            "status": (
                "unknown" if not candidates
                else "conflicted" if len(values) > 1
                else "known_localized" if any(
                    record.verification_status == "known_localized" for record in candidates
                )
                else "unresolved"
            ),
            "candidate_ids": [record.id for record in candidates],
            "cross_source_candidate_count": sum(
                int(record.attributes.get("cross_source_count") or 0) >= 2
                for record in candidates
            ),
        }
    relationship_types = {
        "dimension_controls_feature": "dimension_of",
        "tolerance_applies_to_dimension_or_feature": "tolerance_of",
        "datum_applies_to_object": "datum_of",
        "surface_finish_applies_to_feature_or_object": "affects",
        "technical_requirement_applies_to_part_or_assembly": "affects",
    }
    supported_requirement_ids = {
        relation.source_id
        for relation in relations
        if relation.type == "affects" and relation.decision_status == "supported"
    }
    semantic_requirements = [
        record for record in requirements
        if isinstance(record.normalized_value, dict)
        and isinstance(record.normalized_value.get("manufacturing_semantics"), dict)
        and record.normalized_value["manufacturing_semantics"].get("parse_status")
        == "structured_candidate"
    ]
    target_class_requirements = [
        record for record in semantic_requirements
        if any(
            clause.get("target_class_mentions")
            for clause in record.normalized_value["manufacturing_semantics"].get("clauses", [])
        )
    ]
    unresolved_gaps = []
    for page in page_coverage:
        if not page["all_regions_accounted"]:
            unresolved_gaps.append({
                "kind": "page_coverage_gap",
                "page": page["page"],
                "coverage_gap_count": page["coverage_gap_count"],
                "unresolved_error_count": page["unresolved_error_count"],
            })
    for name, relation_type in relationship_types.items():
        supported = sum(
            relation.type == relation_type and relation.decision_status == "supported"
            for relation in relations
        )
        if not supported:
            unresolved_gaps.append({"kind": name, "status": "unknown", "supported_count": 0})
    unbound_semantic_ids = [
        record.id for record in target_class_requirements
        if record.id not in supported_requirement_ids
    ]
    if unbound_semantic_ids:
        unresolved_gaps.append({
            "kind": "requirement_target_class_not_entity_bound",
            "status": "unresolved",
            "candidate_count": len(unbound_semantic_ids),
            "candidate_ids": unbound_semantic_ids,
        })

    ir = ManufacturingIR(
        source_path=document.source_path,
        source_sha256=document.source_sha256,
        drawing_document_id=document.id,
        page_coverage=page_coverage,
        metadata=metadata,
        bom=bom,
        objects=objects,
        views=views,
        features=features,
        annotations=annotations,
        measurements=measurements,
        requirements=requirements,
        capability_needs=capability_needs,
        relationships=relations,
        field_coverage={
            "title_block": metadata_fields,
            "bom": {"status": "known_localized" if bom else "unknown", "count": len(bom)},
            "views": {"status": "unresolved" if views else "unknown", "count": len(views)},
            "measurements": {
                "status": "unresolved" if measurements else "unknown",
                "count": len(measurements),
            },
            "requirements": {
                "status": "unresolved" if requirements else "unknown",
                "count": len(requirements),
            },
            "requirement_semantics": {
                "status": "unresolved" if semantic_requirements else "unknown",
                "structured_candidate_count": len(semantic_requirements),
                "target_class_candidate_count": len(target_class_requirements),
                "entity_bound_count": sum(
                    record.id in supported_requirement_ids
                    for record in target_class_requirements
                ),
                "hard_constraint_eligible_count": 0,
            },
        },
        unresolved_gaps=unresolved_gaps,
        audit={
            "page_count": len(document.pages),
            "all_pages_accounted": all(row["all_regions_accounted"] for row in page_coverage),
            "supported_relationship_count": sum(
                relation.decision_status == "supported" for relation in relations
            ),
            "unresolved_relationship_count": sum(
                relation.decision_status != "supported" for relation in relations
            ),
            "route_label_access": False,
            "accepted_for_route_hard_constraints": False,
            "structured_requirement_semantics_count": len(semantic_requirements),
            "requirement_target_class_candidate_count": len(
                target_class_requirements
            ),
            "requirement_entity_bound_count": sum(
                record.id in supported_requirement_ids
                for record in target_class_requirements
            ),
        },
    )
    return ir


def materialize_manufacturing_ir(document: DrawingDocument, output: Path) -> ManufacturingIR:
    """Write native evidence, IR, and an automatic audit beside a new snapshot."""

    native_pdf = audit_pdf(Path(document.source_path))
    payloads = []
    for page in document.pages:
        path = output / f"paddle_{page.page_number:04d}.json"
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    ir = build_manufacturing_ir(document, native_pdf, payloads)
    artifacts = {
        "native_pdf.json": native_pdf,
        "manufacturing_ir.json": ir.model_dump(mode="json"),
        "manufacturing_ir.schema.json": ir.model_json_schema(),
        "manufacturing_ir.audit.json": ir.audit,
    }
    for name, value in artifacts.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)
    return ir


__all__ = [
    "IRRecord",
    "IRRelation",
    "ManufacturingIR",
    "build_manufacturing_ir",
    "materialize_manufacturing_ir",
]
