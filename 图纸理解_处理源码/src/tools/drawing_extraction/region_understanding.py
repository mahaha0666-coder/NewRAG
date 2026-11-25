"""Source-bound regional completeness and cell refinement, isolated from baseline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import pymupdf

from .document_schema import DrawingDocument
from .table_refinement import header_band, intersection_fraction, observations, refine_table, table_rotation

REVISION = "region_understanding_v1"
REPORT = "region_understanding.json"
Progress = Callable[[str, dict[str, Any]], None]


def inventory(document: DrawingDocument) -> list[dict[str, Any]]:
    pages = {page.page_number: page for page in document.pages}
    page_tokens = {number: observations(document, page) for number, page in pages.items()}
    records = []
    for entity in document.regions:
        for location in entity.locations:
            if not location.bbox:
                continue
            page = pages[location.page_number]
            box = list(location.bbox)
            tokens = [token for token in page_tokens[page.page_number]
                      if intersection_fraction(token["bbox"], box) >= .8]
            label = str(entity.attributes.get("label") or "unknown")
            is_parts = label == "table" and (bool(header_band(tokens)) or table_rotation(tokens, page) != 0 or (
                "序号" in (entity.raw_text or "") and any(value in (entity.raw_text or "") for value in ("代号", "图号", "零件编号"))))
            pdf_box = list(pymupdf.Rect(box) * pymupdf.Matrix(*page.pixel_to_pdf_matrix))
            reasons = [] if tokens else ["detected_region_has_no_positioned_text"]
            if label == "table" and not is_parts:
                reasons.append("table_structure_requires_review")
            records.append({
                "id": entity.id, "page": page.page_number, "kind": "part_table" if is_parts else label,
                "source_box": box, "pdf_box": pdf_box, "source_sha256": document.source_sha256,
                "raw_text": entity.raw_text, "evidence_ids": [entity.id, *[token["source_id"] for token in tokens]],
                "raw_result_refs": entity.raw_result_refs, "recognition_confidence": entity.confidence,
                "relation_confidence": None, "scope_status": "unresolved",
                "status": "review_required" if reasons else "extracted", "text_count": len(tokens),
                "attempts": [], "review_reasons": reasons,
            })
    return records


def refine(document: DrawingDocument, artifact: Path, endpoints: list[str], *,
           progress: Progress | None = None) -> dict[str, Any]:
    if not endpoints or len(endpoints) > 2 or len(set(endpoints)) != len(endpoints):
        raise ValueError("Use one or two distinct bounded OCR endpoints")
    started = time.monotonic()
    page_by_number = {page.page_number: page for page in document.pages}
    regions = inventory(document)
    tables = []
    for region in regions:
        if region["kind"] != "part_table":
            continue
        if any(other["page"] == region["page"]
               and intersection_fraction(region["source_box"], other["source_box"]) > .9
               for other in tables):
            region["status"] = "review_required"
            region["review_reasons"].append("overlapping_table_candidate")
        else:
            tables.append(region)
    outcomes = []

    def execute(region: dict[str, Any], endpoint: str) -> dict[str, Any]:
        page = page_by_number[region["page"]]
        try:
            return refine_table(document, page, region, endpoint=endpoint, artifact=artifact)
        except (OSError, ValueError, RuntimeError) as error:
            return {"id": region["id"], "page": region["page"], "source_box": region["source_box"],
                    "rows": [], "attempts": [], "review_reasons": ["table_refinement_failed"],
                    "diagnostic": f"{type(error).__name__}: {error}"}

    # Each lane owns one endpoint, so a slow request cannot queue work on the same GPU twice.
    def lane(index: int) -> list[dict[str, Any]]:
        return [execute(region, endpoints[index]) for region in tables[index::len(endpoints)]]

    if progress:
        progress("region_refinement", {"regions_detected": len(regions), "tables_detected": len(tables)})
    with ThreadPoolExecutor(max_workers=len(endpoints), thread_name_prefix="region-reader") as pool:
        for batch in pool.map(lane, range(len(endpoints))):
            outcomes.extend(batch)
    outcomes.sort(key=lambda row: (row["page"], row["id"]))
    by_id = {outcome["id"]: outcome for outcome in outcomes}
    pages = []
    for page in document.pages:
        page_regions = [region for region in regions if region["page"] == page.page_number]
        for region in page_regions:
            outcome = by_id.get(region["id"])
            if outcome is not None:
                region["detected_source_box"] = region["source_box"]
                region["source_box"] = outcome["source_box"]
                region["pdf_box"] = list(pymupdf.Rect(outcome["source_box"]) * pymupdf.Matrix(*page.pixel_to_pdf_matrix))
                region.update(attempts=outcome["attempts"], review_reasons=outcome["review_reasons"],
                              field_count=sum(len(row["field_evidence"]) for row in outcome["rows"]),
                              status="review_required" if outcome["review_reasons"] else "extracted")
        page_tables = [outcome for outcome in outcomes if outcome["page"] == page.page_number]
        rows = [row for table in page_tables for row in table["rows"]]
        fields = [field for row in rows for field in row["field_evidence"].values()]
        pages.append({"page": page.page_number, "image_size": [page.width, page.height],
                      "pixel_to_pdf_matrix": page.pixel_to_pdf_matrix, "regions": page_regions,
                      "tables": page_tables, "part_rows": rows,
                      "metrics": {"region_count": len(page_regions), "field_count": len(fields),
                                  "field_states": dict(Counter(field["status"] for field in fields)),
                                  "review_region_count": sum(bool(region["review_reasons"]) for region in page_regions)}})
    report = {"schema_version": "region_understanding.v1", "revision": REVISION,
              "source_sha256": document.source_sha256, "pages": pages,
              "status": "review_required" if any(page["metrics"]["review_region_count"] for page in pages) else "extracted",
              "timings": {"refinement_seconds": round(time.monotonic() - started, 3)},
              "policy": {"max_attempts_per_cell": 3, "supplement_dpi": [400, 600],
                         "overlap": .12, "historical_ocr_reused": False,
                         "native_text_is_verified": False, "accuracy_verified": False}}
    data = json.dumps(report, ensure_ascii=False, indent=2).encode()
    (artifact / REPORT).write_bytes(data)
    document.configuration["region_understanding"] = {
        "revision": REVISION, "report_ref": REPORT, "report_sha256": hashlib.sha256(data).hexdigest()}
    return report


def load_report(document: DrawingDocument) -> dict[str, Any] | None:
    configuration = document.configuration.get("region_understanding")
    if not isinstance(configuration, dict):
        return None
    if configuration.get("report_ref") != REPORT or not document.pages:
        raise ValueError("Invalid region understanding report reference")
    data = (Path(document.pages[0].image_path).parent / REPORT).read_bytes()
    if hashlib.sha256(data).hexdigest() != configuration.get("report_sha256"):
        raise ValueError("Region understanding report checksum mismatch")
    report = json.loads(data)
    if report.get("source_sha256") != document.source_sha256 or report.get("revision") != REVISION:
        raise ValueError("Region understanding source or revision mismatch")
    return report


def attach_page_analysis(analysis: dict[str, Any], report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return analysis
    if analysis["source_sha256"] != report["source_sha256"]:
        raise ValueError("Region understanding analysis source mismatch")
    page = next((page for page in report["pages"] if page["page"] == analysis["page_number"]), None)
    if page is None or page["image_size"] != analysis["image_size"]:
        raise ValueError("Region understanding page geometry mismatch")
    regions = [dict(region) for region in page["regions"]]
    ir = analysis.get("drawing_ir") or {}
    annotations = ir.get("annotations") or []
    for region in regions:
        region["scope_candidates"] = [{"id": view["id"], "label": view.get("label"),
                                       "confidence": None, "basis": "spatial_candidate_only"}
                                      for view in ir.get("views", [])
                                      if view.get("view_bbox") and intersection_fraction(
                                          region["source_box"], view["view_bbox"]) >= .8]
        region["annotation_ids"] = [row["id"] for row in annotations
                                    if row.get("source_box") and intersection_fraction(
                                        row["source_box"], region["source_box"]) >= .8]
    analysis["region_understanding"] = {"status": report["status"], "regions": regions,
        "metrics": page["metrics"], "timings": report["timings"], "accuracy_verified": False}
    repaired = [table for table in page["tables"] if table["rows"]]

    def replaced(row: dict) -> bool:
        box = row.get("source_box")
        identity = (row.get("item_no"), row.get("part_code"))
        if not box:
            return False
        for table in repaired:
            if intersection_fraction(box, table["source_box"]) >= .95:
                return True
            matches = [new for new in table["rows"] if all(identity)
                       and (new.get("item_no"), new.get("part_code")) == identity]
            if len(matches) == 1 and intersection_fraction(table["source_box"], box) >= .95:
                return True
        return False

    retained = [row for row in analysis.get("part_rows", []) if not replaced(row)]
    analysis["part_rows"] = [*retained, *page["part_rows"]]
    if page["metrics"]["review_region_count"]:
        analysis["status"] = "partial"
        reason = "Region evidence requires review"
        if reason not in (analysis.get("partial_reason") or ""):
            analysis["partial_reason"] = "; ".join(filter(None, [analysis.get("partial_reason"), reason]))
    return analysis
