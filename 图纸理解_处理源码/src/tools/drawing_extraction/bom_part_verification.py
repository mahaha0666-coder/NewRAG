"""Reconcile disputed BOM identities with independent recognition of source pixels."""

from __future__ import annotations

import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

from .document_schema import DrawingDocument, Entity
from .document_tables import normalize_part_number


MIN_SCORE = 0.95
FIELDS = ("part_name", "quantity", "material", "specification", "remark")


def conflict_jobs(document: DrawingDocument) -> tuple[list[dict], dict[str, list[Entity]]]:
    grouped = defaultdict(list)
    for candidate in document.bom:
        if candidate.locations:
            grouped[(candidate.locations[0].page_number,
                     str(candidate.attributes.get("item_number")))].append(candidate)
    raw_ocr = {entity.id: entity for entity in document.raw_ocr}
    pages = {page.page_number: page for page in document.pages}
    jobs, eligible = [], {}
    for (page_number, item_number), candidates in grouped.items():
        if len(candidates) < 2 or len({normalize_part_number(str(row.attributes.get("part_number") or ""))
                                       for row in candidates}) < 2:
            continue
        if not any(issue.get("stage") == "bom_ocr_conflict"
                   and set(issue.get("entity_ids", [])) == {row.id for row in candidates}
                   for issue in document.issues):
            document.issues.append({"stage": "bom_ocr_conflict", "page": page_number,
                                    "item_number": item_number, "fields": ["part_number"],
                                    "entity_ids": [row.id for row in candidates], "resolved": False})
        if any(len({str(row.attributes.get(field)) for row in candidates
                    if row.attributes.get(field) is not None}) > 1 for field in FIELDS):
            continue
        if any(row.attributes.get(field) is None for row in candidates
               for field in ("part_name", "quantity")):
            continue
        spatial = [row for row in candidates
                   if row.attributes.get("source_parse_mode") == "ocr_spatial_signature"]
        if len(candidates) != 2 or len(spatial) != 1:
            continue
        row = spatial[0]
        code = normalize_part_number(str(row.attributes["part_number"]))
        observations = [raw_ocr[eid] for eid in row.evidence_ids if eid in raw_ocr
                        and normalize_part_number(raw_ocr[eid].raw_text or "") == code
                        and raw_ocr[eid].attributes.get("localization") == "text_polygon"
                        and len(raw_ocr[eid].locations) == 1
                        and raw_ocr[eid].locations[0].page_number == page_number
                        and raw_ocr[eid].locations[0].bbox is not None]
        if len(observations) != 1:
            continue
        bbox = observations[0].locations[0].bbox
        row_box = row.locations[0].bbox
        if row_box is None or not (row_box[0] <= bbox[0] < bbox[2] <= row_box[2]
                                   and row_box[1] <= bbox[1] < bbox[3] <= row_box[3]):
            continue
        if not all(any(location.page_number == page_number and location.bbox is not None
                       and location.bbox[0] <= (bbox[0] + bbox[2]) / 2 <= location.bbox[2]
                       and location.bbox[1] <= (bbox[1] + bbox[3]) / 2 <= location.bbox[3]
                       for location in candidate.locations)
                   for candidate in candidates if candidate.id != row.id):
            continue
        page = pages[page_number]
        job_id = f"{page_number}:{item_number}"
        jobs.append({"job_id": job_id, "image_path": page.image_path,
                     "image_sha256": page.image_sha256, "image_size": [page.width, page.height],
                     "bbox": list(bbox), "ocr_entity_id": observations[0].id})
        eligible[job_id] = candidates
    return jobs, eligible


def apply_verifications(document: DrawingDocument, jobs: list[dict],
                        eligible: dict[str, list[Entity]], payload: dict,
                        artifact: str = "bom_part_verification.json") -> int:
    if payload.get("model") != "PP-OCRv6_medium_rec":
        raise ValueError("Unexpected independent recognition model")
    results = payload.get("results", [])
    if len(results) != len(jobs) or {result["job_id"] for result in results} != {
            job["job_id"] for job in jobs}:
        raise ValueError("Incomplete or duplicate BOM verification results")
    if any(not all(key in result for key in ("image_sha256", "crop_box", "text", "score"))
           or not isinstance(result["score"], (int, float))
           or not math.isfinite(result["score"]) for result in results):
        raise ValueError("Malformed BOM verification result")
    job_map = {job["job_id"]: job for job in jobs}
    referenced = {identifier for relation in document.relationships
                  for identifier in (relation.source_id, relation.target_id)}
    resolved = 0
    for index, result in enumerate(results):
        job = job_map[result["job_id"]]
        candidates = eligible[result["job_id"]]
        if (result["image_sha256"] != job["image_sha256"]
                or result.get("score", 0) < MIN_SCORE):
            continue
        code = normalize_part_number(result.get("text", ""))
        spatial_code = next((normalize_part_number(str(row.attributes["part_number"]))
                             for row in candidates if row.attributes.get("source_parse_mode")
                             == "ocr_spatial_signature"), None)
        if code != spatial_code:
            continue
        left, top, right, bottom = job["bbox"]
        width, height = job["image_size"]
        expected_crop = [max(0, int(left) - 8), max(0, int(top) - 8),
                         min(width, int(right + 1) + 8), min(height, int(bottom + 1) + 8)]
        if result.get("crop_box") != expected_crop:
            continue
        winners = [row for row in candidates
                   if normalize_part_number(str(row.attributes.get("part_number") or "")) == code]
        if len(winners) != 1:
            continue
        winner = winners[0]
        rejected = [row for row in candidates if row.id != winner.id]
        if any(row.id in referenced for row in rejected):
            continue
        winner.raw_result_refs = sorted(set(winner.raw_result_refs + [f"{artifact}#/results/{index}"]))
        winner.evidence_ids = sorted(set(winner.evidence_ids + [job["ocr_entity_id"]]))
        winner.attributes["part_number_verification"] = {
            "method": "independent_crop_recognition", "model": payload["model"],
            "score": result["score"], "ocr_entity_id": job["ocr_entity_id"],
            "crop_box": result["crop_box"], "image_sha256": job["image_sha256"],
        }
        document.issues.append({"stage": "bom_ocr_conflict_resolved", "page":
                                int(result["job_id"].split(":")[0]),
                                "item_number": winner.attributes["item_number"],
                                "chosen_id": winner.id, "rejected_candidates": [
                                    row.model_dump(mode="json") for row in rejected],
                                "verification_ref": f"{artifact}#/results/{index}"})
        for issue in document.issues:
            if (issue.get("stage") == "bom_ocr_conflict"
                    and issue.get("item_number") == winner.attributes["item_number"]
                    and set(issue.get("entity_ids", [])) <= {row.id for row in candidates}
                    and set(issue.get("fields", [])) <= {"part_number"}):
                issue["resolved"] = True
                issue["resolution_ref"] = f"{artifact}#/results/{index}"
        document.bom = [row for row in document.bom if row.id not in {r.id for r in rejected}]
        resolved += 1
    return resolved


def verify_bom_part_numbers(document: DrawingDocument, output: Path,
                            paddle_python: Path) -> int:
    jobs, eligible = conflict_jobs(document)
    policy = {"model": "PP-OCRv6_medium_rec", "device": "cpu",
              "minimum_score": MIN_SCORE, "requested": len(jobs), "resolved": 0,
              "eligible_for_training": False}
    document.configuration["bom_part_verification"] = policy
    if not jobs:
        return 0
    from .document_parser import write_json
    input_path = output / "bom_part_verification_jobs.json"
    result_path = output / "bom_part_verification.json"
    write_json(input_path, jobs)
    worker = Path(__file__).resolve().parents[3] / "scripts" / "verify_bom_parts_worker.py"
    with (output / "bom_part_verification.log").open("x") as log:
        try:
            run = subprocess.run([str(paddle_python), str(worker), str(input_path),
                                  str(result_path)], stdout=log, stderr=subprocess.STDOUT,
                                 timeout=300, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            document.issues.append({"stage": "bom_part_verification", "error": str(exc)})
            return 0
    if run.returncode or not result_path.is_file():
        document.issues.append({"stage": "bom_part_verification",
                                "error": f"Worker exit code {run.returncode}",
                                "log": "bom_part_verification.log"})
        return 0
    try:
        count = apply_verifications(document, jobs, eligible,
                                    json.loads(result_path.read_text(encoding="utf-8")))
        policy["resolved"] = count
        return count
    except (ValueError, KeyError, TypeError) as exc:
        document.issues.append({"stage": "bom_part_verification", "error": str(exc),
                                "raw": result_path.name})
        return 0
