"""Run bounded semantic ROI spotting with the pinned PaddleOCR-VL pipeline."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.paddle_document_worker import (  # noqa: E402
    collect_model_responses,
    completion_summary,
    run_targeted_with_refinement,
)
from src.tools.research.artifacts import digest  # noqa: E402


def run(manifest_path: Path, endpoint: str, max_new_tokens: int) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("route_label_access") is not False:
        raise ValueError("ROI manifest label firewall is missing")
    if manifest.get("model") != "PaddleOCR-VL-1.6":
        raise ValueError("Unexpected ROI model")
    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        device="cpu",
        vl_rec_backend="vllm-server",
        vl_rec_server_url=endpoint,
        vl_rec_api_model_name="PaddleOCR-VL-1.6",
        vl_rec_max_concurrency=1,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_ocr_for_image_block=True,
        use_queues=False,
    )
    if version("paddlex") != "3.7.2":
        raise RuntimeError("Completion recorder requires tested PaddleX 3.7.2")
    recognizer = pipeline.paddlex_pipeline.vl_rec_model
    if not callable(getattr(recognizer, "_doc_vlm_genai_collect_responses", None)):
        raise RuntimeError("Official completion collector API has changed")
    for job in manifest.get("jobs") or []:
        image_path = Path(job["image"])
        target = Path(job["output"])
        if target.exists():
            raise FileExistsError(target)
        if digest(image_path) != job["image_sha256"]:
            raise ValueError(f"ROI image changed: {image_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "semantic_roi_response_r02m.v1",
            "status": "running",
            "item_code": job["item_code"],
            "group_id": job["group_id"],
            "source_sha256": job["source_sha256"],
            "page": job["page"],
            "image_sha256": job["image_sha256"],
            "page_size": job["page_size"],
            "region_plan": job["regions"],
            "results": [],
            "frames": [],
            "pass_ids": [],
            "attempts": [],
            "errors": [],
            "warnings": [],
            "route_label_access": False,
        }
        completions: list[dict] = []
        recognizer._doc_vlm_genai_collect_responses = (
            lambda futures: collect_model_responses(futures, target, completions)
        )
        region_results = []
        with Image.open(image_path) as opened:
            full = opened.convert("RGB")
            if list(full.size) != job["page_size"]:
                raise ValueError(f"ROI page size changed: {image_path}")
            for index, region in enumerate(job["regions"], start=1):
                pass_id = f"semantic_{index:02d}_{region['region_kind']}"
                result = run_targeted_with_refinement(
                    pipeline=pipeline,
                    full=full,
                    page_image=image_path,
                    target=target,
                    payload=payload,
                    completions=completions,
                    frame=region["frame"],
                    pass_id=pass_id,
                    max_new_tokens=max_new_tokens,
                )
                result.update({
                    "region_kind": region["region_kind"],
                    "basis": region["basis"],
                    "evidence_ids": region["evidence_ids"],
                })
                region_results.append(result)
        payload["region_results"] = region_results
        payload["completions"] = completions
        payload["completion_summary"] = completion_summary(completions)
        payload["status"] = (
            "ok" if region_results and all(row["complete"] for row in region_results)
            else "partial" if payload["results"] else "failed"
        )
        target.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(target, payload["status"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18022/v1")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()
    if not 1024 <= args.max_new_tokens <= 16384:
        parser.error("--max-new-tokens must be between 1024 and 16384")
    run(args.manifest.resolve(), args.endpoint, args.max_new_tokens)


if __name__ == "__main__":
    main()
