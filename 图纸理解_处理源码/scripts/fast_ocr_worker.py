"""Run bounded, non-generative OCR on rendered engineering drawing pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from src.tools.drawing_extraction.ocr_geometry import map_region_polygon
from src.tools.drawing_extraction.adaptive_regions import full_page_tile_plan


def localized_observation(text: str, score: float, polygon: list, job: dict) -> dict:
    """Both full-page tiles and targeted crops use the same page transform."""
    return {
        "text": text, "score": float(score),
        "polygon": map_region_polygon(polygon, job["frame"], job["rotation"], job["scale"]),
        "crop_polygon": polygon,
        "pass_id": job.get("pass_id", "fast_full_page"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit-side-len", type=int, default=2200)
    parser.add_argument("--regions-file", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu:0"))
    args = parser.parse_args()
    if not 1200 <= args.limit_side_len <= 4000:
        parser.error("--limit-side-len must be between 1200 and 4000")
    if args.output.exists():
        raise FileExistsError(args.output)

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    from paddleocr import PaddleOCR
    from PIL import Image
    import numpy as np

    with args.image.open("rb") as stream:
        image_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with Image.open(args.image) as source:
        width, height = source.size

    started = time.monotonic()
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv6_small_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=args.limit_side_len,
        enable_mkldnn=False,
        device=args.device,
    )
    regions = None
    if args.regions_file:
        regions = json.loads(args.regions_file.read_text(encoding="utf-8"))
        if (regions.get("image_sha256") != image_hash
                or regions.get("image_size") != [width, height]
                or not 1 <= len(regions.get("regions", [])) <= 64):
            raise ValueError("Region plan does not match the rendered page")
    jobs = regions["regions"] if regions else full_page_tile_plan(
        width, height, tile_side=args.limit_side_len)
    observations = []
    with Image.open(args.image) as source:
        for job in jobs:
            frame = job["frame"]
            rotation, scale = job["rotation"], job["scale"]
            if (len(frame) != 4 or not 0 <= frame[0] < frame[2] <= width
                    or not 0 <= frame[1] < frame[3] <= height
                    or rotation not in (0, 90) or scale not in (1, 1.5)):
                raise ValueError("Invalid OCR crop, rotation or zoom")
            crop = source.crop(frame)
            if rotation == 90:
                crop = crop.transpose(Image.Transpose.ROTATE_90)
            if scale != 1:
                crop = crop.resize((round(crop.width * scale), round(crop.height * scale)),
                                   Image.Resampling.LANCZOS)
            pages = list(ocr.predict(input=np.asarray(crop.convert("RGB"))))
            if len(pages) != 1:
                raise ValueError("Expected exactly one OCR result for each region")
            result = pages[0].json["res"]
            texts, scores, polygons = (
                result["rec_texts"], result["rec_scores"], result["rec_polys"]
            )
            if not len(texts) == len(scores) == len(polygons):
                raise ValueError("OCR text, score and polygon counts do not agree")
            for text, score, polygon in zip(texts, scores, polygons, strict=True):
                original = polygon.tolist() if hasattr(polygon, "tolist") else polygon
                observations.append(localized_observation(text, score, original, job))
    payload = {
        "schema_version": "fast_ocr_page.v2",
        "coordinate_space": "rendered_page_pixels",
        "geometry_version": "all_passes_origin_rotation_scale_v2",
        "model": "PP-OCRv6_small_det+PP-OCRv6_medium_rec",
        "image_sha256": image_hash,
        "image_size": [width, height],
        "limit_side_len": args.limit_side_len,
        "coverage_mode": "explicit_regions" if regions else "full_page_tiles_v1",
        "tile_count": len(jobs),
        "tile_frames": [job["frame"] for job in jobs],
        "source_passes": [{**job, "response_status": "complete",
                           "observation_count": sum(row["pass_id"] == job["pass_id"] for row in observations)}
                          for job in jobs],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "observations": observations,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=True, indent=2, allow_nan=False)
    print(f"{args.output}: {len(observations)} localized texts in {payload['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
