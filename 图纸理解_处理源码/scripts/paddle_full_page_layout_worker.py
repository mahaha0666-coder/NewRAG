"""Run the pinned PaddleOCR-VL layout detector over one rendered page."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--frame", nargs=4, type=int, required=True)
    args = parser.parse_args()

    if _digest(args.image) != args.image_sha256:
        raise ValueError("Rendered page hash changed before layout detection")
    with Image.open(args.image) as opened:
        size = [opened.width, opened.height]
        frame = args.frame
        if not (0 <= frame[0] < frame[2] <= size[0] and
                0 <= frame[1] < frame[3] <= size[1]):
            raise ValueError("Layout frame outside rendered page")
        with TemporaryDirectory(prefix="drawing-layout-") as directory:
            frame_image = Path(directory) / "frame.png"
            opened.crop(tuple(frame)).save(frame_image)

            from paddleocr import LayoutDetection
            import paddlex
            import yaml

            config_path = Path(paddlex.__file__).parent / "configs/pipelines/PaddleOCR-VL-1.6.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["SubModules"]["LayoutDetection"]
            model = LayoutDetection(
                model_name=config["model_name"], device="cpu", threshold=config["threshold"],
                layout_nms=config["layout_nms"], layout_unclip_ratio=config["layout_unclip_ratio"],
                layout_merge_bboxes_mode=config["layout_merge_bboxes_mode"],
            )
            started = time.monotonic()
            try:
                results = [result.json for result in model.predict(str(frame_image))]
            finally:
                model.close()
    if not results:
        raise RuntimeError("Layout detector returned no page result")
    payload = {
        "schema_version": "full_page_layout.v1", "status": "ok", "page": 1,
        "source_sha256": args.source_sha256, "image_sha256": args.image_sha256,
        "image_size": size, "frame": frame, "model": config["model_name"],
        "paddlex_version": version("paddlex"), "config": config,
        "seconds": round(time.monotonic() - started, 3), "results": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
