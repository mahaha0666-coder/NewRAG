"""Extract localized OCR and candidate relations without a generative OCR server."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tools.drawing_extraction.fast_ocr import parse_fast_drawing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ocr-python", type=Path,
                        default=Path("/home/userroot/miniconda3/envs/paddleocr-gpu/bin/python"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--limit-side-len", type=int, default=2200)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="cpu")
    args = parser.parse_args()
    document = parse_fast_drawing(
        args.pdf, args.output, args.ocr_python, dpi=args.dpi,
        limit_side_len=args.limit_side_len, timeout=args.timeout,
        device=args.device,
    )
    print(f"{args.output / 'drawing.json'}: {document.extraction_status}; "
          f"{len(document.raw_ocr)} OCR texts; {len(document.bom)} BOM candidates; "
          f"{len(document.relationships)} candidate relations")
    if any(page.paddle_status == "failed" for page in document.pages):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
