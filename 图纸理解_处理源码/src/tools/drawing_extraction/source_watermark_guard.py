"""Keep source-PDF download watermark glyphs out of engineering candidates."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import pymupdf

from .document_schema import DrawingDocument
from .source_page_render import apply_document_page_render


_DOWNLOAD_WATERMARK = re.compile(r"浙江迦南.*下载时间[:：].*20\d{2}")


def _normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split()).upper()


def watermark_candidate_size(
    box: tuple[float, float, float, float], page_size: tuple[int, int],
) -> bool:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return area >= .005 * page_size[0] * page_size[1]


def source_watermark_glyphs(
    document: DrawingDocument,
) -> dict[int, list[tuple[str, float, float]]]:
    source = Path(document.source_path)
    if source.suffix.casefold() != ".pdf" or not source.is_file():
        return {}
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != document.source_sha256:
            raise ValueError("Source PDF hash mismatch in watermark guard")
    glyphs: dict[int, list[tuple[str, float, float]]] = {}
    with pymupdf.open(source) as pdf:
        for page_info in document.pages:
            page = pdf[page_info.page_number - 1]
            apply_document_page_render(page, page_info)
            transform = ~pymupdf.Matrix(*page_info.pixel_to_pdf_matrix)
            for block in page.get_text("rawdict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        color = int(span.get("color", 0))
                        channels = ((color >> 16) & 255, (color >> 8) & 255, color & 255)
                        text = "".join(char["c"] for char in span["chars"])
                        if (span.get("alpha", 255) > 180
                                or min(channels) < 160
                                or not _DOWNLOAD_WATERMARK.search(text)):
                            continue
                        for char in span["chars"]:
                            rect = pymupdf.Rect(char["bbox"])
                            center = pymupdf.Point(
                                (rect.x0 + rect.x1) / 2,
                                (rect.y0 + rect.y1) / 2,
                            )
                            pixel = center * transform
                            glyphs.setdefault(page_info.page_number, []).append(
                                (char["c"], pixel.x, pixel.y)
                            )
    return glyphs


def is_source_watermark_fragment(
    text: str, box: tuple[float, float, float, float],
    glyphs: list[tuple[str, float, float]],
) -> bool:
    normalized = _normalized(text)
    if len(normalized) < 4 or not glyphs:
        return False
    inside = _normalized("".join(
        char for char, x, y in glyphs
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]
    ))
    return normalized in inside


__all__ = ["is_source_watermark_fragment", "source_watermark_glyphs",
           "watermark_candidate_size"]
