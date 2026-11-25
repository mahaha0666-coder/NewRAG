"""Read-only audit of native PDF evidence before raster OCR.

The audit deliberately does not promote PDF objects to drawing relationships. It
records what is physically present in the file so later OCR/VLM experiments can
be compared without losing provenance.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pymupdf


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def _bbox(values: Any) -> list[float] | None:
    if values is None:
        return None
    try:
        rect = values if len(values) == 4 else values[:4]
        return [round(float(value), 4) for value in rect]
    except (TypeError, ValueError):
        return None


def _page_modality(text_chars: int, span_count: int, drawing_count: int,
                   image_count: int) -> str:
    if text_chars == 0 and image_count:
        return "scanned"
    if text_chars and image_count:
        return "hybrid"
    if text_chars or drawing_count:
        return "native_vector"
    return "unknown"


def audit_pdf(path: Path) -> dict[str, Any]:
    """Return a bounded, JSON-serializable evidence inventory for one PDF."""
    source_sha = sha256(path)
    pages: list[dict[str, Any]] = []
    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text_dict = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT)
            spans: list[dict[str, Any]] = []
            text_chars = 0
            for block in text_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = str(span.get("text") or "")
                        if not text.strip():
                            continue
                        text_chars += len(text)
                        spans.append({
                            "text": text,
                            "bbox": _bbox(span.get("bbox")),
                            "font": span.get("font"),
                            "size": round(float(span.get("size", 0.0)), 4),
                            "flags": int(span.get("flags", 0)),
                        })
            drawings = page.get_drawings()
            drawing_records = []
            for index, drawing in enumerate(drawings):
                drawing_records.append({
                    "id": f"page_{page_number:04d}_drawing_{index:05d}",
                    "bbox": _bbox(drawing.get("rect")),
                    "type": drawing.get("type"),
                    "item_count": len(drawing.get("items") or []),
                    "color": drawing.get("color"),
                    "fill": drawing.get("fill"),
                })
            images = page.get_images(full=True)
            modality = _page_modality(text_chars, len(spans), len(drawings), len(images))
            pages.append({
                "page": page_number,
                "width": round(float(page.rect.width), 4),
                "height": round(float(page.rect.height), 4),
                "modality": modality,
                "text_chars": text_chars,
                "text_span_count": len(spans),
                "drawing_count": len(drawings),
                "image_count": len(images),
                "text_spans": spans,
                "native_drawings": drawing_records,
                "source_refs": [
                    f"{path.name}#page={page_number}#text_spans",
                    f"{path.name}#page={page_number}#drawings",
                ],
            })
    counts = {name: sum(page["modality"] == name for page in pages)
              for name in ("native_vector", "scanned", "hybrid", "unknown")}
    return {
        "schema_version": "pdf_modality_audit.v1",
        "source_path": str(path.resolve()),
        "source_sha256": source_sha,
        "page_count": len(pages),
        "modality_counts": counts,
        "pages": pages,
    }


__all__ = ["audit_pdf"]
