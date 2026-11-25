"""Render original PDF evidence with invertible, explicit coordinate transforms."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import fitz

from src.tools.research.artifacts import digest


def checked_box(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Expected four normalized box coordinates")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in value
    ):
        raise ValueError("Box coordinates must be finite numbers")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        raise ValueError("Normalized box lies outside its image or has zero area")
    return [float(v) for v in value]


def pdf_box(box: list[float], image: dict[str, Any]) -> list[float]:
    """Normalized image box -> unrotated PDF cropbox-local points."""
    x0, y0, x1, y1 = checked_box(box)
    w, h = image["width"], image["height"]
    matrix = fitz.Matrix(image["image_to_pdf"])
    rect = (
        fitz.Rect(x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000) * matrix
    )
    return list(rect & fitz.Rect(image["page_bbox_pdf"]))


def render_source(
    page: fitz.Page,
    path: Path,
    *,
    image_id: str,
    dpi: int,
    clip: list[float] | None = None,
    origin: str = "model_proposal",
) -> dict[str, Any]:
    if not 72 <= dpi <= 400:
        raise ValueError("DPI must be between 72 and 400")
    if path.exists():
        raise FileExistsError(path)
    frame = fitz.Rect(0, 0, page.cropbox.width, page.cropbox.height)
    rect = (fitz.Rect(clip) & frame) if clip is not None else frame
    if rect.is_empty or rect.is_infinite:
        raise ValueError("Empty source crop")
    scale = dpi / 72
    # PyMuPDF raster clips are in the displayed page frame. Coordinates saved
    # for evidence remain in the original unrotated PDF frame. Preserve the
    # page's intended reading orientation instead of rotating its text sideways.
    displayed_clip = rect * page.rotation_matrix
    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale), clip=displayed_clip, alpha=False
    )
    pix.save(path)
    transform = (
        page.rotation_matrix
        * fitz.Matrix(scale, scale)
        * fitz.Matrix(1, 0, 0, 1, -pix.x, -pix.y)
    )
    inverse = ~transform
    return {
        "image_id": image_id,
        "path": str(path.resolve()),
        "sha256": digest(path),
        "width": pix.width,
        "height": pix.height,
        "dpi": dpi,
        "page_number": page.number + 1,
        "source_page_rotation": page.rotation,
        "coordinate_system": "unrotated_pdf_cropbox_local_points",
        "page_bbox_pdf": list(frame),
        "crop_bbox_pdf": list(rect),
        "pdf_to_image": list(transform),
        "image_to_pdf": list(inverse),
        "crop_corners_pdf": [
            [rect.x0, rect.y0],
            [rect.x1, rect.y0],
            [rect.x1, rect.y1],
            [rect.x0, rect.y1],
        ],
        "region_origin": origin,
    }
