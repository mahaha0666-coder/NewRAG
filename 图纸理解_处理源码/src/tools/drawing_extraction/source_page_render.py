"""Source-only PDF render policy for narrowly verified right-edge clipping.

All geometry remains in the original PDF's point coordinates. The source file
is never rewritten; PyMuPDF changes only a page in an in-memory document.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pymupdf


ORIGINAL_RENDER = "original_pdf_visible_v1"
RIGHT_EDGE_RECOVERY = "original_pdf_right_edge_recovery_v1"


@dataclass(frozen=True)
class SourcePageRenderPlan:
    policy: str
    source_mediabox: tuple[float, float, float, float]
    source_cropbox: tuple[float, float, float, float]
    render_mediabox: tuple[float, float, float, float]
    native_path_count: int
    outside_path_counts: dict[str, int]


def plan_source_page_render(page: pymupdf.Page) -> SourcePageRenderPlan:
    """Select a review-only expansion only for the established right-edge pattern."""
    media = tuple(float(value) for value in page.mediabox)
    crop = tuple(float(value) for value in page.cropbox)
    frame = page.rect * page.derotation_matrix
    paths = [row["rect"] for row in page.get_drawings()]
    counts = {
        "left": sum(row.x0 < frame.x0 - 3 for row in paths),
        "right": sum(row.x1 > frame.x1 + 3 for row in paths),
        "top": sum(row.y0 < frame.y0 - 3 for row in paths),
        "bottom": sum(row.y1 > frame.y1 + 3 for row in paths),
    }
    max_x = max((row.x1 for row in paths), default=frame.x1)
    eligible = (
        page.rotation == 0
        and media == crop
        and frame.x0 == frame.y0 == 0
        and frame.width < frame.height
        and len(paths) >= 100
        and counts["right"] >= 100
        and counts["right"] / len(paths) >= .2
        and counts["left"] + counts["top"] + counts["bottom"]
        <= max(5, int(len(paths) * .01))
        and math.isfinite(max_x)
        and frame.x1 * 1.15 <= max_x <= frame.height * 1.05
    )
    if eligible:
        margin = max(20.0, frame.width * .03)
        render = (frame.x0, frame.y0, float(math.ceil(max_x + margin)), frame.y1)
    else:
        render = media
    return SourcePageRenderPlan(
        policy=RIGHT_EDGE_RECOVERY if eligible else ORIGINAL_RENDER,
        source_mediabox=media, source_cropbox=crop, render_mediabox=render,
        native_path_count=len(paths), outside_path_counts=counts,
    )


def _field(info: Any, name: str, default: Any = None) -> Any:
    return info.get(name, default) if isinstance(info, Mapping) else getattr(info, name, default)


def apply_document_page_render(page: pymupdf.Page, page_info: Any) -> SourcePageRenderPlan:
    """Reopen a recorded render frame from the same original PDF geometry.

    For legacy pages without render metadata the original visible page is kept.
    A recovered page must exactly match the fresh source-only policy decision.
    """
    plan = plan_source_page_render(page)
    policy = _field(page_info, "render_policy", ORIGINAL_RENDER)
    source_media = _field(page_info, "source_mediabox")
    source_crop = _field(page_info, "pdf_cropbox")
    render_media = _field(page_info, "render_mediabox")
    if source_media is not None and tuple(source_media) != plan.source_mediabox:
        raise ValueError("Original PDF MediaBox differs from render record")
    if source_crop is not None and tuple(source_crop) != plan.source_cropbox:
        raise ValueError("Original PDF CropBox differs from render record")
    if policy == RIGHT_EDGE_RECOVERY:
        if plan.policy != policy or render_media is None \
                or tuple(render_media) != plan.render_mediabox:
            raise ValueError("Recovered render frame differs from original PDF")
        page.set_mediabox(pymupdf.Rect(*plan.render_mediabox))
    elif policy != ORIGINAL_RENDER:
        raise ValueError("Unknown source PDF render policy")
    elif render_media is not None and tuple(render_media) != plan.source_mediabox:
        raise ValueError("Original render frame differs from source PDF")
    return plan


def source_render_evidence(
    source: Path, source_sha256: str, page_info: Any, *, dpi: int = 200,
) -> dict[str, Any]:
    """Reconstruct an artifact image and coordinate frame from original bytes."""
    if not 72 <= dpi <= 400:
        raise ValueError("DPI must be between 72 and 400")
    with source.open("rb") as stream:
        source_bytes = stream.read()
    if hashlib.sha256(source_bytes).hexdigest() != source_sha256:
        raise ValueError("Original PDF SHA-256 differs from render record")
    number = _field(page_info, "page_number")
    with pymupdf.open(stream=source_bytes, filetype="pdf") as document:
        if document.needs_pass or not isinstance(number, int) or not 1 <= number <= len(document):
            raise ValueError("Original PDF page is unavailable")
        page = document[number - 1]
        plan = apply_document_page_render(page, page_info)
        matrix = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        evidence = {
            "source_sha256": source_sha256,
            "image_sha256": hashlib.sha256(pix.tobytes(output="png")).hexdigest(),
            "size": [pix.width, pix.height],
            "pixel_to_pdf_matrix": list((~matrix) * page.derotation_matrix),
            "render_policy": _field(page_info, "render_policy", ORIGINAL_RENDER),
            "source_mediabox": list(plan.source_mediabox),
            "source_cropbox": list(plan.source_cropbox),
            "render_mediabox": list(page.mediabox),
        }
    if source.read_bytes() != source_bytes:
        raise ValueError("Original PDF changed during render reconstruction")
    return evidence


def verify_source_page_image(
    source: Path, source_sha256: str, page_info: Any, *, dpi: int = 200,
) -> bool:
    """Check image bytes and transform before a recovered page is served."""
    evidence = source_render_evidence(source, source_sha256, page_info, dpi=dpi)
    return (
        evidence["image_sha256"] == _field(page_info, "image_sha256")
        and evidence["size"] == [_field(page_info, "width"), _field(page_info, "height")]
        and all(abs(left - right) <= 1e-5 for left, right in zip(
            evidence["pixel_to_pdf_matrix"], _field(page_info, "pixel_to_pdf_matrix", []),
        ))
        and len(_field(page_info, "pixel_to_pdf_matrix", [])) == 6
    )
