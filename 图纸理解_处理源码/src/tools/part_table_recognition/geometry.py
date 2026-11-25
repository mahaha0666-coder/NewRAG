"""Source PDF crops and ruled-table geometry in the displayed page frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import cv2
import numpy as np
import pymupdf

from src.tools.drawing_extraction.source_page_render import plan_source_page_render


@dataclass
class SourceCrop:
    image: np.ndarray
    rotation: int
    raster_size: tuple[int, int]
    raster_origin: tuple[int, int]
    dpi: int
    pixel_to_pdf: list[float]
    pdf_derotation: list[float]
    page_size: Sequence[int]
    requested_box: list[float]
    pdf_clip: list[float]

    def page_box(self, box: Sequence[float]) -> list[float]:
        width, height = self.raster_size
        points = []
        for x, y in ((box[0], box[1]), (box[2], box[1]),
                     (box[2], box[3]), (box[0], box[3])):
            if self.rotation == 90:
                x, y = width - y, x
            elif self.rotation == 180:
                x, y = width - x, height - y
            elif self.rotation == 270:
                x, y = y, height - x
            point = pymupdf.Point(x + self.raster_origin[0], y + self.raster_origin[1])
            point *= pymupdf.Matrix(72 / self.dpi, 72 / self.dpi)
            point *= pymupdf.Matrix(*self.pdf_derotation)
            point *= ~pymupdf.Matrix(*self.pixel_to_pdf)
            points.append(point)
        return [max(0.0, min(point.x for point in points)),
                max(0.0, min(point.y for point in points)),
                min(float(self.page_size[0]), max(point.x for point in points)),
                min(float(self.page_size[1]), max(point.y for point in points))]


def render_source_crop(
    source: Path, *, page: int, box: Sequence[float], page_size: Sequence[int],
    pixel_to_pdf: Sequence[float], dpi: int = 400, rotation: int = 0,
) -> SourceCrop:
    if rotation not in (0, 90, 180, 270) or not 200 <= dpi <= 800:
        raise ValueError("Invalid source crop settings")
    if len(box) != 4 or not (0 <= box[0] < box[2] <= page_size[0]
                            and 0 <= box[1] < box[3] <= page_size[1]):
        raise ValueError("Crop lies outside the displayed source page")
    with pymupdf.open(source) as document:
        if type(page) is not int or not 1 <= page <= len(document):
            raise ValueError("Source page does not exist")
        native = document[page - 1]
        plan = plan_source_page_render(native)
        if plan.render_mediabox != plan.source_mediabox:
            native.set_mediabox(pymupdf.Rect(plan.render_mediabox))
        pdf_clip = pymupdf.Rect(box) * pymupdf.Matrix(*pixel_to_pdf)
        clip = pdf_clip * native.rotation_matrix
        if not native.rect.contains(clip):
            visible = clip & native.rect
            tolerance = 72 / 200 + 1e-4
            if visible.is_empty or max(abs(a - b) for a, b in zip(clip, visible)) > tolerance:
                raise ValueError("Requested crop is outside the original source frame")
            clip &= native.rect
        pixmap = native.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72),
                                  clip=clip, alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if rotation:
            image = np.rot90(image, rotation // 90).copy()
        return SourceCrop(image=image, rotation=rotation, raster_size=(pixmap.width, pixmap.height),
                          raster_origin=(pixmap.x, pixmap.y), dpi=dpi,
                          pixel_to_pdf=list(pixel_to_pdf), pdf_derotation=list(native.derotation_matrix),
                          page_size=page_size, requested_box=list(box), pdf_clip=list(pdf_clip))


def _centers(mask: np.ndarray, axis: int, minimum: float) -> list[int]:
    positions = np.flatnonzero(np.count_nonzero(mask, axis=axis) >= minimum)
    groups: list[list[int]] = []
    for position in positions:
        if not groups or position - groups[-1][-1] > 3:
            groups.append([int(position)])
        else:
            groups[-1].append(int(position))
    return [round(sum(group) / len(group)) for group in groups]


def ruled_grid(image: np.ndarray) -> tuple[list[int], list[int]]:
    """Find ruling axes; never divide a table uniformly from OCR row counts."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground = np.uint8(gray < 245) * 255
    height, width = gray.shape
    horizontal = cv2.morphologyEx(foreground, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 8), 1)))
    vertical = cv2.morphologyEx(foreground, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 8))))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8))
    xs = _centers(vertical, 0, height * .55)
    ys = _centers(horizontal, 1, width * .65)
    return xs, ys


def cell_has_ink(image: np.ndarray, box: Sequence[int]) -> bool:
    x0, y0, x1, y1 = map(int, box)
    interior = image[y0 + 1:y1 - 1, x0 + 1:x1 - 1]
    if not interior.size or min(interior.shape[:2]) < 3:
        return True
    foreground = np.uint8(cv2.cvtColor(interior, cv2.COLOR_BGR2GRAY) < 245)
    height, width = foreground.shape
    horizontal = cv2.morphologyEx(foreground, cv2.MORPH_OPEN,
                                 np.ones((1, max(2, round(width * .9))), np.uint8))
    vertical = cv2.morphologyEx(foreground, cv2.MORPH_OPEN,
                               np.ones((max(2, round(height * .9)), 1), np.uint8))
    foreground[(horizontal | vertical) != 0] = 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    return any(stats[index, cv2.CC_STAT_AREA] >= 3 for index in range(1, count))
