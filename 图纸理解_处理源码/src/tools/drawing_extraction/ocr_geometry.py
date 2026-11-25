"""Coordinate transforms shared by the isolated OCR worker and evidence graph."""

from __future__ import annotations


def map_region_polygon(
    polygon: list[list[float]], frame: list[int], rotation: int, scale: float,
) -> list[list[float]]:
    """Return resized/rotated OCR polygons to original rendered-page pixels."""
    x0, y0, x1, _ = frame
    if rotation == 0:
        return [[x0 + x / scale, y0 + y / scale] for x, y in polygon]
    if rotation == 90:
        return [[x1 - y / scale, y0 + x / scale] for x, y in polygon]
    raise ValueError("Unsupported BOM crop rotation")
