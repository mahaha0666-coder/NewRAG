"""Shared native-geometry helpers for dimension relation candidates."""

from __future__ import annotations

import math
from typing import Any


def source_keys(line: dict[str, Any]) -> set[tuple[int, int, str]]:
    """Return stable native source keys for a dimension line fragment."""

    return {
        (row["path_index"], row["item_index"], row["native_kind"])
        for row in line.get("sources") or []
    }


def stroke_source_keys(stroke: dict[str, Any]) -> set[tuple[int, int, str]]:
    """Return stable native source keys for an extracted stroke."""

    return {
        (row["path_index"], row["item_index"], row["native_kind"])
        for row in stroke.get("sources") or []
    }


def parallel_to_extension(stroke: dict[str, Any], extension: dict[str, Any]) -> bool:
    """Return whether a two-point stroke is parallel to an extension line."""

    points = stroke.get("points") or []
    if len(points) != 2:
        return False
    extension_vector = (
        extension["second"][0] - extension["first"][0],
        extension["second"][1] - extension["first"][1],
    )
    stroke_vector = (points[1][0] - points[0][0], points[1][1] - points[0][1])
    extension_length = math.hypot(*extension_vector)
    stroke_length = math.hypot(*stroke_vector)
    if min(extension_length, stroke_length) <= 1e-6:
        return True
    cosine = abs(
        extension_vector[0] * stroke_vector[0] + extension_vector[1] * stroke_vector[1]
    ) / (extension_length * stroke_length)
    return cosine >= math.cos(math.radians(5.0))


__all__ = ["parallel_to_extension", "source_keys", "stroke_source_keys"]
