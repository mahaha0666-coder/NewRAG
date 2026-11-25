"""DWG drawing extraction tools.

The DWG extraction environment does not need OpenCV. Alignment helpers are
loaded lazily so `extract_dwg` remains usable in the lightweight DWG tool env.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Alignment",
    "choose_alignment",
    "choose_alignment_for_path",
    "extract_dwg",
    "parse_drawing",
    "DrawingDocument",
    "region_display_metadata",
    "region_display_text",
    "region_label",
    "region_to_box",
    "regions_for_mapping",
    "regions_to_boxes",
]

_ALIGNMENT_EXPORTS = {
    "Alignment",
    "choose_alignment",
    "choose_alignment_for_path",
    "region_display_metadata",
    "region_display_text",
    "region_label",
    "region_to_box",
    "regions_for_mapping",
    "regions_to_boxes",
}


def __getattr__(name: str) -> Any:
    if name == "parse_drawing":
        return import_module(".document_parser", __name__).parse_drawing
    if name == "DrawingDocument":
        return import_module(".document_schema", __name__).DrawingDocument
    if name == "extract_dwg":
        return import_module(".workflow", __name__).extract_dwg
    if name in _ALIGNMENT_EXPORTS:
        return getattr(import_module(".alignment", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
