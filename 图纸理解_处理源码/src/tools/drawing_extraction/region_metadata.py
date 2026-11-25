"""Human-readable metadata for DWG extraction regions."""

from __future__ import annotations

from typing import Any


def region_label(region: dict[str, Any]) -> str:
    if region.get("kind") == "annotation":
        return "dimension_cluster"
    if region.get("role") == "main_view" or region.get("id") == "main_view_geometry":
        return "main_view"
    return "detail_view"


def region_display_metadata(region: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return human-readable DWG metadata for a region."""

    label = region_label(region)
    region_id = str(region.get("id", label))
    bbox_cad = region.get("bbox_cad") if isinstance(region.get("bbox_cad"), dict) else None
    bbox_size = _bbox_size(bbox_cad)
    description = str(region.get("description", ""))
    metadata: dict[str, Any] = {
        "display_label": "局部区域",
        "display_title": region_id,
        "display_value": "",
        "description": description,
        "source_region": region_id,
        "source_kind": region.get("kind", ""),
        "cad_bbox": bbox_cad or {},
        "cad_bbox_size": bbox_size or {},
        "cad_bbox_normalized": region.get("bbox_normalized_to_cad_extents", {}),
    }

    dimension = _dimension_for_region(region, payload)
    if dimension is not None:
        display_text = str(dimension.get("display_text") or description.removeprefix("Dimension annotation ").strip() or region_id)
        measurement_mm = _round_optional(dimension.get("measurement_mm"), 3)
        value = f"{_format_number(measurement_mm, 1)} mm" if measurement_mm is not None else ""
        metadata.update(
            {
                "display_label": f"尺寸 {display_text}",
                "display_title": f"尺寸标注 {display_text}",
                "display_value": value,
                "dimension_id": dimension.get("id", ""),
                "dimension_text": display_text,
                "raw_text": dimension.get("raw_text", ""),
                "dimtype": dimension.get("dimtype"),
                "layer": dimension.get("layer", ""),
                "measurement_source_units": _round_optional(dimension.get("measurement_source_units"), 3),
                "measurement_mm": measurement_mm,
                "source_unit": dimension.get("source_unit", ""),
            }
        )
        return metadata

    if region.get("role") == "main_view" or region_id == "main_view_geometry":
        entity_summary = _entity_summary(region.get("entity_types"))
        cad_value = ""
        if bbox_size:
            cad_value = f"CAD {_format_number(bbox_size['width'], 1)} x {_format_number(bbox_size['height'], 1)} mm"
        display_value = " / ".join(part for part in (entity_summary, cad_value) if part)
        metadata.update(
            {
                "display_label": "主视图 几何区域",
                "display_title": "主视图几何",
                "display_value": display_value,
                "entity_types": region.get("entity_types", {}),
            }
        )
        return metadata

    if bbox_size:
        metadata["display_value"] = f"CAD {_format_number(bbox_size['width'], 1)} x {_format_number(bbox_size['height'], 1)} mm"
    metadata["display_label"] = "子视图/局部区域"
    metadata["display_title"] = region_id
    return metadata


def region_display_text(region: dict[str, Any], payload: dict[str, Any] | None = None) -> str:
    metadata = region_display_metadata(region, payload)
    value = metadata.get("display_value")
    return f"{metadata['display_label']} / {value}" if value else str(metadata["display_label"])


def _round_optional(value: Any, digits: int = 3) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits)


def _format_number(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    text = f"{number:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def _bbox_size(bbox: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(bbox, dict):
        return None
    min_x = _round_optional(bbox.get("min_x"))
    min_y = _round_optional(bbox.get("min_y"))
    max_x = _round_optional(bbox.get("max_x"))
    max_y = _round_optional(bbox.get("max_y"))
    if min_x is None or min_y is None or max_x is None or max_y is None:
        return None
    return {
        "width": round(max_x - min_x, 3),
        "height": round(max_y - min_y, 3),
    }


def _dimension_for_region(region: dict[str, Any], payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    region_id = str(region.get("id", ""))
    if not region_id.endswith("_region"):
        return None
    dimension_id = region_id.removesuffix("_region")
    for dimension in payload.get("dimensions", []):
        if isinstance(dimension, dict) and dimension.get("id") == dimension_id:
            return dimension
    return None


def _entity_summary(entity_types: dict[str, Any] | None) -> str:
    if not isinstance(entity_types, dict) or not entity_types:
        return ""
    parts = []
    for key, value in sorted(entity_types.items()):
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        parts.append(f"{key} {count}")
    return " / ".join(parts)


__all__ = [
    "region_display_metadata",
    "region_display_text",
    "region_label",
]
