"""Geometry and entity helpers for DWG extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    def expand(self, other: "BBox") -> "BBox":
        return BBox(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y),
        )

    def normalized_in(self, frame: "BBox") -> dict[str, float]:
        if frame.width == 0 or frame.height == 0:
            return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
        return {
            "x": (self.min_x - frame.min_x) / frame.width,
            "y": 1.0 - ((self.max_y - frame.min_y) / frame.height),
            "width": self.width / frame.width,
            "height": self.height / frame.height,
        }


def _bbox_from_points(points: list[tuple[float, float]]) -> BBox | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def _dimension_geometry_bbox(entity: Any) -> BBox | None:
    points: list[tuple[float, float]] = []
    for attr in ("defpoint", "defpoint2", "defpoint3", "defpoint4", "defpoint5"):
        if hasattr(entity.dxf, attr):
            point = getattr(entity.dxf, attr)
            if abs(point.x) > 1e-9 or abs(point.y) > 1e-9:
                points.append((point.x, point.y))
    return _bbox_from_points(points)


def _bbox_from_center(center_x: float, center_y: float, width: float, height: float) -> BBox:
    half_width = max(width, 1.0) / 2.0
    half_height = max(height, 1.0) / 2.0
    return BBox(center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height)


def _clean_mtext(text: str) -> str:
    value = text.replace("\\P", "\n")
    value = re.sub(r"\{\\H[^;]+;", "", value)
    value = value.replace("}", "")
    return re.sub(r"[ \t]+", " ", value).strip()


def _dimension_display_text(text: str, measurement: float) -> str:
    rounded = int(round(measurement)) if abs(measurement - round(measurement)) < 1e-6 else round(measurement, 3)
    value = text or "<>"
    value = value.replace("%%c", "Ø")
    return value.replace("<>", str(rounded))


def _dimension_text_bbox(entity: Any, display_text: str) -> BBox | None:
    """Approximate the rendered dimension text bbox instead of the full extension-line bbox."""

    text_entity = None
    try:
        for virtual_entity in entity.virtual_entities():
            if virtual_entity.dxftype() in {"TEXT", "MTEXT"}:
                text_entity = virtual_entity
                break
    except Exception:
        text_entity = None

    insert = None
    char_height = 18.0
    direction_x = 1.0
    direction_y = 0.0
    if text_entity is not None and hasattr(text_entity.dxf, "insert"):
        insert = text_entity.dxf.insert
        char_height = float(getattr(text_entity.dxf, "char_height", char_height) or char_height)
        direction = getattr(text_entity.dxf, "text_direction", None)
        if direction is not None:
            direction_x = float(direction.x)
            direction_y = float(direction.y)
    elif hasattr(entity.dxf, "text_midpoint"):
        insert = entity.dxf.text_midpoint

    if insert is None:
        return _dimension_geometry_bbox(entity)

    text_length = max(len(_clean_mtext(display_text)), 1)
    long_side = max(char_height * (0.72 * text_length + 1.2), char_height * 2.2)
    short_side = max(char_height * 1.45, 8.0)
    if abs(direction_y) > abs(direction_x):
        width, height = short_side, long_side
    else:
        width, height = long_side, short_side
    return _bbox_from_center(float(insert.x), float(insert.y), width, height)


def _entity_bbox(entity: Any) -> BBox | None:
    dxftype = entity.dxftype()
    if dxftype == "LEADER":
        return _bbox_from_points([(vertex[0], vertex[1]) for vertex in entity.vertices])
    if dxftype == "LINE":
        return _bbox_from_points([(entity.dxf.start.x, entity.dxf.start.y), (entity.dxf.end.x, entity.dxf.end.y)])
    if dxftype == "DIMENSION":
        raw_text = str(entity.dxf.text or "")
        return _dimension_text_bbox(entity, _dimension_display_text(raw_text, float(entity.get_measurement())))
    if dxftype == "HATCH":
        points = []
        for path in entity.paths:
            if hasattr(path, "vertices"):
                points.extend((float(vertex[0]), float(vertex[1])) for vertex in path.vertices)
            if hasattr(path, "edges"):
                for edge in path.edges:
                    for attr in ("start", "end", "center"):
                        if hasattr(edge, attr):
                            point = getattr(edge, attr)
                            points.append((float(point[0]), float(point[1])))
        return _bbox_from_points(points)
    if dxftype in {"TEXT", "MTEXT", "ATTRIB", "ATTDEF"} and hasattr(entity.dxf, "insert"):
        point = entity.dxf.insert
        return BBox(point.x, point.y, point.x, point.y)
    if dxftype in {"ARC", "CIRCLE", "LWPOLYLINE", "POLYLINE", "SPLINE"}:
        from ezdxf import bbox as ezdxf_bbox

        try:
            extents = ezdxf_bbox.extents([entity])
            if extents.has_data:
                return BBox(extents.extmin.x, extents.extmin.y,
                            extents.extmax.x, extents.extmax.y)
        except (ValueError, TypeError, AttributeError):
            return None
    return None


def _entity_details(entity: Any, unit_to_mm: float) -> dict[str, Any]:
    dxftype = entity.dxftype()
    details: dict[str, Any] = {}
    if dxftype == "LINE":
        start = entity.dxf.start
        end = entity.dxf.end
        length = ((end.x - start.x) ** 2 + (end.y - start.y) ** 2) ** 0.5
        details.update(
            {
                "start": [start.x, start.y, start.z],
                "end": [end.x, end.y, end.z],
                "length_source_units": length,
                "length_mm": length * unit_to_mm,
            }
        )
    elif dxftype in {"ARC", "CIRCLE"}:
        center = entity.dxf.center
        radius = float(entity.dxf.radius)
        details.update({
            "center": [center.x, center.y, center.z],
            "radius_source_units": radius,
            "radius_mm": radius * unit_to_mm,
        })
        if dxftype == "ARC":
            details.update({
                "start_angle_deg": float(entity.dxf.start_angle),
                "end_angle_deg": float(entity.dxf.end_angle),
            })
    elif dxftype == "LEADER":
        details.update({
            "vertices_cad": [[vertex[0], vertex[1], vertex[2]] for vertex in entity.vertices],
            "arrowhead_flag": entity.dxf.get("has_arrowhead"),
            "annotation_handle": entity.dxf.get("annotation_handle"),
        })
    elif dxftype == "DIMENSION":
        measurement = float(entity.get_measurement())
        raw_text = str(entity.dxf.text or "")
        details.update(
            {
                "dimtype": int(entity.dxf.dimtype),
                "raw_text": raw_text,
                "display_text": _dimension_display_text(raw_text, measurement),
                "measurement_source_units": measurement,
                "measurement_mm": measurement * unit_to_mm,
                "angle": float(getattr(entity.dxf, "angle", 0.0) or 0.0),
            }
        )
        for attr in ("defpoint", "defpoint2", "defpoint3", "defpoint4", "defpoint5"):
            if hasattr(entity.dxf, attr):
                point = getattr(entity.dxf, attr)
                details[attr] = [point.x, point.y, point.z]
    elif dxftype == "HATCH":
        details.update(
            {
                "pattern_name": entity.dxf.pattern_name,
                "solid_fill": int(entity.dxf.solid_fill),
                "path_count": len(entity.paths),
            }
        )
    elif dxftype in {"TEXT", "MTEXT", "ATTRIB", "ATTDEF"}:
        if dxftype == "TEXT":
            details["text"] = entity.dxf.text
        elif dxftype == "MTEXT":
            details["text"] = entity.text
        else:
            details["tag"] = getattr(entity.dxf, "tag", "")
            details["text"] = entity.dxf.text
    return details
