"""Bounded native elliptic HATCH edges, with no invented straight evidence.

LibreDWG fields map to DXF HATCH groups 10/11/40/50/51/73. The major-axis
endpoint is a vector relative to the center. Clockwise angles use the native
complementary-angle convention, as in ezdxf's EllipseEdge loader.
"""

from __future__ import annotations

import math

__all__ = ["ellipse_endpoints", "flatten_ellipse_edge"]


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("invalid_native_ellipse_field")
    return float(value)


def _frame(segment: dict) -> tuple[list[float], list[float], float, float, float]:
    if segment.get("curve_type") != 3:
        raise ValueError("native_ellipse_edge_required")
    center, major = segment.get("center"), segment.get("endpoint")
    if (not isinstance(center, list) or len(center) != 2
            or not isinstance(major, list) or len(major) != 2):
        raise ValueError("invalid_native_ellipse_frame")
    center, major = [_number(v) for v in center], [_number(v) for v in major]
    radius = math.hypot(*major)
    ratio = _number(segment.get("minor_major_ratio"))
    a, b = _number(segment.get("start_angle")), _number(segment.get("end_angle"))
    ccw = segment.get("is_ccw")
    if (isinstance(ccw, bool) or ccw not in (0, 1) or not math.isfinite(radius)
            or radius <= 1e-7 or not 0 < ratio <= 1 or not 0 < abs(b - a) <= math.tau + 1e-12):
        raise ValueError("unsupported_native_ellipse_form")
    sign = 1 if ccw else -1

    def parameter(angle: float) -> float:
        return math.atan2(math.sin(angle) / ratio, math.cos(angle))

    start = parameter(sign * a)
    if math.isclose(abs(b - a), math.tau, rel_tol=0, abs_tol=1e-12):
        sweep = sign * math.tau
    else:
        sweep = sign * ((sign * (parameter(sign * b) - start)) % math.tau)
        if not sweep:
            raise ValueError("degenerate_native_ellipse_span")
    return center, major, ratio, start, sweep


def _at(center: list, major: list, ratio: float, parameter: float) -> list[float]:
    cosine, sine = math.cos(parameter), math.sin(parameter)
    point = [center[0] + major[0] * cosine - major[1] * ratio * sine,
             center[1] + major[1] * cosine + major[0] * ratio * sine]
    if any(not math.isfinite(v) for v in point):
        raise ValueError("nonfinite_native_ellipse_point")
    return point


def ellipse_endpoints(segment: dict) -> tuple[list[float], list[float]]:
    center, major, ratio, start, sweep = _frame(segment)
    return _at(center, major, ratio, start), _at(center, major, ratio, start + sweep)


def flatten_ellipse_edge(segment: dict, error: float, *, max_points: int = 20000) -> list[list[float]]:
    """Bound every chord by sup(norm(f''))*h²/8, not midpoint heuristics.

    For f(t)=center+A*cos(t)+B*sin(t), A and B are perpendicular and
    norm(B)<=norm(A), so sup(norm(f''))<=norm(A). The declared error also
    propagates through the existing occurrence affine/material-side guard.
    """
    if (isinstance(error, bool) or not isinstance(error, (int, float))
            or not math.isfinite(error) or error <= 0 or max_points < 3):
        raise ValueError("invalid_native_ellipse_display_budget")
    center, major, ratio, start, sweep = _frame(segment)
    radius = math.hypot(*major)
    step = min(math.pi / 2, math.sqrt(8 * error / radius))
    if step == 0:
        raise ValueError("native_ellipse_exceeds_display_budget")
    count = max(2, math.ceil(abs(sweep) / step))
    if count + 1 > max_points:
        raise ValueError("native_ellipse_exceeds_display_budget")
    return [_at(center, major, ratio, start + sweep * i / count) for i in range(count + 1)]
