"""Bounded display of native cubic HATCH splines, never exact line evidence.

Only nonrational, nonperiodic, clamped cubic B-splines are admitted. Decompose
into Bezier spans, then use their control hulls to bound chord error. An error
or exhausted budget rejects the boundary rather than replacing it by a chord.
"""

from __future__ import annotations

from collections import Counter
import math

from ezdxf.math import BSpline

__all__ = ["flatten_cubic_spline"]


def _point(value: list) -> list[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in value)):
        raise ValueError("invalid_native_spline_control_point")
    return list(value)


def _control_data(segment: dict) -> tuple[list[list[float]], list[float]]:
    if (segment.get("degree") != 3 or segment.get("is_rational") != 0
            or segment.get("is_periodic") != 0):
        raise ValueError("unsupported_native_hatch_spline_form")
    rows, knots = segment.get("control_points", []), segment.get("knots", [])
    if (not isinstance(rows, list) or not 4 <= len(rows) <= 10000
            or not isinstance(knots, list) or len(knots) != len(rows) + 4):
        raise ValueError("invalid_native_spline_counts")
    if "weights" in segment or any(not isinstance(row, dict) or "weight" in row for row in rows):
        raise ValueError("unexpected_native_spline_weights")
    points = [_point(row.get("point")) for row in rows]
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in knots) or any(a > b for a, b in zip(knots, knots[1:]))):
        raise ValueError("invalid_native_spline_knots")
    if (len(set(knots[:4])) != 1 or len(set(knots[-4:])) != 1 or knots[-1] <= knots[0]
            or any(n > 3 for n in Counter(knots[4:-4]).values())
            or any(v <= knots[0] or v >= knots[-1] for v in knots[4:-4])):
        raise ValueError("unclamped_or_discontinuous_native_spline")
    # ezdxf normalizes nonzero knot origins and uses isclose while decomposing.
    # Normalize explicitly and reject distinct spans it could silently merge.
    domain = knots[-1] - knots[0]
    if not math.isfinite(domain):
        raise ValueError("invalid_native_spline_knots")
    normalized = [(v - knots[0]) / domain for v in knots]
    if any(a != b and math.isclose(a, b) for a, b in zip(normalized, normalized[1:])):
        raise ValueError("native_spline_knots_below_decomposition_precision")
    if len(set(normalized)) != len(set(knots)):
        raise ValueError("native_spline_knots_below_decomposition_precision")
    return points, normalized


def _chord_distance(point: list, start: list, end: list) -> float:
    delta = [b - a for a, b in zip(start, end, strict=True)]
    squared = sum(v * v for v in delta)
    t = (max(0., min(1., sum((p - a) * d for p, a, d in
                              zip(point, start, delta, strict=True)) / squared)) if squared else 0.)
    return math.dist(point, [a + t * d for a, d in zip(start, delta, strict=True)])


def _split(points: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    levels = [points]
    while len(levels[-1]) > 1:
        levels.append([[(x + y) / 2 for x, y in zip(a, b, strict=True)]
                       for a, b in zip(levels[-1], levels[-1][1:])])
    return [row[0] for row in levels], [row[-1] for row in reversed(levels)]


def flatten_cubic_spline(segment: dict, error: float, *, max_points: int = 20000) -> list[list[float]]:
    """Return native-XY display points with a control-hull chord error bound.

    Each accepted chord is within ``error`` of its Bezier control hull. This
    bounds distance of the native curve to the chord without midpoint guesses.
    Original control points and knots remain in the hashed native response.
    """
    if not math.isfinite(error) or error <= 0 or max_points < 2:
        raise ValueError("invalid_native_spline_display_budget")
    controls, knots = _control_data(segment)
    spans = list(BSpline(controls, order=4, knots=knots).bezier_decomposition())
    if len(spans) != len(set(knots)) - 1:
        raise ValueError("native_spline_span_decomposition_mismatch")
    result = [controls[0]]
    for span in spans:
        polygon = [[point.x, point.y] for point in span]
        if len(polygon) != 4 or math.dist(result[-1], polygon[0]) > 1e-7:
            raise ValueError("gap_in_native_spline_decomposition")
        stack = [(polygon, 0)]
        while stack:
            points, depth = stack.pop()
            if max(_chord_distance(p, points[0], points[-1]) for p in points) <= error:
                if len(result) >= max_points:
                    raise ValueError("native_spline_exceeds_display_budget")
                result.append(points[-1])
            else:
                if depth >= 24:
                    raise ValueError("native_spline_exceeds_subdivision_budget")
                left, right = _split(points)
                stack.extend([(right, depth + 1), (left, depth + 1)])
    if math.dist(result[-1], controls[-1]) > 1e-7:
        raise ValueError("gap_in_native_spline_decomposition")
    return result
