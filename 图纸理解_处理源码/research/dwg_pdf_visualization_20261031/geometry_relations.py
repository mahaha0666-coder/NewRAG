"""Finite geometric incidences, distinct from physical joints or part ownership."""

from __future__ import annotations

import math
from typing import Any

from research.dwg_route_validation_20261015.native_balloon_probe import finite_line_distance
from research.dwg_route_validation_20261015.native_curve_contacts import finite_curve_contact

TOL = 1e-7  # Native drawing units, not a measurement accuracy or snapping radius.


def bounds(points: list[list[float]]) -> list[float]:
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]


def boxes_overlap(a: list[float], b: list[float]) -> bool:
    return a[0] <= b[2] + TOL and a[2] + TOL >= b[0] and a[1] <= b[3] + TOL and a[3] + TOL >= b[1]


def line_relation(a: dict, b: dict) -> tuple[str, dict] | None:
    """Return one strongest geometric relation without using infinite extensions.

    Proper crossings require matching interpolated Z. Parallel collinear overlap
    is retained separately from complete coincidence and endpoint contact.
    """
    p, q, r, s = a["start"], a["end"], b["start"], b["end"]
    if min(math.dist(p, q), math.dist(r, s)) <= TOL or not boxes_overlap(bounds([p, q]), bounds([r, s])):
        return None
    if ((math.dist(p, r) <= TOL and math.dist(q, s) <= TOL)
            or (math.dist(p, s) <= TOL and math.dist(q, r) <= TOL)):
        return "coincident_line", {"tolerance_native_units": TOL}
    checks = [(point, finite_line_distance(point, left, right))
              for point, left, right in [(p, r, s), (q, r, s), (r, p, q), (s, p, q)]]
    contacts = [point for point, check in checks if check["numeric_contact"]]
    if contacts:
        u = [q[i] - p[i] for i in range(3)]
        v = [s[i] - r[i] for i in range(3)]
        cross = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        if math.hypot(*cross) <= TOL * min(math.hypot(*u), math.hypot(*v)):
            axis = max(range(3), key=lambda i: abs(u[i]))
            overlap = min(max(p[axis], q[axis]), max(r[axis], s[axis])) - max(min(p[axis], q[axis]), min(r[axis], s[axis]))
            if overlap > TOL:
                return "collinear_overlap", {"contact_points": contacts, "tolerance_native_units": TOL}
        return "finite_line_contact", {"contact_points": contacts,
                                       "checks": [check for _, check in checks if check["numeric_contact"]]}
    dx, dy, ex, ey = q[0] - p[0], q[1] - p[1], s[0] - r[0], s[1] - r[1]
    denominator = dx * ey - dy * ex
    if abs(denominator) <= 1e-12 * max(1., math.hypot(dx, dy) * math.hypot(ex, ey)):
        return None
    t = ((r[0] - p[0]) * ey - (r[1] - p[1]) * ex) / denominator
    u = ((r[0] - p[0]) * dy - (r[1] - p[1]) * dx) / denominator
    if 0 < t < 1 and 0 < u < 1:
        left = [p[i] + t * (q[i] - p[i]) for i in range(3)]
        right = [r[i] + u * (s[i] - r[i]) for i in range(3)]
        if math.dist(left, right) <= TOL:
            return "finite_line_crossing", {"point": left, "parameters": [t, u],
                                             "physical_joint_confirmed": False}
    return None


def topology(lines: list[dict]) -> list[dict[str, Any]]:
    """Sweep native X bounds; emit relationships without a connectivity flood."""
    result: list[dict[str, Any]] = []
    ordered = sorted([(bounds([row["start"], row["end"]]), row) for row in lines], key=lambda x: x[0][0])
    for index, (box, left) in enumerate(ordered):
        for other_box, right in ordered[index + 1:]:
            if other_box[0] > box[2] + TOL:
                break
            if not boxes_overlap(box, other_box):
                continue
            relation = line_relation(left, right)
            if relation:
                kind, evidence = relation
                result.append({"source": left["id"], "target": right["id"], "kind": kind, "evidence": evidence})
    return result


def point_contacts(point: list[float], lines: list[dict], curves: list[dict]) -> list[dict]:
    result = []
    for line in lines:
        if math.dist(line["start"], line["end"]) <= TOL or not boxes_overlap(bounds([point]), bounds([line["start"], line["end"]])):
            continue
        check = finite_line_distance(point, line["start"], line["end"])
        if check["numeric_contact"]:
            result.append({"target": line["id"], "kind": "pointer_on_finite_line", "evidence": check})
    for curve in curves:
        check = finite_curve_contact(point, curve)
        if check["numeric_contact"]:
            result.append({"target": curve["id"], "kind": "pointer_on_finite_curve", "evidence": check})
    return result
