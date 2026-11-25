"""Bounded planar faces of finite native LINEs, never physical part regions.

Intersections split source parameter intervals without altering the native
segments. Junctions keep every source coordinate; no averaged/snap coordinates,
extensions, hidden-line completion, raster inputs or cross-block flood is used.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import math

from ezdxf.math import Vec2, is_point_in_polygon_2d

from research.dwg_pdf_visualization_20261031.geometry_relations import TOL, bounds
from research.dwg_route_validation_20261015.native_balloon_probe import finite_line_distance


def _at(line: dict, parameter: float) -> list[float]:
    return [a + parameter * (b - a) for a, b in zip(line["start"], line["end"])]


def _project(point: list, line: dict) -> float | None:
    check = finite_line_distance(point, line["start"], line["end"])
    if check["distance_native_units"] <= TOL and 0 <= check["t_unclamped"] <= 1:
        return check["t_unclamped"]
    return None


def _splits(lines: list[dict]) -> list[list[float]]:
    parameters = [{0., 1.} for _ in lines]
    for (i, left), (j, right) in combinations(enumerate(lines), 2):
        for point in [left["start"], left["end"]]:
            parameter = _project(point, right)
            if parameter is not None:
                parameters[j].add(parameter)
        for point in [right["start"], right["end"]]:
            parameter = _project(point, left)
            if parameter is not None:
                parameters[i].add(parameter)
        p, r = left["start"], right["start"]
        u = [b - a for a, b in zip(p, left["end"])]
        v = [b - a for a, b in zip(r, right["end"])]
        determinant = u[0] * v[1] - u[1] * v[0]
        if determinant == 0:
            continue
        delta = [b - a for a, b in zip(p, r)]
        t = (delta[0] * v[1] - delta[1] * v[0]) / determinant
        s = (delta[0] * u[1] - delta[1] * u[0]) / determinant
        if not (0 <= t <= 1 and 0 <= s <= 1):
            continue
        if abs(determinant) <= 1e-12 * math.hypot(*u) * math.hypot(*v):
            raise ValueError("numerically_ambiguous_line_intersection")
        if math.dist(_at(left, t), _at(right, s)) > TOL:
            raise ValueError("inconsistent_finite_line_intersection")
        parameters[i].add(t)
        parameters[j].add(s)
    result = []
    for line, values in zip(lines, parameters):
        ordered = []
        for value in sorted(values):
            if not ordered or math.dist(_at(line, value), _at(line, ordered[-1])) > TOL:
                ordered.append(value)
            elif value == 1.:
                ordered[-1] = value
        result.append(ordered)
    return result


def finite_faces(lines: list[dict]) -> dict:
    """Walk planar half-edges; preserve each derived edge's native interval."""
    lines = sorted(lines, key=lambda row: row["id"])
    if not 3 <= len(lines) <= 128:
        raise ValueError("finite_face_source_budget")
    points = [point for line in lines for point in (line["start"], line["end"])]
    if any(len(point) != 3 or not all(math.isfinite(value) for value in point) for point in points):
        raise ValueError("nonfinite_line_face_coordinate")
    if max(point[2] for point in points) - min(point[2] for point in points) > TOL:
        raise ValueError("nonplanar_line_face")
    if any(math.dist(line["start"], line["end"]) <= TOL for line in lines):
        raise ValueError("degenerate_line_face_source")
    parameters = _splits(lines)
    if sum(len(values) - 1 for values in parameters) > 512:
        raise ValueError("finite_face_split_budget")
    junctions, intervals = [], []

    def junction(point: list, source: str, parameter: float) -> int:
        matches = [i for i, row in enumerate(junctions) if any(math.dist(point, old["point"]) <= TOL for old in row)]
        if len(matches) > 1 or matches and any(math.dist(point, old["point"]) > TOL for old in junctions[matches[0]]):
            raise ValueError("ambiguous_nontransitive_junction")
        if not matches:
            junctions.append([])
        index = matches[0] if matches else len(junctions) - 1
        junctions[index].append({"source": source, "parameter": parameter, "point": point})
        return index

    segments = {}
    for line, values in zip(lines, parameters):
        for start, end in zip(values, values[1:]):
            a, b = _at(line, start), _at(line, end)
            u, v = junction(a, line["id"], start), junction(b, line["id"], end)
            if u == v:
                raise ValueError("collapsed_finite_face_interval")
            interval = {"source": line["id"], "source_ref": line["source_ref"], "parameter_interval": [start, end],
                        "start": a, "end": b, "junctions": [u, v]}
            ordinal = len(intervals)
            intervals.append(interval)
            segments.setdefault(tuple(sorted((u, v))), []).append(ordinal)
    if len(segments) > 512:
        raise ValueError("finite_face_split_budget")
    neighbors = defaultdict(list)
    centers = [row[0]["point"] for row in junctions]  # Topological ordering only, never rewritten geometry.
    for a, b in segments:
        neighbors[a].append(b)
        neighbors[b].append(a)
    for center, values in neighbors.items():
        values.sort(key=lambda i: math.atan2(centers[i][1] - centers[center][1], centers[i][0] - centers[center][0]))
        angles = [math.atan2(centers[i][1] - centers[center][1], centers[i][0] - centers[center][0]) for i in values]
        if any(abs(math.remainder(a - b, math.tau)) < 1e-12 for a, b in combinations(angles, 2)):
            raise ValueError("ambiguous_halfedge_order")
    visited, faces = set(), []
    for start in sorted((a, b) for a, values in neighbors.items() for b in values):
        if start in visited:
            continue
        walk, current = [], start
        while current not in visited:
            visited.add(current)
            walk.append(current)
            a, b = current
            around = neighbors[b]
            current = b, around[(around.index(a) - 1) % len(around)]
        if current != start or len({a for a, _ in walk}) != len(walk) or len(walk) < 3:
            continue  # Bridges and repeated vertices are not filled into a part.
        ring = [centers[a] for a, _ in walk]
        origin = ring[0]
        area = .5 * sum((a[0] - origin[0]) * (b[1] - origin[1]) - (b[0] - origin[0]) * (a[1] - origin[1])
                        for a, b in zip(ring, [*ring[1:], ring[0]]))
        if area <= TOL * TOL:
            continue  # Clockwise exterior, zero-area or retraced walk.
        sources = [index for a, b in walk for index in segments[tuple(sorted((a, b)))]]
        face = {"id": f"finite-face:{len(faces)}", "ring": [*ring, ring[0]], "area": area,
                "bounds": bounds(ring), "plane_z": origin[2], "status": "bounded_finite_line_face",
                "intervals": [intervals[i] for i in sources], "walk_junctions": [a for a, _ in walk],
                "boundary_source_ids": sorted({intervals[i]["source"] for i in sources}),
                "physical_owner_implied": False, "material_region_implied": False}
        faces.append(face)
    # A disconnected nested contour can bound a hole. Do not fill its outer
    # cycle or infer material parity from plain linework.
    for face in faces:
        nested = [other["id"] for other in faces if other is not face and
                  all(is_point_in_polygon_2d(Vec2(point), [Vec2(p) for p in face["ring"]], abs_tol=TOL) == 1
                      for point in other["ring"][:-1])]
        if nested:
            face.update(status="nested_boundary_unresolved", nested_faces=nested)
    return {"faces": faces, "junctions": junctions, "source_line_ids": [row["id"] for row in lines],
            "source_interval_count": len(intervals), "tolerance_native_units": TOL,
            "native_segments_changed": False, "intersections_imply_physical_joint": False,
            "physical_owner_implied": False, "material_region_implied": False}
