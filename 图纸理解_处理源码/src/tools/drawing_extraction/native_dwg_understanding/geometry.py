"""Finite contacts and endpoint topology, separate from physical part ownership."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
import math
from typing import Any

from research.dwg_route_validation_20261015.native_balloon_probe import (
    finite_line_distance,
)
from research.dwg_route_validation_20261015.native_curve_contacts import (
    finite_curve_contact,
)

TOLERANCE = 1e-7  # Native CAD units; a numerical tolerance, never a snapping radius.


def bbox(points: list) -> list[float] | None:
    if not points:
        return None
    return [
        min(p[0] for p in points),
        min(p[1] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points),
    ]


class GeometryIndex:
    """A bounded grid only accelerates exact finite native contact checks."""

    def __init__(self, geometry: list[dict[str, Any]]) -> None:
        self.rows = {r["id"]: r for r in geometry}
        self.buckets: dict[tuple[int, int], set[str]] = defaultdict(set)
        self.global_ids: set[str] = set()
        boxes = {}
        for row in geometry:
            circle_box = row.get("world_circle_aabb")
            boxes[row["id"]] = (
                [*circle_box["min"][:2], *circle_box["max"][:2]]
                if circle_box
                else bbox(row["points"])
            )
        extent = bbox([p for box in boxes.values() if box for p in (box[:2], box[2:])])
        self.cell = max(
            (max(extent[2] - extent[0], extent[3] - extent[1]) / 64) if extent else 1.0,
            TOLERANCE * 10,
        )
        for identifier, box in boxes.items():
            if box is None:
                continue
            first = self.key([box[0] - TOLERANCE, box[1] - TOLERANCE])
            last = self.key([box[2] + TOLERANCE, box[3] + TOLERANCE])
            if (last[0] - first[0] + 1) * (last[1] - first[1] + 1) > 256:
                self.global_ids.add(identifier)
            else:
                for x in range(first[0], last[0] + 1):
                    for y in range(first[1], last[1] + 1):
                        self.buckets[x, y].add(identifier)

    def key(self, point: list) -> tuple[int, int]:
        return math.floor(point[0] / self.cell), math.floor(point[1] / self.cell)

    def contacts(self, point: list) -> list[str]:
        result = []
        point = list(point[:3]) + [0.0] * max(0, 3 - len(point))
        for identifier in sorted(
            self.buckets.get(self.key(point), set()) | self.global_ids
        ):
            row = self.rows[identifier]
            if row.get("line_role") != "contour_solid_declared":
                continue
            if row["kind"] == "LINE":
                check = finite_line_distance(point, row["start"], row["end"])
                contact = check["numeric_contact"] and 0 <= check["t_unclamped"] <= 1
            elif row["kind"] in {"ARC", "CIRCLE"}:
                contact = finite_curve_contact(point, row)["numeric_contact"]
            else:
                contact = False
            if contact:
                result.append(identifier)
        return result


def topology_groups(
    geometry: list[dict], parts: list[dict]
) -> tuple[list[dict], dict[str, str]]:
    """Connected finite primitives are drawing groups, never complete physical parts.

    Endpoint distances are checked in XYZ after a bucket lookup. Nearby lines,
    centre lines and annotation lines cannot bridge two contour groups.
    """
    parent: dict[str, str] = {}
    vertices: dict[tuple[int, ...], list[tuple[list, str, int]]] = defaultdict(list)
    degree: dict[tuple[str, int], int] = defaultdict(lambda: 1)
    endpoints: dict[str, list] = {}

    def root(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    for row in geometry:
        if row.get("line_role") != "contour_solid_declared" or row["kind"] not in {
            "LINE",
            "ARC",
            "CIRCLE",
        }:
            continue
        identifier = row["id"]
        parent[identifier] = identifier
        if row["kind"] == "CIRCLE":
            endpoints[identifier] = []
            continue
        points = (
            [row["start"], row["end"]]
            if row["kind"] == "LINE"
            else [row["points"][0], row["points"][-1]]
        )
        # ARC display endpoints are exact native evaluations, not intermediate tessellation.
        endpoints[identifier] = points
        for slot, point in enumerate(points):
            point = list(point[:3]) + [0.0] * max(0, 3 - len(point))
            key = tuple(math.floor(v / TOLERANCE) for v in point)
            for delta in product((-1, 0, 1), repeat=3):
                for other_point, other, other_slot in vertices.get(
                    tuple(a + b for a, b in zip(key, delta)), []
                ):
                    if math.dist(point, other_point) <= TOLERANCE:
                        parent[root(identifier)] = root(other)
                        degree[identifier, slot] += 1
                        degree[other, other_slot] += 1
            vertices[key].append((point, identifier, slot))
    members: dict[str, list[str]] = defaultdict(list)
    for identifier in parent:
        members[root(identifier)].append(identifier)
    owners: dict[str, set[str]] = defaultdict(set)
    for part in parts:
        for identifier in part["geometry_ids"]:
            if identifier in parent:
                owners[root(identifier)].add(part["id"])
    groups, mapping = [], {}
    by_id = {r["id"]: r for r in geometry}
    for component, ids in sorted(members.items()):
        ids.sort()
        group_id = "topology:" + ids[0].removeprefix("geometry:")
        for identifier in ids:
            mapping[identifier] = group_id
        ends = sum(degree[i, s] == 1 for i in ids for s in range(len(endpoints[i])))
        branches = sum(degree[i, s] > 2 for i in ids for s in range(len(endpoints[i])))
        groups.append(
            {
                "id": group_id,
                "geometry_ids": ids,
                "part_candidates": sorted(owners[component]),
                "basis": "finite_native_endpoint_connectivity",
                "free_endpoints": ends,
                "branch_endpoints": branches,
                "closed_topology": ends == 0 and branches == 0,
                "bbox": bbox([p for i in ids for p in by_id[i]["points"]]),
                "physical_owner_confirmed": False,
                "relation_confidence": None,
            }
        )
    return groups, mapping
