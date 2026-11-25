"""Audit proposed precedence without treating a linear list as proved order."""

from __future__ import annotations

from collections import deque
from graphlib import CycleError, TopologicalSorter
from itertools import combinations
from typing import Any


def dependency_layers(
    operation_ids: list[str], relations: list[dict[str, Any]]
) -> list[list[str]]:
    """Return stable dependency groups, not a parallel production schedule."""
    predecessors: dict[str, set[str]] = {key: set() for key in operation_ids}
    for relation in relations:
        before, after = relation["before"], relation["after"]
        if before not in predecessors or after not in predecessors or before == after:
            raise ValueError("Order references unknown or identical operations")
        predecessors[after].add(before)
    sorter = TopologicalSorter(predecessors)
    try:
        sorter.prepare()
    except CycleError as error:
        raise ValueError("Cyclic process order") from error
    positions = {key: index for index, key in enumerate(operation_ids)}
    layers = []
    while sorter.is_active():
        ready = sorted(sorter.get_ready(), key=positions.__getitem__)
        layers.append(ready)
        sorter.done(*ready)
    return layers


def precedence_paths(
    operation_ids: list[str], relations: list[dict[str, Any]]
) -> dict[tuple[str, str], list[str]]:
    """Find one shortest supporting path per pair in a validated order graph."""
    successors: dict[str, list[str]] = {key: [] for key in operation_ids}
    for relation in relations:
        successors[relation["before"]].append(relation["after"])
    paths = {}
    for start in operation_ids:
        queue = deque([[start]])
        seen = {start}
        while queue:
            path = queue.popleft()
            for child in successors[path[-1]]:
                if child not in seen:
                    seen.add(child)
                    next_path = [*path, child]
                    paths[start, child] = next_path
                    queue.append(next_path)
    return paths


def order_audit(plan: dict[str, Any]) -> dict[str, Any]:
    """Expose source orders, conditional orders and unproved route transitions.

    Literal citations and an acyclic graph do not certify manufacturing semantics.
    An explicit relation is still the model's reading of the supplied evidence.
    """
    ids = [operation["id"] for operation in plan["operations"]]
    relations = plan["order_relations"]
    layers = dependency_layers(ids, relations)
    explicit = [row for row in relations if row["basis"] == "explicit"]
    conditional = [row for row in relations if row["basis"] == "inferred"]
    source_paths = precedence_paths(ids, explicit)
    all_paths = precedence_paths(ids, relations)
    route_checks = []
    placed = set()
    for route in plan["routes"]:
        sequence = route["operation_ids"]
        placed.update(sequence)
        transitions = []
        for before, after in zip(sequence, sequence[1:]):
            if (after, before) in all_paths:
                raise ValueError("Route contradicts its declared ordering evidence")
            path = source_paths.get((before, after))
            basis = "explicit" if path else "conditional"
            path = path or all_paths.get((before, after))
            transitions.append(
                {
                    "before": before,
                    "after": after,
                    "support": basis if path else "unspecified",
                    "relation_path": path or [],
                }
            )
        route_checks.append({"rank": route["rank"], "transitions": transitions})
    return {
        "explicit_relations": explicit,
        "conditional_relations": conditional,
        "explicit_dependency_layers": dependency_layers(ids, explicit),
        "proposed_dependency_layers": layers,
        "layers_are_parallel_schedule": False,
        "unconstrained_pairs": [
            [a, b]
            for a, b in combinations(ids, 2)
            if (a, b) not in all_paths and (b, a) not in all_paths
        ],
        "unplaced_operation_ids": [key for key in ids if key not in placed],
        "route_checks": route_checks,
        "manufacturing_semantics_verified": False,
        "complete_route_verified": False,
    }
