"""Risk-based routing for localized drawing semantics."""

from __future__ import annotations

from typing import Any, Literal

Route = Literal["small_4b", "high_27b", "manual_review"]

LOW_RISK_TASKS = frozenset({"bom_row_gap", "technical_scope", "ocr_gap"})
HIGH_RISK_TASKS = frozenset({"bom_conflict", "leader_or_relation", "geometry_attribution"})


def route_task(task: dict[str, Any]) -> Route:
    """Select the smallest model that can adjudicate a bounded task."""
    kind = str(task.get("kind") or "")
    if kind in LOW_RISK_TASKS:
        return "small_4b"
    if kind in HIGH_RISK_TASKS:
        return "high_27b"
    return "manual_review"


def route_tasks(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    routed = {"small_4b": [], "high_27b": [], "manual_review": []}
    for task in tasks:
        routed[route_task(task)].append({**task, "model_tier": route_task(task)})
    return routed


__all__ = ["HIGH_RISK_TASKS", "LOW_RISK_TASKS", "route_task", "route_tasks"]
