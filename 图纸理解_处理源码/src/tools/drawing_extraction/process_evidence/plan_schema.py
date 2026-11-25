"""Reviewable process proposals, separate from the frozen route predictor."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .geometry import checked_box
from .plan_order import dependency_layers, precedence_paths


class PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Citation(PlanModel):
    evidence_id: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=1, max_length=700)


def bind_citation_sources(schema: dict, records: list[dict]) -> None:
    """Constrain structured citations to exact source values during decoding.

    Table rows and native symbols are structured records, not a printed sentence.
    A model may select one literal value or the complete serialized record; it
    may not concatenate fields and pass that synthesis off as a quotation.
    Free text and visual descriptions still pass the usual literal validator.
    """
    special = {}
    for row in records:
        try:
            value = json.loads(row["text"])
        except (ValueError, TypeError):
            continue
        if not isinstance(value, dict):
            continue
        if {"native_class", "raw_strings"}.issubset(value):
            fields = [value["native_class"], *value["raw_strings"]]
        elif {"sequence", "drawing_number", "name"}.issubset(value):
            fields = list(value.values())
        else:
            continue
        choices = [
            item
            for item in [row["text"], *fields]
            if isinstance(item, str) and 0 < len(item) <= 700 and item in row["text"]
        ]
        if choices:
            special[row["id"]] = list(dict.fromkeys(choices))
    if not special:
        return
    original = schema["$defs"]["Citation"]
    branches = []
    for identifier, choices in special.items():
        branch = deepcopy(original)
        branch["properties"]["evidence_id"]["const"] = identifier
        branch["properties"]["quote"]["enum"] = choices
        branches.append(branch)
    remaining = [row["id"] for row in records if row["id"] not in special]
    free = deepcopy(original)
    allowed = [{"type": "string", "pattern": r"^v[1-9][0-9]*$"}]
    if remaining:
        allowed.append({"type": "string", "enum": remaining})
    free["properties"]["evidence_id"] = {"anyOf": allowed}
    schema["$defs"]["Citation"] = {"anyOf": [*branches, free]}


class DrawingObservation(PlanModel):
    id: str = Field(pattern=r"^v[1-9][0-9]*$")
    image_id: str
    bbox_coordinate_system: Literal["normalized_1000", "image_pixels"]
    bbox: list[float] = Field(min_length=4, max_length=4)
    description: str = Field(min_length=1, max_length=260)
    native_evidence_ids: list[str] = Field(max_length=5)


def normalized_drawing_box(
    box: list[float], coordinate_system: str, image: dict
) -> list[float]:
    if coordinate_system == "image_pixels":
        box = [
            value * 1000 / image["width" if index % 2 == 0 else "height"]
            for index, value in enumerate(box)
        ]
    elif coordinate_system != "normalized_1000":
        raise ValueError("Unknown drawing box coordinate system")
    return checked_box(box)


class ProcessOperation(PlanModel):
    id: str = Field(pattern=r"^p[1-9][0-9]*$")
    name: str = Field(min_length=1, max_length=50)
    basis: Literal["explicit", "inferred"]
    target: str = Field(min_length=1, max_length=120)
    evidence: list[Citation] = Field(min_length=1, max_length=5)
    explanation: str = Field(min_length=1, max_length=300)
    conditions: list[str] = Field(max_length=6)


class ProcessOrder(PlanModel):
    before: str
    after: str
    basis: Literal["explicit", "inferred"]
    evidence: list[Citation] = Field(min_length=1, max_length=5)
    explanation: str = Field(min_length=1, max_length=260)
    scope: str | None = Field(default=None, min_length=1, max_length=160)
    conditions: list[str] = Field(default_factory=list, max_length=6)


class ProcessRoute(PlanModel):
    rank: int = Field(ge=1, le=3)
    operation_ids: list[str] = Field(min_length=1, max_length=16)
    explanation: str = Field(min_length=1, max_length=300)


class ProcessPlan(PlanModel):
    needs_review: Literal[True]
    summary: str = Field(min_length=1, max_length=300)
    observations: list[DrawingObservation] = Field(default_factory=list, max_length=12)
    operations: list[ProcessOperation] = Field(max_length=16)
    order_relations: list[ProcessOrder] = Field(max_length=20)
    routes: list[ProcessRoute] = Field(max_length=3)
    unresolved: list[str] = Field(max_length=12)


def _check_citations(citations: list[Citation], evidence: dict[str, dict]) -> None:
    for citation in citations:
        source = evidence.get(citation.evidence_id)
        if source is None or citation.quote not in source["text"]:
            raise ValueError("Process proposal has missing or rewritten evidence")


def validate_order_relations(
    values: list[dict],
    operation_ids: list[str],
    evidence: dict[str, dict],
    *,
    require_context: bool = True,
) -> list[dict]:
    """Validate the same order contract for joint and separate model stages."""
    edges = set()
    relations = []
    for value in values:
        relation = ProcessOrder.model_validate(value)
        _check_citations(relation.evidence, evidence)
        if relation.basis == "explicit" and all(
            evidence[c.evidence_id].get("visual") for c in relation.evidence
        ):
            raise ValueError(
                "A drawing interpretation is not an explicit order requirement"
            )
        if require_context and not relation.scope:
            raise ValueError("Order must name the object or stage it applies to")
        if require_context and relation.basis == "inferred" and not relation.conditions:
            raise ValueError("Inferred order must retain its applicability conditions")
        edge = (relation.before, relation.after)
        if edge in edges:
            raise ValueError("Duplicate order relation")
        edges.add(edge)
        relations.append(relation.model_dump(exclude_unset=True))
    dependency_layers(operation_ids, relations)
    return relations


def validate_plan(
    value: dict,
    records: list[dict],
    images: list[dict] | None = None,
    *,
    require_order_context: bool = False,
) -> dict:
    """Check literal support and graph consistency without certifying semantics."""
    plan = ProcessPlan.model_validate(value)
    evidence = {row["id"]: row for row in records}
    image_by_id = {image["image_id"]: image for image in (images or [])}
    for observation in plan.observations:
        if observation.id in evidence or observation.image_id not in image_by_id:
            raise ValueError("Missing image or duplicate visual observation")
        normalized_drawing_box(
            observation.bbox,
            observation.bbox_coordinate_system,
            image_by_id[observation.image_id],
        )
        if any(
            key not in {row["id"] for row in records}
            for key in observation.native_evidence_ids
        ):
            raise ValueError("Visual observation references unknown native evidence")
        evidence[observation.id] = {"text": observation.description, "visual": True}
    operations = {row.id: row for row in plan.operations}
    if len(operations) != len(plan.operations):
        raise ValueError("Duplicate operation ID")
    if bool(operations) != bool(plan.routes):
        raise ValueError("Operations and proposed routes must coexist")

    for operation in plan.operations:
        _check_citations(operation.evidence, evidence)
        if operation.basis == "explicit" and all(
            evidence[c.evidence_id].get("visual") for c in operation.evidence
        ):
            raise ValueError(
                "A drawing interpretation is not an explicit native requirement"
            )
        if operation.basis == "inferred" and not operation.conditions:
            raise ValueError(
                "Inferred operation must retain its applicability conditions"
            )
    relations = validate_order_relations(
        value["order_relations"],
        list(operations),
        evidence,
        require_context=require_order_context,
    )
    paths = precedence_paths(list(operations), relations)
    routes = set()
    for rank, route in enumerate(plan.routes, start=1):
        if route.rank != rank or len(set(route.operation_ids)) != len(
            route.operation_ids
        ):
            raise ValueError("Invalid route rank or repeated operation instance")
        if any(key not in operations for key in route.operation_ids):
            raise ValueError("Route references unknown operation")
        signature = tuple(route.operation_ids)
        if signature in routes:
            raise ValueError("Duplicate proposed route")
        routes.add(signature)
        positions = {key: index for index, key in enumerate(route.operation_ids)}
        # Check paths as well as direct edges. A route may omit an intermediate
        # operation; that omission must not let it reverse a declared dependency.
        if any(
            positions[a] >= positions[b]
            for a, b in paths
            if a in positions and b in positions
        ):
            raise ValueError("Route contradicts its declared ordering evidence")
    if not operations and not plan.unresolved:
        raise ValueError("Empty proposal must explain the missing evidence")
    result = plan.model_dump(exclude_unset=True)
    result.setdefault("observations", [])
    return result
