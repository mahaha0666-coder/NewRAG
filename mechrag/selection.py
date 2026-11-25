"""Budgeted complementary evidence selection and answer verification."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Iterable, Mapping, Sequence

from .evidence import EvidenceUnit
from .questions import QueryCondition


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def evidence_redundancy(left: EvidenceUnit, right: EvidenceUnit) -> float:
    """Estimate whether two records repeat the same underlying evidence."""

    if left.evidence_id == right.evidence_id:
        return 1.0
    left_faces = set(left.cad_face_ids)
    right_faces = set(right.cad_face_ids)
    face_overlap = 0.0
    if left.part_id == right.part_id and (left_faces or right_faces):
        face_overlap = len(left_faces & right_faces) / max(1, len(left_faces | right_faces))
    left_text = _tokens(left.summary)
    right_text = _tokens(right.summary)
    text_overlap = len(left_text & right_text) / max(1, len(left_text | right_text))
    same_feature_type = 1.0 if (
        left.part_id == right.part_id and left.feature_type == right.feature_type
    ) else 0.0
    return max(face_overlap, text_overlap, same_feature_type * 0.5)


class EvidenceSupportScorer:
    """Score how explicitly one evidence unit supports one query condition."""

    def score(self, unit: EvidenceUnit, condition: QueryCondition) -> float:
        if condition.exhaustive:
            if unit.feature_type != "structured_aggregate":
                return 0.0
            aggregate_property = unit.geometric_properties.get("aggregate_property")
            if aggregate_property not in condition.property_names:
                return 0.0
            # Partial scans may be shown as a warning, but may not satisfy an
            # exhaustive condition or validate a global claim.
            return 1.0 if unit.geometric_properties.get("complete") else 0.4

        if not condition.property_names:
            query_tokens = _tokens(condition.description)
            evidence_tokens = _tokens(unit.summary + " " + unit.feature_type)
            return len(query_tokens & evidence_tokens) / max(1, len(query_tokens))

        present = sum(
            name in unit.geometric_properties for name in condition.property_names
        )
        score = present / len(condition.property_names)
        if unit.feature_type == "structured_aggregate":
            aggregate_property = unit.geometric_properties.get("aggregate_property")
            if aggregate_property in condition.property_names:
                # An incomplete scan is useful context, but is not proof of a
                # global maximum/minimum.
                return 1.0 if unit.geometric_properties.get("complete") else 0.4
        return score


@dataclass(frozen=True)
class CoverageReport:
    condition_scores: Mapping[str, float]
    selected_evidence_ids: tuple[str, ...]
    total_cost: float
    budget: float

    @property
    def coverage(self) -> float:
        if not self.condition_scores:
            return 1.0
        return sum(self.condition_scores.values()) / len(self.condition_scores)

    @property
    def uncovered_condition_ids(self) -> tuple[str, ...]:
        return tuple(
            condition_id
            for condition_id, score in self.condition_scores.items()
            if score < 1.0
        )


@dataclass(frozen=True)
class SelectionResult:
    evidence: tuple[EvidenceUnit, ...]
    report: CoverageReport


class EvidenceCoverageSelector:
    """Greedy weighted coverage with explicit redundancy and cost penalties."""

    def __init__(
        self,
        *,
        redundancy_penalty: float = 0.2,
        support_scorer: EvidenceSupportScorer | None = None,
    ) -> None:
        if not 0 <= redundancy_penalty <= 1:
            raise ValueError("redundancy_penalty must be in [0, 1]")
        self.redundancy_penalty = redundancy_penalty
        self.support_scorer = support_scorer or EvidenceSupportScorer()

    def select(
        self,
        evidence: Iterable[EvidenceUnit],
        conditions: Sequence[QueryCondition],
        *,
        budget: float,
    ) -> SelectionResult:
        if budget <= 0:
            raise ValueError("budget must be positive")
        remaining = list(evidence)
        selected: list[EvidenceUnit] = []
        coverage = {condition.condition_id: 0.0 for condition in conditions}
        condition_map = {condition.condition_id: condition for condition in conditions}
        spent = 0.0

        while remaining:
            best_index = None
            best_utility = 0.0
            for index, unit in enumerate(remaining):
                if spent + unit.cost > budget:
                    continue
                gain = 0.0
                for condition_id, condition in condition_map.items():
                    support = self.support_scorer.score(unit, condition)
                    gain += condition.weight * max(0.0, support - coverage[condition_id])
                redundancy = max(
                    (evidence_redundancy(unit, prior) for prior in selected),
                    default=0.0,
                )
                utility = (gain - self.redundancy_penalty * redundancy) / unit.cost
                if utility > best_utility:
                    best_index = index
                    best_utility = utility

            if best_index is None:
                break
            unit = remaining.pop(best_index)
            selected.append(unit)
            spent += unit.cost
            for condition_id, condition in condition_map.items():
                coverage[condition_id] = max(
                    coverage[condition_id], self.support_scorer.score(unit, condition)
                )

            if all(value >= 1.0 for value in coverage.values()):
                break

        report = CoverageReport(
            condition_scores=coverage,
            selected_evidence_ids=tuple(unit.evidence_id for unit in selected),
            total_cost=spent,
            budget=budget,
        )
        return SelectionResult(tuple(selected), report)


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    condition_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    unsupported_claim_ids: tuple[str, ...]
    unknown_evidence_ids: tuple[str, ...]
    uncovered_condition_ids: tuple[str, ...]


class AnswerEvidenceVerifier:
    """Reject claims that do not point to evidence supporting their condition."""

    def __init__(self, support_scorer: EvidenceSupportScorer | None = None) -> None:
        self.support_scorer = support_scorer or EvidenceSupportScorer()

    def verify(
        self,
        claims: Sequence[Claim],
        evidence: Sequence[EvidenceUnit],
        conditions: Sequence[QueryCondition],
    ) -> VerificationResult:
        evidence_map = {unit.evidence_id: unit for unit in evidence}
        condition_map = {condition.condition_id: condition for condition in conditions}
        unsupported = []
        unknown = set()
        covered_conditions = set()

        for claim in claims:
            if not claim.evidence_ids or not claim.condition_ids:
                unsupported.append(claim.claim_id)
                continue
            claim_supported = True
            for evidence_id in claim.evidence_ids:
                if evidence_id not in evidence_map:
                    unknown.add(evidence_id)
                    claim_supported = False
            for condition_id in claim.condition_ids:
                condition = condition_map.get(condition_id)
                if condition is None:
                    claim_supported = False
                    continue
                supporting_ids = [
                    evidence_id
                    for evidence_id in claim.evidence_ids
                    if evidence_id in evidence_map
                    and self.support_scorer.score(evidence_map[evidence_id], condition)
                    >= (1.0 if condition.exhaustive else 1e-12)
                ]
                if supporting_ids:
                    covered_conditions.add(condition_id)
                else:
                    claim_supported = False
            if not claim_supported:
                unsupported.append(claim.claim_id)

        uncovered = tuple(
            condition_id
            for condition_id in condition_map
            if condition_id not in covered_conditions
        )
        return VerificationResult(
            valid=not unsupported and not unknown and not uncovered,
            unsupported_claim_ids=tuple(dict.fromkeys(unsupported)),
            unknown_evidence_ids=tuple(sorted(unknown)),
            uncovered_condition_ids=uncovered,
        )

    @staticmethod
    def claims_from_json(response: str) -> tuple[Claim, ...]:
        payload = json.loads(response)
        claims = payload["claims"] if isinstance(payload, dict) else payload
        return tuple(
            Claim(
                claim_id=str(item["claim_id"]),
                text=str(item["text"]),
                condition_ids=tuple(map(str, item.get("condition_ids", []))),
                evidence_ids=tuple(map(str, item.get("evidence_ids", []))),
            )
            for item in claims
        )


def build_evidence_prompt(
    question: str,
    conditions: Sequence[QueryCondition],
    evidence: Sequence[EvidenceUnit],
    coverage: CoverageReport,
) -> str:
    """Build a generation prompt with an auditable claim/evidence contract."""

    condition_text = "\n".join(
        f"- {condition.condition_id}: {condition.description}; "
        f"properties={list(condition.property_names)}; exhaustive={condition.exhaustive}"
        for condition in conditions
    )
    evidence_text = "\n".join(unit.prompt_block() for unit in evidence)
    citable_evidence_ids = [unit.evidence_id for unit in evidence]
    return f"""You are an evidence-grounded mechanical engineering assistant.

Question:
{question}

Conditions that must be verified:
{condition_text}

Selected evidence (coverage={coverage.coverage:.3f}, budget={coverage.total_cost:.2f}/{coverage.budget:.2f}):
{evidence_text}

Citable evidence IDs (the complete allow-list):
{json.dumps(citable_evidence_ids, ensure_ascii=False)}

Rules:
1. Use only the evidence above. Do not turn visual similarity into a geometric fact.
2. Every key engineering judgment must identify the feature, view/region or CAD faces,
   geometric property, and source that support it.
3. A global maximum/minimum is allowed only when a structured_aggregate record has
   complete=true. Otherwise say that the global claim is not verified.
4. Return strict JSON with this shape:
{{"claims":[{{"claim_id":"J1","text":"...","condition_ids":["C1"],
"evidence_ids":["part:feature"]}}],"answer":"..."}}
5. Every value in `evidence_ids` must be copied exactly from the citable allow-list.
   IDs nested inside an aggregate's properties are provenance only and must not be cited.
"""
