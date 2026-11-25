"""Evidence-first orchestration shared by text and multimodal MechRAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .evidence import AggregateResult, EvidenceStore, EvidenceUnit
from .questions import QueryPlan, RuleBasedQuestionDecomposer
from .reranker import (
    MLPReranker,
    PairFeatureBuilder,
    PartEvidenceProfile,
    RankedPart,
    WeightedEvidenceReranker,
    rerank_candidates,
)
from .selection import (
    AnswerEvidenceVerifier,
    Claim,
    CoverageReport,
    EvidenceCoverageSelector,
    build_evidence_prompt,
)


@dataclass(frozen=True)
class RetrievalBundle:
    plan: QueryPlan
    candidate_part_ids: tuple[str, ...]
    reranked_parts: tuple[RankedPart, ...]
    selected_evidence: tuple[EvidenceUnit, ...]
    coverage: CoverageReport
    prompt: str
    aggregate: AggregateResult | None = None


class EvidenceFirstPipeline:
    """Retrieve, rerank, cover, and expose proof for every answer condition."""

    def __init__(
        self,
        store: EvidenceStore,
        *,
        decomposer: RuleBasedQuestionDecomposer | None = None,
        selector: EvidenceCoverageSelector | None = None,
        critical_properties: Sequence[str] = (
            "hole_count",
            "hole_type",
            "coaxial",
            "symmetry",
        ),
        reranker: WeightedEvidenceReranker | MLPReranker | None = None,
    ) -> None:
        self.store = store
        self.decomposer = decomposer or RuleBasedQuestionDecomposer()
        self.selector = selector or EvidenceCoverageSelector()
        self.feature_builder = PairFeatureBuilder(critical_properties)
        self.reranker = reranker or WeightedEvidenceReranker()
        self.verifier = AnswerEvidenceVerifier()

    def retrieve(
        self,
        question: str,
        *,
        overall_candidate_part_ids: Sequence[str] = (),
        query_profile: PartEvidenceProfile | None = None,
        candidate_profiles: Mapping[str, PartEvidenceProfile] | None = None,
        candidate_limit: int = 10,
        evidence_budget: float = 20,
    ) -> RetrievalBundle:
        plan = self.decomposer.decompose(question)
        aggregate = None
        mandatory_evidence = []

        # Exact global operators deliberately bypass semantic Top-K.
        if plan.aggregate_property and plan.aggregate_operation:
            aggregate = self.store.aggregate(
                plan.aggregate_property, plan.aggregate_operation
            )
            mandatory_evidence.append(aggregate.as_evidence_unit())

        candidate_ids = list(dict.fromkeys(overall_candidate_part_ids))
        ranked: list[RankedPart] = []
        if query_profile is not None and candidate_profiles:
            profiles = [
                candidate_profiles[part_id]
                for part_id in candidate_ids
                if part_id in candidate_profiles
            ]
            ranked = rerank_candidates(
                query_profile,
                profiles,
                feature_builder=self.feature_builder,
                scorer=self.reranker,
            )
            candidate_ids = [item.profile.part_id for item in ranked]
        candidate_ids = candidate_ids[:candidate_limit]

        local_evidence = self.store.for_parts(candidate_ids)
        # Mandatory aggregate evidence is included in the same coverage/cost
        # calculation, so an oversized result cannot be smuggled past budget.
        selection = self.selector.select(
            [*mandatory_evidence, *local_evidence],
            plan.conditions,
            budget=evidence_budget,
        )
        prompt = build_evidence_prompt(
            question,
            plan.conditions,
            selection.evidence,
            selection.report,
        )
        return RetrievalBundle(
            plan=plan,
            candidate_part_ids=tuple(candidate_ids),
            reranked_parts=tuple(ranked),
            selected_evidence=selection.evidence,
            coverage=selection.report,
            prompt=prompt,
            aggregate=aggregate,
        )

    def verify_response(self, response: str, bundle: RetrievalBundle):
        claims = self.verifier.claims_from_json(response)
        return self.verifier.verify(
            claims,
            bundle.selected_evidence,
            bundle.plan.conditions,
        )
