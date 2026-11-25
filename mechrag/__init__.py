"""Evidence-first retrieval components for MechRAG."""

from .evidence import (
    AggregateResult,
    BoundingBox,
    EvidenceSource,
    EvidenceStore,
    EvidenceUnit,
    ViewRegion,
    deduplicate_cross_view,
)
from .questions import QueryCondition, QueryPlan, RuleBasedQuestionDecomposer
from .selection import (
    AnswerEvidenceVerifier,
    Claim,
    CoverageReport,
    EvidenceCoverageSelector,
)

__all__ = [
    "AggregateResult",
    "AnswerEvidenceVerifier",
    "BoundingBox",
    "Claim",
    "CoverageReport",
    "EvidenceCoverageSelector",
    "EvidenceSource",
    "EvidenceStore",
    "EvidenceUnit",
    "QueryCondition",
    "QueryPlan",
    "RuleBasedQuestionDecomposer",
    "ViewRegion",
    "deduplicate_cross_view",
]
