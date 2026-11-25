"""Evidence-aware reranking for visually similar CAD parts.

Stage 1 remains a frozen, whole-image CLIP retrieval.  Stage 2 compares local
feature evidence and is either a transparent weighted scorer or a small MLP.
The training utilities mine high-CLIP-similarity/property-mismatch pairs and
split them by variant family to prevent view/variant leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .evidence import EvidenceUnit, ViewRegion


FEATURE_NAMES = (
    "overall_clip_similarity",
    "local_alignment_max",
    "local_alignment_mean",
    "critical_property_match",
    "critical_property_mismatch",
    "local_feature_coverage",
    "multi_view_coverage",
)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    if not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(frozen=True)
class PartEvidenceProfile:
    part_id: str
    family_id: str
    global_embedding: tuple[float, ...]
    evidence: tuple[EvidenceUnit, ...]

    def __post_init__(self) -> None:
        if any(unit.part_id != self.part_id for unit in self.evidence):
            raise ValueError("Every evidence unit must belong to the profile part")

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "family_id": self.family_id,
            "global_embedding": list(self.global_embedding),
            "evidence": [unit.to_dict() for unit in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PartEvidenceProfile":
        return cls(
            part_id=str(data["part_id"]),
            family_id=str(data["family_id"]),
            global_embedding=tuple(map(float, data["global_embedding"])),
            evidence=tuple(
                EvidenceUnit.from_dict(unit) for unit in data.get("evidence", [])
            ),
        )


def load_profiles_jsonl(path: str | Path) -> list[PartEvidenceProfile]:
    profiles = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    profiles.append(PartEvidenceProfile.from_dict(json.loads(line)))
                except Exception as exc:
                    raise ValueError(f"Invalid profile at line {line_number}") from exc
    return profiles


def save_profiles_jsonl(
    profiles: Iterable[PartEvidenceProfile], path: str | Path
) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for profile in profiles:
            handle.write(json.dumps(profile.to_dict(), ensure_ascii=False) + "\n")


def _property_signature(
    profile: PartEvidenceProfile, property_names: Sequence[str]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    signature = []
    for property_name in property_names:
        values = {
            repr(unit.geometric_properties[property_name])
            for unit in profile.evidence
            if property_name in unit.geometric_properties
        }
        signature.append((property_name, tuple(sorted(values))))
    return tuple(signature)


def _region_similarities(
    query: PartEvidenceProfile, candidate: PartEvidenceProfile
) -> tuple[list[float], int, int]:
    similarities: list[float] = []
    matched_features = 0
    total_features = 0
    for query_unit in query.evidence:
        query_regions = [region for region in query_unit.view_regions if region.embedding]
        if not query_regions:
            continue
        total_features += 1
        candidates = [
            unit
            for unit in candidate.evidence
            if unit.feature_type == query_unit.feature_type
        ]
        best_for_feature = -1.0
        for query_region in query_regions:
            for candidate_unit in candidates:
                for candidate_region in candidate_unit.view_regions:
                    if candidate_region.embedding:
                        best_for_feature = max(
                            best_for_feature,
                            cosine_similarity(
                                query_region.embedding, candidate_region.embedding
                            ),
                        )
        if best_for_feature > -1:
            matched_features += 1
            similarities.append(best_for_feature)
    return similarities, matched_features, total_features


@dataclass(frozen=True)
class PairExample:
    query_part_id: str
    candidate_part_id: str
    query_family_id: str
    candidate_family_id: str
    features: tuple[float, ...]
    label: int

    def __post_init__(self) -> None:
        if len(self.features) != len(FEATURE_NAMES):
            raise ValueError("Unexpected reranker feature dimension")
        if self.label not in {0, 1}:
            raise ValueError("label must be 0 or 1")


class PairFeatureBuilder:
    def __init__(self, critical_properties: Sequence[str]) -> None:
        if not critical_properties:
            raise ValueError("At least one critical property is required")
        self.critical_properties = tuple(critical_properties)

    def build(
        self, query: PartEvidenceProfile, candidate: PartEvidenceProfile
    ) -> tuple[float, ...]:
        overall = cosine_similarity(query.global_embedding, candidate.global_embedding)
        local_scores, matched_features, total_features = _region_similarities(
            query, candidate
        )
        local_max = max(local_scores, default=0.0)
        local_mean = sum(local_scores) / len(local_scores) if local_scores else 0.0

        query_signature = dict(_property_signature(query, self.critical_properties))
        candidate_signature = dict(
            _property_signature(candidate, self.critical_properties)
        )
        comparable = 0
        matches = 0
        mismatches = 0
        for property_name in self.critical_properties:
            left = query_signature[property_name]
            right = candidate_signature[property_name]
            if not left or not right:
                continue
            comparable += 1
            if left == right:
                matches += 1
            else:
                mismatches += 1
        property_match = matches / comparable if comparable else 0.0
        property_mismatch = mismatches / comparable if comparable else 0.0
        local_coverage = matched_features / total_features if total_features else 0.0

        query_views = {
            region.view_id for unit in query.evidence for region in unit.view_regions
        }
        candidate_views = {
            region.view_id for unit in candidate.evidence for region in unit.view_regions
        }
        multi_view_coverage = min(
            1.0, len(candidate_views) / max(1, len(query_views))
        )
        return (
            overall,
            local_max,
            local_mean,
            property_match,
            property_mismatch,
            local_coverage,
            multi_view_coverage,
        )


class HardNegativeMiner:
    """Mine similar-looking parts that disagree on critical structure."""

    def __init__(
        self,
        critical_properties: Sequence[str],
        *,
        candidates_per_query: int = 20,
        minimum_clip_similarity: float = 0.0,
    ) -> None:
        self.builder = PairFeatureBuilder(critical_properties)
        self.critical_properties = tuple(critical_properties)
        self.candidates_per_query = candidates_per_query
        self.minimum_clip_similarity = minimum_clip_similarity

    def mine(self, profiles: Sequence[PartEvidenceProfile]) -> list[PairExample]:
        examples = []
        for query in profiles:
            candidates = sorted(
                (candidate for candidate in profiles if candidate.part_id != query.part_id),
                key=lambda candidate: cosine_similarity(
                    query.global_embedding, candidate.global_embedding
                ),
                reverse=True,
            )[: self.candidates_per_query]
            query_signature = _property_signature(query, self.critical_properties)
            for candidate in candidates:
                similarity = cosine_similarity(
                    query.global_embedding, candidate.global_embedding
                )
                if similarity < self.minimum_clip_similarity:
                    continue
                candidate_signature = _property_signature(
                    candidate, self.critical_properties
                )
                # Equal critical structure is a positive correspondence;
                # visually close but structurally different is a hard negative.
                observed = any(values for _, values in query_signature) or any(
                    values for _, values in candidate_signature
                )
                if not observed:
                    continue
                label = int(query_signature == candidate_signature)
                examples.append(
                    PairExample(
                        query_part_id=query.part_id,
                        candidate_part_id=candidate.part_id,
                        query_family_id=query.family_id,
                        candidate_family_id=candidate.family_id,
                        features=self.builder.build(query, candidate),
                        label=label,
                    )
                )
        return examples


@dataclass(frozen=True)
class FamilySplit:
    train: tuple[PairExample, ...]
    test: tuple[PairExample, ...]
    dropped_cross_split: tuple[PairExample, ...]
    train_families: tuple[str, ...]
    test_families: tuple[str, ...]


def split_pairs_by_family(
    examples: Sequence[PairExample], *, test_ratio: float = 0.2, seed: int = 42
) -> FamilySplit:
    """Keep every part/view/variant family entirely in one data split."""

    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    families = sorted(
        {
            family
            for example in examples
            for family in (example.query_family_id, example.candidate_family_id)
        }
    )
    if len(families) < 2:
        raise ValueError("At least two families are required for a leak-free split")
    random.Random(seed).shuffle(families)
    test_count = max(1, min(len(families) - 1, round(len(families) * test_ratio)))
    test_families = set(families[:test_count])
    train_families = set(families[test_count:])
    train = []
    test = []
    dropped = []
    for example in examples:
        pair_families = {example.query_family_id, example.candidate_family_id}
        if pair_families <= train_families:
            train.append(example)
        elif pair_families <= test_families:
            test.append(example)
        else:
            dropped.append(example)
    return FamilySplit(
        train=tuple(train),
        test=tuple(test),
        dropped_cross_split=tuple(dropped),
        train_families=tuple(sorted(train_families)),
        test_families=tuple(sorted(test_families)),
    )


def make_single_feature_variant(
    base: PartEvidenceProfile,
    *,
    variant_part_id: str,
    feature_id: str,
    property_name: str,
    new_value: Any,
    global_embedding: Sequence[float] | None = None,
    replacement_regions: Sequence[ViewRegion] | None = None,
) -> PartEvidenceProfile:
    """Register an editable-CAD variant in which exactly one property changes."""

    changed = 0
    units = []
    for unit in base.evidence:
        properties = dict(unit.geometric_properties)
        if unit.feature_id == feature_id:
            if property_name not in properties:
                raise KeyError(f"{feature_id} has no property {property_name}")
            if properties[property_name] == new_value:
                raise ValueError("The variant must change the selected property")
            properties[property_name] = new_value
            changed += 1
            unit = replace(
                unit,
                part_id=variant_part_id,
                geometric_properties=properties,
                view_regions=tuple(replacement_regions or unit.view_regions),
                family_id=base.family_id,
            )
        else:
            unit = replace(unit, part_id=variant_part_id, family_id=base.family_id)
        units.append(unit)
    if changed != 1:
        raise ValueError("feature_id must identify exactly one evidence unit")
    return PartEvidenceProfile(
        part_id=variant_part_id,
        family_id=base.family_id,
        global_embedding=tuple(
            global_embedding if global_embedding is not None else base.global_embedding
        ),
        evidence=tuple(units),
    )


class WeightedEvidenceReranker:
    """Deterministic fallback before an MLP checkpoint has been trained."""

    DEFAULT_WEIGHTS = (0.2, 0.2, 0.15, 0.3, -0.45, 0.1, 0.05)

    def __init__(self, weights: Sequence[float] = DEFAULT_WEIGHTS) -> None:
        if len(weights) != len(FEATURE_NAMES):
            raise ValueError("Unexpected weight dimension")
        self.weights = tuple(weights)

    def score(self, features: Sequence[float]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, features))


class MLPReranker:
    """A small trainable scorer over frozen visual/evidence features."""

    def __init__(self, *, hidden_dims: Sequence[int] = (16, 8), seed: int = 42) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("MLPReranker requires PyTorch") from exc
        torch.manual_seed(seed)
        layers = []
        input_dim = len(FEATURE_NAMES)
        for hidden_dim in hidden_dims:
            layers.extend([torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU()])
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self._torch = torch
        self.model = torch.nn.Sequential(*layers)
        self.hidden_dims = tuple(hidden_dims)

    def fit(
        self,
        examples: Sequence[PairExample],
        *,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> list[float]:
        if not examples:
            raise ValueError("No training examples")
        labels_present = {example.label for example in examples}
        if labels_present != {0, 1}:
            raise ValueError(
                "Training requires both positive correspondences and hard negatives"
            )
        torch = self._torch
        features = torch.tensor(
            [example.features for example in examples], dtype=torch.float32
        )
        labels = torch.tensor(
            [example.label for example in examples], dtype=torch.float32
        ).reshape(-1, 1)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        positives = max(1, sum(example.label for example in examples))
        negatives = max(1, len(examples) - positives)
        loss_function = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], dtype=torch.float32)
        )
        losses = []
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_function(self.model(features), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return losses

    def score(self, features: Sequence[float]) -> float:
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor([features], dtype=torch.float32)
            return float(torch.sigmoid(self.model(tensor)).item())

    def save(self, path: str | Path) -> None:
        self._torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hidden_dims": self.hidden_dims,
                "feature_names": FEATURE_NAMES,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLPReranker":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("MLPReranker requires PyTorch") from exc
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if tuple(checkpoint["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Checkpoint feature schema is incompatible")
        reranker = cls(hidden_dims=checkpoint["hidden_dims"])
        reranker.model.load_state_dict(checkpoint["state_dict"])
        return reranker


@dataclass(frozen=True)
class RankedPart:
    profile: PartEvidenceProfile
    score: float
    features: tuple[float, ...]


def rerank_candidates(
    query: PartEvidenceProfile,
    candidates: Iterable[PartEvidenceProfile],
    *,
    feature_builder: PairFeatureBuilder,
    scorer: WeightedEvidenceReranker | MLPReranker,
) -> list[RankedPart]:
    ranked = []
    for candidate in candidates:
        features = feature_builder.build(query, candidate)
        ranked.append(RankedPart(candidate, scorer.score(features), features))
    return sorted(ranked, key=lambda item: item.score, reverse=True)
