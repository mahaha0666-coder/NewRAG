"""Thin helpers for migrating the original notebooks to evidence-first RAG."""

from __future__ import annotations

from dataclasses import dataclass
import base64
from io import BytesIO
from typing import Iterable, Mapping, Sequence

from .evidence import EvidenceStore
from .metadata import build_text_evidence_store
from .pipeline import EvidenceFirstPipeline, RetrievalBundle
from .reranker import (
    MLPReranker,
    PartEvidenceProfile,
    WeightedEvidenceReranker,
)


def prepare_text_bundle(
    question: str,
    *,
    semantic_records: Sequence[tuple[str, str]],
    all_records: Iterable[tuple[str, str]] | None = None,
    evidence_budget: float = 20,
) -> RetrievalBundle:
    """Prepare grounded context for a text query.

    ``all_records`` is required for a verified global max/min/sort.  If it is
    omitted, the aggregate evidence will explicitly be marked incomplete.
    """

    semantic_records = list(semantic_records)
    records = list(all_records) if all_records is not None else semantic_records
    store = build_text_evidence_store(
        records, universe_complete=all_records is not None
    )
    candidate_parts = []
    semantic_store = build_text_evidence_store(semantic_records)
    for unit in semantic_store.units:
        if unit.part_id not in candidate_parts:
            candidate_parts.append(unit.part_id)
    pipeline = EvidenceFirstPipeline(store)
    return pipeline.retrieve(
        question,
        overall_candidate_part_ids=candidate_parts,
        evidence_budget=evidence_budget,
    )


@dataclass
class MultimodalEvidenceRuntime:
    """State plugged into the multimodal notebook after CLIP candidate search."""

    store: EvidenceStore
    profiles: Mapping[str, PartEvidenceProfile]
    reranker: WeightedEvidenceReranker | MLPReranker

    @classmethod
    def load(
        cls,
        evidence_jsonl: str,
        profiles: Mapping[str, PartEvidenceProfile],
        *,
        checkpoint: str | None = None,
    ) -> "MultimodalEvidenceRuntime":
        reranker = (
            MLPReranker.load(checkpoint)
            if checkpoint
            else WeightedEvidenceReranker()
        )
        return cls(EvidenceStore.load_jsonl(evidence_jsonl), profiles, reranker)

    def prepare(
        self,
        question: str,
        *,
        query_profile: PartEvidenceProfile,
        clip_candidate_part_ids: Sequence[str],
        candidate_limit: int = 10,
        evidence_budget: float = 20,
    ) -> RetrievalBundle:
        pipeline = EvidenceFirstPipeline(self.store, reranker=self.reranker)
        return pipeline.retrieve(
            question,
            overall_candidate_part_ids=clip_candidate_part_ids,
            query_profile=query_profile,
            candidate_profiles=self.profiles,
            candidate_limit=candidate_limit,
            evidence_budget=evidence_budget,
        )


def _image_data_url(path: str, bbox=None) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Multimodal message construction requires Pillow") from exc
    with Image.open(path) as image:
        image = image.convert("RGB")
        if bbox is not None:
            width, height = image.size
            image = image.crop(
                (
                    round(bbox.x_min * width),
                    round(bbox.y_min * height),
                    round(bbox.x_max * width),
                    round(bbox.y_max * height),
                )
            )
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_multimodal_message_content(
    bundle: RetrievalBundle, query_image_uri: str
) -> list[dict]:
    """Create GPT-compatible content with target image and selected crops."""

    content = [
        {"type": "text", "text": bundle.prompt},
        {"type": "text", "text": "Target part image:"},
        {
            "type": "image_url",
            "image_url": {"url": _image_data_url(query_image_uri)},
        },
    ]
    emitted_regions = set()
    for unit in bundle.selected_evidence:
        for region in unit.view_regions:
            key = region.identity_key()
            if key in emitted_regions:
                continue
            emitted_regions.add(key)
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Local evidence [E:{unit.evidence_id}], view={region.view_id}, "
                            f"bbox={region.bbox.as_tuple()}, CAD_faces={list(region.cad_face_ids)}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_data_url(region.image_uri, region.bbox)
                        },
                    },
                ]
            )
    return content


def verify_bundle_response(bundle: RetrievalBundle, response: str):
    """Parse the required JSON output and check every claim/evidence edge."""

    pipeline = EvidenceFirstPipeline(EvidenceStore(bundle.selected_evidence))
    return pipeline.verify_response(response, bundle)
