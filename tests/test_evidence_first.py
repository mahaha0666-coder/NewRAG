from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from mechrag.evidence import (
    BoundingBox,
    EvidenceConflictError,
    EvidenceSource,
    EvidenceStore,
    EvidenceUnit,
    ViewRegion,
    deduplicate_cross_view,
)
from mechrag.drawing_pdf import (
    DRAWING_CONDITIONS,
    DrawingAsset,
    DrawingDocument,
    response_to_evidence,
)
from mechrag.metadata import build_text_evidence_store, parse_metadata_text
from mechrag.pipeline import EvidenceFirstPipeline
from mechrag.providers import (
    DEEPSEEK_ANTHROPIC_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_OPENAI_BASE_URL,
    create_deepseek_openai_client,
    require_deepseek_api_key,
)
from mechrag.questions import QueryCondition, RuleBasedQuestionDecomposer
from mechrag.reranker import (
    FEATURE_NAMES,
    PairExample,
    PairFeatureBuilder,
    PartEvidenceProfile,
    split_pairs_by_family,
)
from mechrag.selection import (
    AnswerEvidenceVerifier,
    Claim,
    EvidenceCoverageSelector,
    build_evidence_prompt,
)


def unit(
    part_id: str,
    feature_id: str,
    properties: dict,
    *,
    feature_type: str = "hole",
    view_id: str = "front",
    embedding=(1.0, 0.0),
) -> EvidenceUnit:
    return EvidenceUnit(
        part_id=part_id,
        feature_id=feature_id,
        feature_type=feature_type,
        cad_face_ids=("F1",),
        view_regions=(
            ViewRegion(
                view_id=view_id,
                image_uri=f"{part_id}_{view_id}.png",
                bbox=BoundingBox(0.1, 0.1, 0.4, 0.4),
                cad_face_ids=("F1",),
                embedding=tuple(embedding),
            ),
        ),
        geometric_properties=properties,
        sources=(EvidenceSource(f"{part_id}.step"),),
        summary=f"{feature_id} {properties}",
        family_id=f"family_{part_id[0]}",
    )


class EvidenceModelTests(unittest.TestCase):
    def test_cross_view_dedup_uses_feature_identity(self):
        front = unit("P1", "H1", {"hole_type": "through"})
        side = unit(
            "P1",
            "H1",
            {"hole_type": "through"},
            view_id="side",
            embedding=(0.9, 0.1),
        )
        merged = deduplicate_cross_view([front, side])
        self.assertEqual(len(merged), 1)
        self.assertEqual({region.view_id for region in merged[0].view_regions}, {"front", "side"})

    def test_cross_view_conflict_is_not_silently_merged(self):
        with self.assertRaises(EvidenceConflictError):
            deduplicate_cross_view(
                [
                    unit("P1", "H1", {"hole_type": "through"}),
                    unit("P1", "H1", {"hole_type": "blind"}, view_id="side"),
                ]
            )

    def test_exact_aggregate_does_not_depend_on_top_k(self):
        store = EvidenceStore(
            [
                unit("P1", "volume", {"volume": 10}, feature_type="geometric_property"),
                unit("P2", "volume", {"volume": 100}, feature_type="geometric_property"),
            ],
            expected_part_ids=("P1", "P2"),
            universe_complete=True,
        )
        bundle = EvidenceFirstPipeline(store).retrieve(
            "Which model has the maximum volume?",
            overall_candidate_part_ids=("P1",),
            evidence_budget=5,
        )
        self.assertTrue(bundle.aggregate.is_complete)
        self.assertEqual(bundle.aggregate.winners[0].part_id, "P2")
        self.assertEqual(bundle.selected_evidence[0].feature_type, "structured_aggregate")

    def test_incomplete_universe_cannot_prove_global_maximum(self):
        store = EvidenceStore(
            [unit("P1", "volume", {"volume": 10})],
            expected_part_ids=("P1", "P2"),
            universe_complete=True,
        )
        result = store.aggregate("volume", "max")
        self.assertFalse(result.is_complete)
        self.assertEqual(result.missing_part_ids, ("P2",))


class QuestionAndSelectionTests(unittest.TestCase):
    def test_question_is_decomposed_into_engineering_conditions(self):
        plan = RuleBasedQuestionDecomposer().decompose(
            "Which model has the maximum volume?"
        )
        self.assertTrue(plan.requires_exhaustive_scan)
        self.assertEqual(plan.aggregate_property, "volume")
        self.assertEqual(plan.aggregate_operation, "max")

    def test_distinct_set_question_uses_exhaustive_aggregation(self):
        plan = RuleBasedQuestionDecomposer().decompose(
            "How many different symmetries are there?"
        )
        self.assertEqual(plan.aggregate_operation, "count_distinct")
        store = EvidenceStore(
            [
                unit("P1", "sym", {"symmetry": "axial"}),
                unit("P2", "sym", {"symmetry": "none"}),
                unit("P3", "sym", {"symmetry": "axial"}),
            ],
            expected_part_ids=("P1", "P2", "P3"),
            universe_complete=True,
        )
        result = store.aggregate("symmetry", "count_distinct")
        derived = result.as_evidence_unit()
        self.assertTrue(result.is_complete)
        self.assertEqual(derived.geometric_properties["result"]["distinct_count"], 2)

    def test_greedy_selector_prefers_complementary_evidence(self):
        conditions = (
            QueryCondition("C1", "hole count", ("hole_count",)),
            QueryCondition("C2", "coaxial relation", ("coaxial",)),
        )
        first = unit("P1", "H1", {"hole_count": 2})
        duplicate = unit("P2", "H1", {"hole_count": 2})
        complementary = unit("P3", "A1", {"coaxial": True})
        result = EvidenceCoverageSelector().select(
            [first, duplicate, complementary], conditions, budget=2
        )
        selected_properties = {
            name
            for evidence in result.evidence
            for name in evidence.geometric_properties
        }
        self.assertEqual(selected_properties, {"hole_count", "coaxial"})
        self.assertEqual(result.report.coverage, 1.0)

    def test_answer_verifier_rejects_missing_or_wrong_evidence(self):
        evidence = (unit("P1", "H1", {"hole_count": 2}),)
        conditions = (QueryCondition("C1", "hole count", ("hole_count",)),)
        verifier = AnswerEvidenceVerifier()
        valid = verifier.verify(
            [Claim("J1", "two holes", ("C1",), ("P1:H1",))],
            evidence,
            conditions,
        )
        invalid = verifier.verify(
            [Claim("J1", "two holes", ("C1",), ("UNKNOWN",))],
            evidence,
            conditions,
        )
        self.assertTrue(valid.valid)
        self.assertFalse(invalid.valid)

    def test_incomplete_aggregate_cannot_support_exhaustive_claim(self):
        condition = QueryCondition(
            "C1", "global maximum volume", ("volume",), exhaustive=True
        )
        incomplete = EvidenceUnit(
            part_id="GLOBAL",
            feature_id="aggregate_max_volume",
            feature_type="structured_aggregate",
            geometric_properties={
                "aggregate_property": "volume",
                "complete": False,
            },
        )
        result = AnswerEvidenceVerifier().verify(
            [Claim("J1", "P1 is globally largest", ("C1",), (incomplete.evidence_id,))],
            (incomplete,),
            (condition,),
        )
        self.assertFalse(result.valid)

    def test_local_record_cannot_support_exhaustive_claim(self):
        condition = QueryCondition(
            "C1", "global maximum volume", ("volume",), exhaustive=True
        )
        local = unit("P1", "volume", {"volume": 100})
        result = AnswerEvidenceVerifier().verify(
            [Claim("J1", "P1 is globally largest", ("C1",), (local.evidence_id,))],
            (local,),
            (condition,),
        )
        self.assertFalse(result.valid)

    def test_prompt_has_an_explicit_citable_evidence_allow_list(self):
        condition = QueryCondition("C1", "hole count", ("hole_count",))
        evidence = unit("P1", "H1", {"hole_count": 2})
        selection = EvidenceCoverageSelector().select(
            [evidence], (condition,), budget=1
        )
        prompt = build_evidence_prompt(
            "How many holes?",
            (condition,),
            selection.evidence,
            selection.report,
        )
        self.assertIn('Citable evidence IDs (the complete allow-list)', prompt)
        self.assertIn('"P1:H1"', prompt)
        self.assertIn('provenance only and must not be cited', prompt)


class RerankerTests(unittest.TestCase):
    def test_local_property_mismatch_penalizes_visual_neighbour(self):
        query = PartEvidenceProfile(
            "P1", "F1", (1.0, 0.0), (unit("P1", "H1", {"hole_count": 1}),)
        )
        candidate = PartEvidenceProfile(
            "P2", "F2", (0.999, 0.001), (unit("P2", "H1", {"hole_count": 2}),)
        )
        features = PairFeatureBuilder(("hole_count",)).build(query, candidate)
        feature_map = dict(zip(FEATURE_NAMES, features))
        self.assertGreater(feature_map["overall_clip_similarity"], 0.99)
        self.assertEqual(feature_map["critical_property_mismatch"], 1.0)

    def test_family_split_prevents_variant_leakage(self):
        vector = (0.0,) * len(FEATURE_NAMES)
        examples = [
            PairExample("A1", "A2", "A", "A", vector, 0),
            PairExample("B1", "B2", "B", "B", vector, 1),
            PairExample("C1", "C2", "C", "C", vector, 0),
            PairExample("A1", "B1", "A", "B", vector, 0),
            PairExample("A1", "C1", "A", "C", vector, 0),
            PairExample("B1", "C1", "B", "C", vector, 0),
        ]
        split = split_pairs_by_family(examples, test_ratio=1 / 3, seed=7)
        self.assertTrue(set(split.train_families).isdisjoint(split.test_families))
        for example in split.train:
            self.assertIn(example.query_family_id, split.train_families)
            self.assertIn(example.candidate_family_id, split.train_families)
        for example in split.test:
            self.assertIn(example.query_family_id, split.test_families)
            self.assertIn(example.candidate_family_id, split.test_families)
        self.assertEqual(len(split.dropped_cross_split), 2)


class MetadataTests(unittest.TestCase):
    def test_pdf_text_is_split_into_small_property_records(self):
        text = """
        The volume of the model is 93000 cubic MM.
        The surface area of the model is 20000 square MM.
        The volume of this model bounding box is 180000 cubic MM.
        Total number of round holes of this model is 2.
        """
        units = parse_metadata_text("model_00219733.pdf", text)
        properties = {
            key: value
            for evidence in units
            for key, value in evidence.geometric_properties.items()
        }
        self.assertEqual(properties["volume"], 93000)
        self.assertEqual(properties["bounding_box_volume"], 180000)
        self.assertEqual(properties["hole_count"], 2)

    def test_record_without_property_remains_in_global_universe(self):
        store = build_text_evidence_store(
            [
                ("model_1.pdf", "The volume of the model is 10 cubic MM."),
                ("model_2.pdf", "This record does not contain a volume."),
            ]
        )
        result = store.aggregate("volume", "max")
        self.assertFalse(result.is_complete)
        self.assertEqual(result.missing_part_ids, ("model_2",))

    def test_real_pdf_sentence_does_not_parse_model_id_as_value(self):
        text = """
        The volume of the model_00210008 is 3258.334 cubic MM.
        The surface area of the model_00210008 is 1460.754 square MM.
        The volume of this model_00210008 bounding box is 4149.070 cubic MM.
        The variance of the Gaussian curvature of this model_00210008 is 0.000.
        The flat plane of this model_00210008 occupies 0.000%.
        This model_00210008 is radially symmetrical.
        The axis of symmetry of this model_00210008 is [-0.000 1.000 -4.5e-18].
        Total number of round holes of this model_00210008 is 1.
        """
        units = parse_metadata_text("model_00210008.pdf", text)
        properties = {
            key: value
            for evidence in units
            for key, value in evidence.geometric_properties.items()
        }
        self.assertEqual(properties["volume"], 3258.334)
        self.assertEqual(properties["surface_area"], 1460.754)
        self.assertEqual(properties["bounding_box_volume"], 4149.07)
        self.assertEqual(properties["gaussian_curvature_variance"], 0)
        self.assertEqual(properties["flat_surface_ratio"], 0)
        self.assertEqual(properties["symmetry"], "radially symmetrical")
        self.assertEqual(properties["symmetry_axis"], "[-0.000 1.000 -4.5e-18]")
        self.assertEqual(properties["hole_count"], 1)


class ProviderTests(unittest.TestCase):
    def test_deepseek_defaults_match_compatible_endpoints(self):
        self.assertEqual(DEEPSEEK_OPENAI_BASE_URL, "https://api.deepseek.com")
        self.assertEqual(
            DEEPSEEK_ANTHROPIC_BASE_URL,
            "https://api.deepseek.com/anthropic",
        )
        self.assertEqual(DEEPSEEK_MODEL, "deepseek-flash")

    def test_deepseek_key_is_only_read_at_runtime(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "  test-key  "}):
            self.assertEqual(require_deepseek_api_key(), "test-key")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            with self.assertRaises(RuntimeError):
                require_deepseek_api_key()

    def test_openai_compatible_client_uses_runtime_key(self):
        with patch("openai.OpenAI") as client_class:
            create_deepseek_openai_client(api_key="test-key")
        client_class.assert_called_once_with(
            api_key="test-key", base_url=DEEPSEEK_OPENAI_BASE_URL
        )


class DrawingPdfTests(unittest.TestCase):
    @staticmethod
    def document() -> DrawingDocument:
        assets = tuple(
            DrawingAsset(
                asset_id=f"drawing_p1_{role}",
                document_id="drawing",
                pdf_path="drawing.pdf",
                page_number=1,
                image_path=f"{role}.jpg",
                full_page_bbox=(0.1, 0.2, 0.9, 0.8),
                role=role,
                width=1000,
                height=800,
            )
            for role in ("overview", "content")
        )
        return DrawingDocument("drawing", "drawing.pdf", 1, 0, False, assets)

    def test_drawing_response_becomes_four_citable_units(self):
        results = []
        feature_ids = (
            "entity_inventory",
            "document_text",
            "drawing_structure",
            "assembly_relations",
        )
        for condition, feature_id in zip(DRAWING_CONDITIONS, feature_ids):
            results.append(
                {
                    "condition_id": condition.condition_id,
                    "feature_id": feature_id,
                    "answer": f"answer {condition.condition_id}",
                    "facts": {"value": condition.condition_id},
                    "evidence_regions": [
                        {
                            "asset_id": "drawing_p1_content",
                            "bbox": [0.25, 0.25, 0.75, 0.75],
                            "note": "test",
                        }
                    ],
                    "confidence": 0.9,
                }
            )
        units, _ = response_to_evidence(
            self.document(), {"document_id": "drawing", "results": results}
        )
        self.assertEqual(len(units), 4)
        for actual, expected in zip(
            units[0].view_regions[0].bbox.as_tuple(), (0.3, 0.35, 0.7, 0.65)
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertIn(
            "entity_identification_and_count", units[0].geometric_properties
        )
        self.assertEqual(units[0].view_regions[0].image_uri, "overview.jpg")

    def test_drawing_response_rejects_missing_condition(self):
        with self.assertRaisesRegex(ValueError, "exactly four"):
            response_to_evidence(
                self.document(), {"document_id": "drawing", "results": []}
            )


if __name__ == "__main__":
    unittest.main()
