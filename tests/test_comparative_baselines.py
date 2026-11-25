from mechrag.comparative_baselines import (
    BASELINES,
    _drawing_key,
    _int_or_none,
    _score_self,
)


def test_eight_baseline_systems_are_unique():
    assert len(BASELINES) == 8
    assert len({config.name for config in BASELINES}) == 8
    assert {config.model for config in BASELINES} == {"deepseek-flash"}
    assert {config.view_mode for config in BASELINES} == {"global", "multiview"}
    assert {config.prompt_mode for config in BASELINES} == {
        "direct",
        "rationale",
        "decompose",
        "verify",
    }


def test_objective_field_normalization():
    assert _drawing_key(" NTG.T.5008＃103 ") == "NTG.T.5008#103"
    assert _int_or_none("共 23 件") == 23
    assert _int_or_none(None) is None


def test_self_scoring_ignores_unlabelled_fields():
    rows = [
        {
            "document_id": "d1",
            "responses": {
                "flash-global-direct": {
                    "prediction": {
                        "drawing_no": "A#1",
                        "bom_line_items": 2,
                        "bom_quantity_sum": 3,
                    }
                }
            },
        }
    ]
    truth = {
        "d1": {
            "drawing_no": "A#1",
            "bom_line_items": 2,
            "bom_quantity_sum": None,
        }
    }
    result = _score_self(rows, truth)["flash-global-direct"]
    assert result["count"] == 2
    assert result["correct"] == 2
    assert result["accuracy"] == 1.0
