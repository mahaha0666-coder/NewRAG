import json
from pathlib import Path

from PIL import Image

from mechrag.mechvqa_benchmark import _extract_json, prepare_views
from mechrag.mechvqa_report import wilson_interval


def test_extract_json_accepts_fenced_payload():
    payload = _extract_json('```json\n{"answer":"42","claims":[]}\n```')
    assert payload["answer"] == "42"


def test_prepare_views_creates_overview_and_four_overlapping_tiles(tmp_path: Path):
    source = tmp_path / "drawing.png"
    Image.new("RGB", (1000, 600), "white").save(source)
    views = prepare_views(source, tmp_path / "views", max_side=256)
    assert len(views) == 5
    assert {view.view_id for view in views} == {
        "overview",
        "tile_top_left",
        "tile_top_right",
        "tile_bottom_left",
        "tile_bottom_right",
    }
    assert all(Path(view.path).is_file() for view in views)
    assert views[1].bbox[2] > views[2].bbox[0]


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(7, 8)
    assert low < 7 / 8 < high
