"""Tests for common/regions.py — fractional region storage & retrieval."""
import json

from common.regions import get_region, load_regions, regions_path, save_region


def test_save_and_get_round_trip(tmp_path):
    save_region(tmp_path, "score", {"left": 100, "top": 50, "width": 80, "height": 20}, win_w=1000, win_h=500)

    region = get_region(tmp_path, "score", win_w=1000, win_h=500)
    assert region == {"left": 100, "top": 50, "width": 80, "height": 20}


def test_save_writes_fractions(tmp_path):
    save_region(tmp_path, "bar", {"left": 200, "top": 100, "width": 400, "height": 40}, win_w=1000, win_h=500)

    raw = json.loads(regions_path(tmp_path).read_text())
    assert raw["bar"]["left_frac"] == 0.2
    assert raw["bar"]["top_frac"] == 0.2
    assert raw["bar"]["width_frac"] == 0.4
    assert raw["bar"]["height_frac"] == 0.08


def test_get_scales_to_new_window_size(tmp_path):
    save_region(tmp_path, "score", {"left": 100, "top": 50, "width": 80, "height": 20}, win_w=1000, win_h=500)

    # Same fractions, different window → scaled pixel coords.
    region = get_region(tmp_path, "score", win_w=2000, win_h=1000)
    assert region == {"left": 200, "top": 100, "width": 160, "height": 40}


def test_get_returns_none_for_missing_name(tmp_path):
    save_region(tmp_path, "bar", {"left": 0, "top": 0, "width": 10, "height": 10}, win_w=100, win_h=100)

    assert get_region(tmp_path, "score", win_w=100, win_h=100) is None


def test_get_returns_none_when_no_file(tmp_path):
    assert get_region(tmp_path, "score", win_w=100, win_h=100) is None


def test_load_returns_empty_when_no_file(tmp_path):
    assert load_regions(tmp_path) == {}


def test_legacy_pixel_format_still_works(tmp_path):
    """Pre-refactor regions used pixel coords directly. Reading them must
    still return the same coords (treated as absolute)."""
    p = regions_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"score": {"left": 6, "top": 384, "width": 78, "height": 18}}))

    region = get_region(tmp_path, "score", win_w=1280, win_h=1392)
    assert region == {"left": 6, "top": 384, "width": 78, "height": 18}


def test_save_preserves_other_regions(tmp_path):
    save_region(tmp_path, "score", {"left": 10, "top": 10, "width": 50, "height": 50}, win_w=100, win_h=100)
    save_region(tmp_path, "wind", {"left": 80, "top": 10, "width": 15, "height": 15}, win_w=100, win_h=100)

    assert get_region(tmp_path, "score", 100, 100) == {"left": 10, "top": 10, "width": 50, "height": 50}
    assert get_region(tmp_path, "wind", 100, 100) == {"left": 80, "top": 10, "width": 15, "height": 15}
