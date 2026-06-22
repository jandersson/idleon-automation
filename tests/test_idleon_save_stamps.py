"""Tests for read_stamps (discovery map: Stamps) — StampLevel decode.

Monkeypatches load_save with a synthetic StampLevel to pin the 3-tab shape,
the Haxe string→int coercion, and the None/short/malformed handling.
"""
import common.idleon_save as save
from common.idleon_save import read_stamps, _STAMP_TABS


def test_decodes_three_tabs_and_coerces_strings(monkeypatch):
    raw = {"StampLevel": [[55, 40, "10"], [85, "5", 0], [30, 0, "4"]]}
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: raw)
    out = read_stamps()
    assert out == {"combat": [55, 40, 10], "skills": [85, 5, 0], "misc": [30, 0, 4]}
    assert list(out.keys()) == list(_STAMP_TABS)


def test_none_when_save_missing(monkeypatch):
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: None)
    assert read_stamps() is None


def test_none_when_stamplevel_missing_or_short(monkeypatch):
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: {})
    assert read_stamps() is None
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: {"StampLevel": [[1], [2]]})
    assert read_stamps() is None


def test_non_list_subtab_becomes_empty(monkeypatch):
    monkeypatch.setattr(save, "load_save",
                        lambda save_dir=None: {"StampLevel": [[1, 2], None, "x"]})
    assert read_stamps() == {"combat": [1, 2], "skills": [], "misc": []}
