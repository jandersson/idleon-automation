"""Tests for read_book_checkouts (#44) — the OLA[55] checkout pool.

The read-helpers in common.idleon_save normally need a live LevelDB save,
so this monkeypatches load_save with a synthetic OptionsListAccount to
pin the magic index (55) and the None/short-array handling.
"""
import common.idleon_save as save
from common.idleon_save import (
    read_book_checkouts,
    read_max_book_level,
    _max_book_level_from_data,
    _MAX_BOOK_BASE,
    _OLA_BOOK_CHECKOUTS,
)


def _ola(value, idx=_OLA_BOOK_CHECKOUTS):
    ola = [0] * (idx + 1)
    ola[idx] = value
    return {"OptionsListAccount": ola}


def test_reads_checkout_pool_at_index_55(monkeypatch):
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: _ola(29))
    assert read_book_checkouts() == 29


def test_float_value_is_coerced_to_int(monkeypatch):
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: _ola(29.0))
    assert read_book_checkouts() == 29


def test_none_when_save_missing(monkeypatch):
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: None)
    assert read_book_checkouts() is None


def test_none_when_array_too_short(monkeypatch):
    monkeypatch.setattr(save, "load_save",
                        lambda save_dir=None: {"OptionsListAccount": [0, 1, 2]})
    assert read_book_checkouts() is None


def test_none_when_value_non_numeric(monkeypatch):
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: _ola("oops"))
    assert read_book_checkouts() is None


# --- max book level (#94): base + W3 book merit (Tasks[2][2][2], +2/lvl) -----

def _merit_save(w3_book_levels):
    # Tasks[2] = merit shop; [2] selects World 3; [2] is the 3rd (book) merit.
    return {"Tasks": [None, None, [[], [], [0, 0, w3_book_levels]]]}


def test_max_book_base_only_when_no_merit():
    assert _max_book_level_from_data({}) == _MAX_BOOK_BASE  # 125


def test_max_book_adds_w3_merit_the_127_case():
    # Local-save case: one level of the W3 book merit (+2) → 127.
    assert _max_book_level_from_data(_merit_save(1)) == 127


def test_max_book_caps_merit_at_5_levels():
    assert _max_book_level_from_data(_merit_save(99)) == _MAX_BOOK_BASE + 5 * 2  # 135


def test_max_book_handles_missing_or_short_tasks():
    assert _max_book_level_from_data({"Tasks": None}) == _MAX_BOOK_BASE
    assert _max_book_level_from_data({"Tasks": [0, 0, [[], []]]}) == _MAX_BOOK_BASE


def test_read_max_book_level_none_when_save_missing(monkeypatch):
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: None)
    assert read_max_book_level() is None


def test_read_max_book_level_sums_from_data(monkeypatch):
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: _merit_save(1))
    assert read_max_book_level() == 127
