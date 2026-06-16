"""Tests for read_book_checkouts (#44) — the OLA[55] checkout pool.

The read-helpers in common.idleon_save normally need a live LevelDB save,
so this monkeypatches load_save with a synthetic OptionsListAccount to
pin the magic index (55) and the None/short-array handling.
"""
import common.idleon_save as save
from common.idleon_save import read_book_checkouts, _OLA_BOOK_CHECKOUTS


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
