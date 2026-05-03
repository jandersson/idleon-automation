"""Tests for common.tries_counter."""
import json

from common.tries_counter import read, write


def test_read_returns_none_when_file_missing(tmp_path):
    assert read(tmp_path / "nope.json") is None


def test_round_trip(tmp_path):
    p = tmp_path / "tries.json"
    write(12, p)
    assert read(p) == 12


def test_overwrite(tmp_path):
    p = tmp_path / "tries.json"
    write(5, p)
    write(3, p)
    assert read(p) == 3


def test_creates_parent_dir(tmp_path):
    p = tmp_path / "data" / "subdir" / "tries.json"
    write(7, p)
    assert read(p) == 7


def test_read_handles_garbage(tmp_path):
    p = tmp_path / "tries.json"
    p.write_text("not json")
    assert read(p) is None


def test_read_handles_missing_key(tmp_path):
    p = tmp_path / "tries.json"
    p.write_text(json.dumps({"something_else": 1}))
    assert read(p) is None
