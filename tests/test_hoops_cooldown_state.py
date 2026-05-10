"""Tests for the local hoops cooldown cache file."""
import json
import time

import pytest

from common import hoops_cooldown_state


def test_read_returns_none_when_file_missing(tmp_path):
    assert hoops_cooldown_state.read(tmp_path / "missing.json") is None


def test_write_then_read_roundtrip(tmp_path):
    path = tmp_path / "cd.json"
    before = time.time()
    hoops_cooldown_state.write_set_now(path, duration_seconds=300.0)
    after = time.time()
    state = hoops_cooldown_state.read(path)
    assert state is not None
    assert before <= state["set_at"] <= after
    assert state["duration_seconds"] == 300.0


def test_remaining_seconds_after_immediate_write_is_close_to_duration(tmp_path):
    path = tmp_path / "cd.json"
    hoops_cooldown_state.write_set_now(path, duration_seconds=120.0)
    remaining = hoops_cooldown_state.remaining_seconds(path)
    assert remaining is not None
    # Tight upper bound — write and read should be near-instant.
    assert 119.0 < remaining <= 120.0


def test_read_returns_none_on_malformed_file(tmp_path):
    path = tmp_path / "cd.json"
    path.write_text("not json", encoding="utf-8")
    assert hoops_cooldown_state.read(path) is None


def test_read_returns_none_when_required_keys_missing(tmp_path):
    path = tmp_path / "cd.json"
    path.write_text(json.dumps({"set_at": "now"}), encoding="utf-8")
    assert hoops_cooldown_state.read(path) is None
