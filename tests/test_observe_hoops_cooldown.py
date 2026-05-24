"""Unit tests for the pure helpers in common/hoops_cooldown_observer.py.

The actual save-watching loop is IO-heavy and not unit-testable without
a fake leveldb; instead we test the data-shaping functions against
synthetic save dicts. We also verify append_observation creates parent
dirs and writes valid JSONL."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from common.hoops_cooldown_observer import (
    _active_char,
    append_observation,
    build_observation,
)


def _make_data(ola: list, players: dict | None = None) -> dict:
    return {
        "OptionsListAccount": ola,
        "PlayerDATABASE": players or {},
    }


def test_build_observation_captures_ola_indices():
    ola = [0] * 600
    ola[423] = -926
    ola[424] = 4
    ola[434] = 20
    ola[435] = 106
    ola[439] = -26
    ola[440] = 2
    ola[442] = 125
    row = build_observation(_make_data(ola), save_mtime_at=1000.0, now=1005.0)
    assert row["cooldown_ticks"] == -926
    assert row["streak"] == 4
    assert row["ola_434"] == 20
    assert row["ola_435"] == 106
    assert row["darts_cooldown_ticks"] == -26
    assert row["darts_plays_today"] == 2
    assert row["darts_cooldown_base"] == 125
    assert row["save_age"] == 5.0
    assert row["save_mtime"] == 1000.0


def test_build_observation_handles_short_ola():
    # OLA truncated before our indices of interest — all logged fields
    # should be None rather than crashing on an IndexError.
    row = build_observation(_make_data([1, 2, 3]), save_mtime_at=500.0, now=510.0)
    assert row["cooldown_ticks"] is None
    assert row["streak"] is None
    assert row["darts_plays_today"] is None


def test_build_observation_handles_missing_ola():
    row = build_observation({}, save_mtime_at=500.0, now=510.0)
    assert row["cooldown_ticks"] is None
    assert row["streak"] is None
    assert row["save_age"] == 10.0


def test_build_observation_zero_mtime_means_unknown_age():
    # save_mtime_at=0 means "couldn't find any leveldb files" — don't
    # report a bogus save_age of (now - 0).
    row = build_observation(_make_data([0] * 500), save_mtime_at=0.0, now=12345.6)
    assert row["save_mtime"] is None
    assert row["save_age"] is None


def test_build_observation_rejects_non_numeric_ola_entries():
    ola: list = [0] * 600
    ola[423] = "weird"  # corrupted / unexpected type
    ola[424] = 7
    row = build_observation(_make_data(ola), save_mtime_at=1000.0, now=1001.0)
    assert row["cooldown_ticks"] is None
    assert row["streak"] == 7


def test_active_char_picks_highest_away_time():
    # PlayerAwayTime is a *timestamp*, not a duration — highest = most
    # recently active. Matches read_minigame_plays_shared's logic.
    data = _make_data(
        ola=[],
        players={
            "OldChar": {"PlayerAwayTime": 100.0},
            "RecentChar": {"PlayerAwayTime": 500.0},
            "AnotherOld": {"PlayerAwayTime": 200.0},
        },
    )
    assert _active_char(data) == "RecentChar"


def test_active_char_handles_missing_or_non_numeric_field():
    data = _make_data(
        ola=[],
        players={
            "NoTime": {},
            "BadType": {"PlayerAwayTime": "not a number"},
            "Real": {"PlayerAwayTime": 50.0},
        },
    )
    assert _active_char(data) == "Real"


def test_active_char_none_when_no_players():
    assert _active_char(_make_data([])) is None


def test_append_observation_creates_parent_dir_and_writes_jsonl(tmp_path):
    out_path = tmp_path / "sub" / "dir" / "cooldown.jsonl"
    append_observation({"a": 1}, out_path)
    append_observation({"a": 2, "b": "x"}, out_path)
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln) for ln in lines] == [{"a": 1}, {"a": 2, "b": "x"}]
