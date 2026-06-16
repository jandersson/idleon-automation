"""Tests for the daily-reset time estimate (#22) in
common.hoops_cooldown_observer.

The estimate reads logged plays-drop boundaries and recovers the
account's daily-reset time-of-day in local time. Timestamps are built
via `datetime(...).timestamp()` and read back with `fromtimestamp`, so
the round-trip is local-timezone-consistent on any host (the asserted
hour/minute match regardless of the runner's TZ).
"""
import json
from datetime import datetime
from pathlib import Path

from common.hoops_cooldown_observer import (
    _reset_boundary_midpoints,
    estimate_daily_reset,
    next_daily_reset,
)


def _ts(y, mo, d, h, mi, s=0) -> float:
    return datetime(y, mo, d, h, mi, s).timestamp()


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_boundary_detects_tight_drop_to_zero():
    rows = [
        {"ts": _ts(2026, 6, 10, 1, 50), "streak": 15, "darts_plays_today": 3},
        {"ts": _ts(2026, 6, 10, 1, 52), "streak": 0, "darts_plays_today": 1},
    ]
    mids = _reset_boundary_midpoints(rows)
    assert len(mids) == 1
    lt = datetime.fromtimestamp(mids[0])
    assert (lt.hour, lt.minute) == (1, 51)  # midpoint of 01:50 and 01:52


def test_boundary_ignores_wide_gap_decay_and_plays():
    rows = [
        # wide gap (2h): a reset happened but it can't be localised → drop
        {"ts": _ts(2026, 6, 11, 0, 5), "streak": 19, "darts_plays_today": 18},
        {"ts": _ts(2026, 6, 11, 2, 5), "streak": 1, "darts_plays_today": 0},
        # mid-day decay 4 -> 2: a decrease but not to <=1, so not a reset
        {"ts": _ts(2026, 6, 11, 12, 0), "streak": 4, "darts_plays_today": 5},
        {"ts": _ts(2026, 6, 11, 12, 1), "streak": 2, "darts_plays_today": 5},
        # a play: counter increases, not a drop
        {"ts": _ts(2026, 6, 11, 13, 0), "streak": 2, "darts_plays_today": 5},
        {"ts": _ts(2026, 6, 11, 13, 1), "streak": 3, "darts_plays_today": 5},
    ]
    assert _reset_boundary_midpoints(rows) == []


def test_estimate_recovers_time_of_day(tmp_path):
    p = tmp_path / "obs.jsonl"
    rows = []
    for d in (10, 12, 15):  # three days, reset bracketed 01:51..01:53
        rows.append({"ts": _ts(2026, 6, d, 1, 51), "streak": 12, "darts_plays_today": 4})
        rows.append({"ts": _ts(2026, 6, d, 1, 53), "streak": 0, "darts_plays_today": 0})
    _write(p, rows)
    est = estimate_daily_reset(p)
    assert est is not None
    assert est["samples"] == 3
    assert (est["hour"], est["minute"]) == (1, 52)
    assert est["spread_min"] < 1.0


def test_estimate_circular_mean_handles_midnight_wrap(tmp_path):
    p = tmp_path / "obs.jsonl"
    rows = [
        {"ts": _ts(2026, 6, 10, 23, 58), "streak": 5, "darts_plays_today": 2},
        {"ts": _ts(2026, 6, 11, 0, 0), "streak": 0, "darts_plays_today": 0},   # mid 23:59
        {"ts": _ts(2026, 6, 12, 0, 0), "streak": 5, "darts_plays_today": 2},
        {"ts": _ts(2026, 6, 12, 0, 2), "streak": 0, "darts_plays_today": 0},   # mid 00:01
    ]
    _write(p, rows)
    est = estimate_daily_reset(p)
    assert est is not None and est["samples"] == 2
    # Circular mean of 23:59 and 00:01 is 00:00 — a naive average would
    # land at ~12:00.
    assert (est["hour"], est["minute"]) == (0, 0)


def test_estimate_none_without_observations(tmp_path):
    assert estimate_daily_reset(tmp_path / "missing.jsonl") is None


def test_next_daily_reset_today_then_tomorrow():
    est = {"hour": 1, "minute": 52, "samples": 2, "spread_min": 0.3}
    before = datetime(2026, 6, 17, 0, 30)
    assert next_daily_reset(est, before) == datetime(2026, 6, 17, 1, 52)
    after = datetime(2026, 6, 17, 10, 0)
    assert next_daily_reset(est, after) == datetime(2026, 6, 18, 1, 52)
    assert next_daily_reset(None) is None
