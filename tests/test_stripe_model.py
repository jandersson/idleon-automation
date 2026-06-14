"""Tests for minigames.darts.stripe_model (#41 step 3). vy units: px/s."""
import math
import random
from pathlib import Path

from minigames.darts.shot_log import open_db, log_throw, log_poll
from minigames.darts.stripe_model import (
    EV_MARGIN,
    MIN_SAMPLES,
    VALID_INCREMENTS,
    VY_GRID,
    VY_OUTLIER_ABS,
    fetch_stripe_rows,
    fit_stripe_gp,
    model_vy_band,
)


def _log(db, idx, **kw):
    defaults = dict(
        session_started="2026-06-10T00:00:00",
        throw_idx=idx,
        fired_at="2026-06-10T00:00:00",
        release_pose_x=400,
        release_pose_y=316,
        release_conf=0.8,
        wind_x=0.0,
        wind_y=0.0,
        arm_centroid_vy_at_fire=-32,
        hit=1,
        score_increment=3,
    )
    defaults.update(kw)
    log_throw(db, **defaults)


def test_valid_increments_match_stripe_table():
    """The model's score domain must track the game's stripe values."""
    from minigames.darts.main import STRIPE_COLOR_BY_INCREMENT
    assert VALID_INCREMENTS == frozenset(STRIPE_COLOR_BY_INCREMENT.keys())


def test_fetch_rows_filtering(tmp_path: Path):
    db = open_db(tmp_path / "t.db")
    _log(db, 1)                                        # complete hit row
    _log(db, 2, hit=0, score_increment=None)           # miss → score 0
    _log(db, 3, wind_x=None, wind_y=None)              # no wind → dropped
    _log(db, 4, arm_centroid_vy_at_fire=None)          # no vy → dropped
    _log(db, 5, hit=None)                              # no outcome → dropped
    _log(db, 6, score_increment=195)                   # OCR garbage → dropped
    _log(db, 7, score_increment=None)                  # hit, label unknown → dropped
    _log(db, 8, arm_centroid_vy_at_fire=-(VY_OUTLIER_ABS + 1))  # mask glitch → dropped
    rows = fetch_stripe_rows(db)
    db.close()
    assert len(rows) == 2
    scores = sorted(r[4] for r in rows)
    assert scores == [0.0, 3.0]


def test_fetch_converts_legacy_px_per_poll_rows(tmp_path: Path):
    """Pre-2026-06-10 rows logged px/poll in arm_centroid_dy_at_fire.
    They convert to px/s via the fire's actual poll gap from the polls
    table: dy=-8 over a 250ms gap → -32 px/s."""
    db = open_db(tmp_path / "t.db")
    s = "2026-06-09T00:00:00"
    _log(db, 1, session_started=s,
         arm_centroid_vy_at_fire=None, arm_centroid_dy_at_fire=-8)
    log_poll(db, s, t_ms=1000, conf=0.5, match_x=0, match_y=0, threw=0)
    log_poll(db, s, t_ms=1250, conf=0.8, match_x=400, match_y=316, threw=1)
    # A second legacy throw with NO recoverable gap (its fire poll is
    # the session's first poll) → dropped, not guessed.
    s2 = "2026-06-08T00:00:00"
    _log(db, 1, session_started=s2,
         arm_centroid_vy_at_fire=None, arm_centroid_dy_at_fire=-8)
    log_poll(db, s2, t_ms=0, conf=0.8, match_x=400, match_y=316, threw=1)
    rows = fetch_stripe_rows(db)
    db.close()
    assert len(rows) == 1
    assert rows[0][2] == -32.0


def test_fit_returns_none_below_floor():
    rows = [(0.0, 0.0, -32.0, 316.0, 3.0)] * (MIN_SAMPLES - 1)
    assert fit_stripe_gp(rows) is None


def test_fit_recovers_vy_peak_and_wind_penalty():
    """Synthetic surface: EV peaks at vy=-32 px/s, headwind halves it.
    The fitted band should sit on the peak and the calm EV should beat
    the headwind EV at the peak."""
    rng = random.Random(0)
    rows = []
    for _ in range(MIN_SAMPLES + 50):
        wx = rng.choice([-6.0, 0.0, 6.0])
        vy = float(rng.randint(-64, 0))
        ev = 4.0 * math.exp(-((vy + 32.0) ** 2) / 288.0) * (0.5 if wx < 0 else 1.0)
        score = float(rng.choice([0, 1, 2, 3, 5])) if rng.random() < 0.5 else ev
        # Mix deterministic EV with stripe-like noise to keep variance honest.
        rows.append((wx, 0.0, vy, 316.0, max(0.0, min(5.0, score))))
    model = fit_stripe_gp(rows)
    assert model is not None
    band, best_vy, best_ev = model_vy_band(model, 0.0, 0.0, 316.0)
    assert -44 <= best_vy <= -20
    assert band[0] <= best_vy <= band[1]
    head_ev = model.predict(-6.0, 0.0, best_vy, 316.0)
    assert head_ev < best_ev


def test_pose_support_excludes_sparse_tails():
    """A single stray pose_y row (e.g. the historical phantom-match
    fire at pose_y=38) must not extend the trusted interval — the band
    outside data support is mean-reversion junk, observed live as a
    junk band hugging vy=0 on phantom (146, 38) matches."""
    rng = random.Random(1)
    rows = [
        (0.0, 0.0, float(rng.randint(-48, -16)), float(rng.randint(250, 400)), 3.0)
        for _ in range(MIN_SAMPLES + 10)
    ]
    rows.append((0.0, 0.0, -32.0, 38.0, 1.0))  # the phantom row
    model = fit_stripe_gp(rows)
    assert model is not None
    assert not model.supports_pose(38.0)
    assert model.supports_pose(316.0)
    lo, hi = model.pose_support
    assert lo > 150.0  # the stray row didn't drag the interval down
    assert 250.0 - 30 <= lo <= 280.0


def test_model_band_is_contiguous():
    """A disconnected over-threshold bump at the grid edge (GP mean-
    reversion in unexplored territory) must not enter the band."""

    class StubModel:
        def predict_grid(self, wx, wy, pose_y, vys):
            # Peak at vy=-32 (EV 3.0), low flat elsewhere, fake bump at
            # the far edge poking back over the margin threshold.
            out = []
            for vy in vys:
                if vy == min(vys):
                    out.append(2.8)  # disconnected edge bump
                else:
                    out.append(3.0 - min(2.5, abs(vy + 32) * 0.2))
            return out

    band, best_vy, _ = model_vy_band(StubModel(), 0.0, 0.0, 316.0)
    assert best_vy == -32
    assert band[0] > min(VY_GRID)  # edge bump excluded
    assert band[0] <= -32 <= band[1]


def test_band_members_are_within_margin():
    class FlatModel:
        def predict_grid(self, wx, wy, pose_y, vys):
            return [1.0 for _ in vys]

    band, best_vy, best_ev = model_vy_band(FlatModel(), 0.0, 0.0, 316.0)
    # Flat surface: everything is within margin → whole grid fires.
    assert band == (min(VY_GRID), max(VY_GRID))
    assert best_ev == 1.0
    assert EV_MARGIN > 0
