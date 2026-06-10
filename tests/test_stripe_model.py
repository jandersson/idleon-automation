"""Tests for minigames.darts.stripe_model (#41 step 3)."""
import math
import random
from pathlib import Path

from minigames.darts.shot_log import open_db, log_throw
from minigames.darts.stripe_model import (
    DY_GRID,
    DY_OUTLIER_ABS,
    EV_MARGIN,
    MIN_SAMPLES,
    VALID_INCREMENTS,
    fetch_stripe_rows,
    fit_stripe_gp,
    model_dy_band,
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
        arm_centroid_dy_at_fire=-8,
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
    _log(db, 4, arm_centroid_dy_at_fire=None)          # no dy → dropped
    _log(db, 5, hit=None)                              # no outcome → dropped
    _log(db, 6, score_increment=195)                   # OCR garbage → dropped
    _log(db, 7, score_increment=None)                  # hit, label unknown → dropped
    _log(db, 8, arm_centroid_dy_at_fire=-(DY_OUTLIER_ABS + 1))  # mask glitch → dropped
    rows = fetch_stripe_rows(db)
    db.close()
    assert len(rows) == 2
    scores = sorted(r[4] for r in rows)
    assert scores == [0.0, 3.0]


def test_fit_returns_none_below_floor():
    rows = [(0.0, 0.0, -8.0, 316.0, 3.0)] * (MIN_SAMPLES - 1)
    assert fit_stripe_gp(rows) is None


def test_fit_recovers_dy_peak_and_wind_penalty():
    """Synthetic surface: EV peaks at dy=-8, headwind halves it. The
    fitted band should sit on the peak and the calm EV should beat the
    headwind EV at the peak."""
    rng = random.Random(0)
    rows = []
    for _ in range(MIN_SAMPLES + 50):
        wx = rng.choice([-6.0, 0.0, 6.0])
        dy = float(rng.randint(-16, 0))
        ev = 4.0 * math.exp(-((dy + 8.0) ** 2) / 18.0) * (0.5 if wx < 0 else 1.0)
        score = float(rng.choice([0, 1, 2, 3, 5])) if rng.random() < 0.5 else ev
        # Mix deterministic EV with stripe-like noise to keep variance honest.
        rows.append((wx, 0.0, dy, 316.0, max(0.0, min(5.0, score))))
    model = fit_stripe_gp(rows)
    assert model is not None
    band, best_dy, best_ev = model_dy_band(model, 0.0, 0.0, 316.0)
    assert -11 <= best_dy <= -5
    assert best_dy in band
    head_ev = model.predict(-6.0, 0.0, best_dy, 316.0)
    assert head_ev < best_ev


def test_model_band_is_contiguous():
    """A disconnected over-threshold bump at the grid edge (GP mean-
    reversion in unexplored territory) must not enter the band."""

    class StubModel:
        def predict_grid(self, wx, wy, pose_y, dys):
            # Peak at dy=-8 (EV 3.0), low flat elsewhere, fake bump at
            # the far edge poking back over the margin threshold.
            out = []
            for dy in dys:
                if dy == min(dys):
                    out.append(2.8)  # disconnected edge bump
                else:
                    out.append(3.0 - min(2.5, abs(dy + 8) * 0.8))
            return out

    band, best_dy, _ = model_dy_band(StubModel(), 0.0, 0.0, 316.0)
    assert best_dy == -8
    assert min(DY_GRID) not in band
    # Band is a contiguous integer run.
    assert sorted(band) == list(range(min(band), max(band) + 1))


def test_band_members_are_within_margin():
    class FlatModel:
        def predict_grid(self, wx, wy, pose_y, dys):
            return [1.0 for _ in dys]

    band, best_dy, best_ev = model_dy_band(FlatModel(), 0.0, 0.0, 316.0)
    # Flat surface: everything is within margin → whole grid fires.
    assert band == set(DY_GRID)
    assert best_ev == 1.0
    assert EV_MARGIN > 0
