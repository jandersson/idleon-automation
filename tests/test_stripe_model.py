"""Tests for minigames.darts.stripe_model (#41 step 3). vy units: px/s."""
import math
import random
from pathlib import Path

from minigames.darts.shot_log import open_db, log_throw, log_poll
from minigames.darts.stripe_model import (
    EV_MARGIN,
    GRAY_DEAD_AFTER_HITS,
    MIN_SAMPLES,
    RegimeStripeModel,
    TAN_DEAD_AFTER_HITS,
    VALID_INCREMENTS,
    VY_GRID,
    VY_OUTLIER_ABS,
    fetch_stripe_rows,
    fit_regime_models,
    fit_stripe_gp,
    model_vy_band,
    regime_for_hits,
    relabel_for_hits,
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


# ---- Throw-count stripe gate (#51) -----------------------------------


def test_gate_thresholds_match_wiki():
    """idleon.wiki Throwy Darts ("Until 10th/25th hit") + darts.db
    confirmation. A regression guard: bumping these silently would
    de-gate (or over-gate) the late-game model."""
    assert GRAY_DEAD_AFTER_HITS == 10
    assert TAN_DEAD_AFTER_HITS == 25
    assert regime_for_hits(0) == "full"
    assert regime_for_hits(9) == "full"
    assert regime_for_hits(10) == "gray dead"
    assert regime_for_hits(24) == "gray dead"
    assert regime_for_hits(25) == "gray+tan dead"


def test_relabel_for_hits_zeroes_dead_stripes():
    """Gray (score 1) dies at hit 10, tan (score 2) at hit 25; green/red/
    miss are untouched, and the input list is never mutated in place."""
    base = [
        (0.0, 0.0, -20.0, 316.0, 0.0),  # miss
        (0.0, 0.0, -56.0, 316.0, 1.0),  # gray
        (0.0, 0.0, -12.0, 316.0, 2.0),  # tan
        (0.0, 0.0, -36.0, 316.0, 3.0),  # green
        (0.0, 0.0, -40.0, 316.0, 5.0),  # red
    ]
    # Nothing dead before hit 10 → identity (same object, no copy).
    assert relabel_for_hits(base, 9) is base
    # Gray dead 10..24, tan still alive.
    assert [r[4] for r in relabel_for_hits(base, 10)] == [0.0, 0.0, 2.0, 3.0, 5.0]
    assert [r[4] for r in relabel_for_hits(base, 24)] == [0.0, 0.0, 2.0, 3.0, 5.0]
    # Gray + tan both dead from hit 25.
    assert [r[4] for r in relabel_for_hits(base, 25)] == [0.0, 0.0, 0.0, 3.0, 5.0]
    # Original rows untouched.
    assert [r[4] for r in base] == [0.0, 1.0, 2.0, 3.0, 5.0]


def test_regime_model_routes_by_hit_count():
    """for_hits picks the surface matching the dead-stripe set, and falls
    back to the full surface when a warm regime fit is missing."""
    full, gray_dead, all_dead = object(), object(), object()
    m = RegimeStripeModel(full, gray_dead, all_dead)
    assert m.for_hits(0) is full
    assert m.for_hits(9) is full
    assert m.for_hits(10) is gray_dead
    assert m.for_hits(24) is gray_dead
    assert m.for_hits(25) is all_dead
    assert m.for_hits(40) is all_dead
    # Missing warm fits → full (keeps the live aim path total).
    assert RegimeStripeModel(full, None, None).for_hits(10) is full
    assert RegimeStripeModel(full, None, None).for_hits(25) is full


def test_regime_fit_collapses_dead_stripe_ev():
    """End to end: relabel -> warm refit -> predict. Three well-separated
    vy clusters each land one stripe; zeroing a stripe must drop its vy
    region's EV (the band scan then walks off it) while leaving live
    stripes' EV intact. ~15% misses give the WhiteKernel real noise."""
    rng = random.Random(2)
    rows = []
    for _ in range(70):
        for vy, stripe in ((-56.0, 1.0), (-36.0, 3.0), (-12.0, 2.0)):
            score = 0.0 if rng.random() < 0.15 else stripe
            rows.append((0.0, 0.0, vy + rng.uniform(-2, 2), 316.0, score))
    m = fit_regime_models(rows)
    assert m is not None
    full, gray_dead, all_dead = m.full, m.gray_dead, m.all_dead
    # Routing identity (same objects the EV checks use).
    assert m.for_hits(9) is full
    assert m.for_hits(10) is gray_dead
    assert m.for_hits(25) is all_dead

    def ev(model, vy):
        return model.predict(0.0, 0.0, vy, 316.0)

    GRAY_VY, GREEN_VY, TAN_VY = -56.0, -36.0, -12.0
    # hits>=10: gray region collapses; tan and green hold.
    assert ev(gray_dead, GRAY_VY) < ev(full, GRAY_VY) - 0.4
    assert abs(ev(gray_dead, TAN_VY) - ev(full, TAN_VY)) < 0.4
    assert abs(ev(gray_dead, GREEN_VY) - ev(full, GREEN_VY)) < 0.4
    # hits>=25: tan region also collapses; green still holds.
    assert ev(all_dead, TAN_VY) < ev(full, TAN_VY) - 0.4
    assert ev(all_dead, GRAY_VY) < ev(full, GRAY_VY) - 0.4
    assert abs(ev(all_dead, GREEN_VY) - ev(full, GREEN_VY)) < 0.4
