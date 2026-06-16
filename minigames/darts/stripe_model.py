"""E[stripe] model for the darts vy-band aim (#41 step 3).

GP regression over (wind_x, wind_y, vy_at_fire, pose_y) → expected
score increment, where misses contribute 0 and hits contribute their
stripe value (+1/+2/+3/+5). A single regression target deliberately
folds hit probability and stripe steering into one EV surface: a
headwind that cuts hit rate and a vy that lands gray both lower E[score],
and the fire gate should trade them off in one currency.

The vy feature is the arm-centroid vertical velocity in px/SECOND
(cadence-invariant; see arm_motion.centroid_vy_px_s). Historical rows
logged px/poll at the ~250ms compute-bound cadence — fetch converts
them using each fire's actual poll gap from the polls table.

The live gate can't CHOOSE vy — passes arrive as the arm swings — so
the model's job is pass selection: given the current wind, which vy
values are worth firing on? `model_vy_band` scans a vy grid and returns
the near-argmax interval, the wind-conditioned generalization of the
static calm-air band VY_AIM_LO..VY_AIM_HI in main.py.

See docs/predictors.md for the GP refresher; the hoops models in
common/predictor.py established the kernel conventions used here.
"""
from __future__ import annotations

import sqlite3

# Score values a hit can produce (Throwy Darts stripes). Kept in sync
# with main.STRIPE_COLOR_BY_INCREMENT by the unit tests.
VALID_INCREMENTS = frozenset({1, 2, 3, 5})

# Sample floor before the model replaces the static band. ~150 was the
# agreed bar (#41): enough vy spread from the aim-gated sessions and
# wind spread from the backfill to pin a 4D surface.
MIN_SAMPLES = 150

# Fire on any vy whose predicted EV is within this of the grid argmax.
# 0.5 = half a gray stripe; tighter would starve fires on flat surfaces,
# looser would readmit the up-swing-adjacent vys the band exists to skip.
EV_MARGIN = 0.5

# vy grid the band scan covers (px/s): observed at-fire vys live in
# roughly -64..0 (slow release passes cluster at -16..-48). Kept inside
# the data support — the GP mean-reverts in unexplored territory, which
# can fake a second EV peak at the grid edge. Step 4 px/s ≈ the 1
# px/poll resolution the surface was learned at.
VY_GRID = tuple(range(-64, 1, 4))

# Training rows with |vy| beyond this are mask glitches (a centroid
# jump between the arm and some other moving white sprite), not arm
# motion — the recorded swing tops out around 120 px/s.
VY_OUTLIER_ABS = 160

# pose_y support interval: the model's band is only trusted between the
# 5th and 95th percentile of training pose_y (± padding). Outside it
# the GP mean-reverts and the "band" is junk — observed live in the
# first model session (2026-06-10 21:30): the (146, 38) phantom
# template match passed the adaptive conf gate during pose droughts,
# fed pose_y=38 (1 training row within 50px) to the model, and got
# back a junk band hugging vy=0. Callers fall back to the static band
# outside support.
POSE_SUPPORT_PCT = 0.05
POSE_SUPPORT_PAD = 25.0

# Throw-count stripe gating (#51). Low-value stripes stop scoring as a
# game progresses (Throwy Darts; idleon.wiki "Until 10th/25th hit" +
# confirmed against darts.db): gray (+1) becomes a miss once this many
# hits have been made, tan (+2) likewise. The thresholds count cumulative
# HITS in the current game, not throws — across 52 logged games tan's
# last scoring hit-number is exactly 25 (n=207) and gray's is <=9 (n=25),
# which only lines up on the hit axis (throw-count would smear below 25
# once misses are included). Firing on a vy that lands a dead stripe is a
# guaranteed miss (a lost life), so the regime models below re-score
# those stripes to 0 and the band scan walks off the doomed vys.
GRAY_DEAD_AFTER_HITS = 10
TAN_DEAD_AFTER_HITS = 25


class StripeEvModel:
    """Wraps a fitted sklearn GP regressor over
    (wind_x, wind_y, vy, pose_y) → E[score increment]."""

    def __init__(self, rows, gp):
        self.rows = rows
        self._gp = gp
        poses = sorted(r[3] for r in rows)
        lo_i = int(len(poses) * POSE_SUPPORT_PCT)
        hi_i = max(lo_i, int(len(poses) * (1 - POSE_SUPPORT_PCT)) - 1)
        self.pose_support = (
            poses[lo_i] - POSE_SUPPORT_PAD,
            poses[hi_i] + POSE_SUPPORT_PAD,
        )

    def supports_pose(self, pose_y: float) -> bool:
        """Whether the training data covers this pose_y well enough for
        the band to be data-driven rather than prior mean-reversion."""
        return self.pose_support[0] <= pose_y <= self.pose_support[1]

    def predict(self, wind_x: float, wind_y: float, vy: float, pose_y: float) -> float:
        import numpy as np

        X = np.array([[wind_x, wind_y, vy, pose_y]])
        return float(self._gp.predict(X)[0])

    def predict_grid(
        self, wind_x: float, wind_y: float, pose_y: float, vys
    ) -> list[float]:
        import numpy as np

        X = np.array([[wind_x, wind_y, float(vy), pose_y] for vy in vys])
        return [float(v) for v in self._gp.predict(X)]


def _legacy_fire_dts_ms(db: sqlite3.Connection, session: str) -> list[int | None]:
    """Per-fire poll gaps (ms) for one session, in throw_idx order.

    Fire polls (threw=1) in t_ms order correspond 1:1 to the session's
    throws in throw_idx order (validated during the #42 diagnosis). The
    gap to the immediately preceding poll is the dt the legacy px/poll
    dy was measured over.
    """
    cur = db.execute(
        "SELECT t_ms, threw FROM polls WHERE session_started=? ORDER BY t_ms",
        (session,),
    )
    polls = cur.fetchall()
    dts: list[int | None] = []
    for i, (t, threw) in enumerate(polls):
        if threw:
            dts.append(t - polls[i - 1][0] if i > 0 else None)
    return dts


def fetch_stripe_rows(
    db: sqlite3.Connection,
) -> list[tuple[float, float, float, float, float]]:
    """(wind_x, wind_y, vy_px_s, pose_y, score) training rows.

    Misses are score 0. Hits with an OCR'd increment outside
    VALID_INCREMENTS (or NULL) are dropped — the label is somewhere in
    1..5 but unknown, and guessing would bias the stripe axis.

    vy comes from `arm_centroid_vy_at_fire` (px/s, logged from
    2026-06-10 on) when present; legacy rows carry px/poll in
    `arm_centroid_dy_at_fire` and are converted via their fire's actual
    poll gap from the polls table (skipped when no gap is recoverable).
    """
    cur = db.execute(
        """
        SELECT session_started, throw_idx, wind_x, wind_y,
               arm_centroid_vy_at_fire, arm_centroid_dy_at_fire,
               release_pose_y, hit, score_increment
        FROM throws
        WHERE wind_x IS NOT NULL AND wind_y IS NOT NULL
          AND (arm_centroid_vy_at_fire IS NOT NULL
               OR arm_centroid_dy_at_fire IS NOT NULL)
          AND release_pose_y IS NOT NULL
          AND hit IS NOT NULL
        ORDER BY session_started, throw_idx
        """
    )
    raw = cur.fetchall()
    # Throw position within its session (for the legacy dt lookup):
    # _legacy_fire_dts_ms indexes fires by order, and throws are
    # ordered by throw_idx — but throw_idx is 1-based per session and
    # some throws may be filtered out of `raw` by the WHERE clause, so
    # use throw_idx - 1 as the fire-list position.
    dts_by_session: dict[str, list[int | None]] = {}
    rows = []
    for session, throw_idx, wind_x, wind_y, vy, dy, pose_y, hit, increment in raw:
        if vy is None:
            if session not in dts_by_session:
                dts_by_session[session] = _legacy_fire_dts_ms(db, session)
            dts = dts_by_session[session]
            pos = throw_idx - 1
            dt = dts[pos] if 0 <= pos < len(dts) else None
            if dt is None or not 0 < dt <= 800:
                continue
            vy = dy * 1000.0 / dt
        if abs(vy) > VY_OUTLIER_ABS:
            continue
        if hit:
            if increment not in VALID_INCREMENTS:
                continue
            score = float(increment)
        else:
            score = 0.0
        rows.append((float(wind_x), float(wind_y), float(vy), float(pose_y), score))
    return rows


def fit_stripe_gp(
    rows: list[tuple[float, float, float, float, float]],
    min_samples: int = MIN_SAMPLES,
    *,
    kernel=None,
    optimize: bool = True,
) -> StripeEvModel | None:
    """Fit the EV surface. Returns None below the sample floor.

    Anisotropic RBF length scales per feature: wind components are in
    mph (-10..+10, scale 3), vy in px/s (-64..0, scale 16 — the band
    itself is ~12 wide, so curvature at that scale must be
    expressible), pose_y in window px (scale 50, matching the hoops
    convention for sprite-position dims). WhiteKernel absorbs the
    outcome noise — the target is 0-or-stripe, so per-point variance
    is large by construction.

    `kernel`/`optimize` support the warm-started regime fits (#51): pass
    an already-fitted kernel with optimize=False to skip hyperparameter
    search and just recompute the posterior for a relabelled target. A
    dead-stripe surface differs from the full one only in a handful of
    zeroed points, so its smoothness is the full model's — the warm fit
    is ~0.01s vs ~5s for a fresh optimisation, so the gate costs nothing
    extra at startup.
    """
    if len(rows) < min_samples:
        return None
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    X = np.array([[r[0], r[1], r[2], r[3]] for r in rows])
    y = np.array([r[4] for r in rows])
    if kernel is None:
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e3))
            * RBF(
                length_scale=[3.0, 3.0, 16.0, 50.0],
                length_scale_bounds=(0.5, 1e3),
            )
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-2, 1e2))
        )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=4 if optimize else 0,
        optimizer="fmin_l_bfgs_b" if optimize else None,
        random_state=0,
    )
    gp.fit(X, y)
    return StripeEvModel(rows, gp)


def model_vy_band(
    model: StripeEvModel,
    wind_x: float,
    wind_y: float,
    pose_y: float,
    vys=VY_GRID,
    margin: float = EV_MARGIN,
) -> tuple[tuple[int, int], int, float]:
    """The wind-conditioned fire band: the CONTIGUOUS run of grid vys
    around the argmax whose EV stays within `margin` of it, returned as
    an inclusive (lo, hi) interval — observed vys land between grid
    points, so membership is a range check, not set membership.
    Contiguity matters: when the surface is low and flat (strong
    headwind), the GP's mean-reversion at the grid edges can poke back
    over the threshold, and a disconnected band member there would
    admit fires on vys no throw has ever sampled.
    Returns ((lo, hi), best_vy, best_ev)."""
    evs = model.predict_grid(wind_x, wind_y, pose_y, vys)
    best_i = max(range(len(vys)), key=lambda i: evs[i])
    floor = evs[best_i] - margin
    lo_i = best_i
    while lo_i > 0 and evs[lo_i - 1] >= floor:
        lo_i -= 1
    hi_i = best_i
    while hi_i < len(vys) - 1 and evs[hi_i + 1] >= floor:
        hi_i += 1
    return (int(vys[lo_i]), int(vys[hi_i])), int(vys[best_i]), evs[best_i]


def relabel_for_hits(
    rows: list[tuple[float, float, float, float, float]],
    hits_made: int,
) -> list[tuple[float, float, float, float, float]]:
    """Re-score training rows for the dead-stripe set at `hits_made` (#51).

    A row's score VALUE uniquely identifies its stripe (1=gray, 2=tan,
    3=green, 5=red, 0=miss — fetch_stripe_rows maps the OCR'd increment
    straight to score), so the relabel is a pure value remap: gray (1) ->
    0 once hits_made >= GRAY_DEAD_AFTER_HITS, tan (2) -> 0 once hits_made
    >= TAN_DEAD_AFTER_HITS. Green/red/miss are untouched. Returns the
    input list unchanged when nothing is dead yet (hits < 10)."""
    gray_dead = hits_made >= GRAY_DEAD_AFTER_HITS
    tan_dead = hits_made >= TAN_DEAD_AFTER_HITS
    if not gray_dead and not tan_dead:
        return rows
    out = []
    for wx, wy, vy, py, score in rows:
        if gray_dead and score == 1.0:
            score = 0.0
        elif tan_dead and score == 2.0:
            score = 0.0
        out.append((wx, wy, vy, py, score))
    return out


def regime_for_hits(hits_made: int) -> str:
    """Human label for the active throw-count regime (#51), for logging."""
    if hits_made >= TAN_DEAD_AFTER_HITS:
        return "gray+tan dead"
    if hits_made >= GRAY_DEAD_AFTER_HITS:
        return "gray dead"
    return "full"


class RegimeStripeModel:
    """Per-regime E[stripe] surfaces for the throw-count gate (#51).

    Gray (+1) stops scoring after the 10th hit and tan (+2) after the
    25th. Once a stripe is dead, firing on a vy that lands it is a
    guaranteed miss (a lost life). Re-scoring those stripes to 0 and
    re-fitting the EV surface makes the band scan walk off the doomed vys
    toward green/red on its own — the issue's "feed throw count into the
    GP" option, done as three surfaces selected by cumulative hits:
      full       — hits 0-9:   gray + tan both score
      gray_dead  — hits 10-24: gray worth 0
      all_dead   — hits 25+:   gray + tan both worth 0

    The three share one training set (relabelling only touches the score
    target, not pose_y), so pose support is identical — supports_pose
    delegates to the full model.
    """

    def __init__(
        self,
        full: StripeEvModel,
        gray_dead: StripeEvModel | None,
        all_dead: StripeEvModel | None,
    ):
        self.full = full
        self.gray_dead = gray_dead
        self.all_dead = all_dead

    def for_hits(self, hits_made: int) -> StripeEvModel:
        """The EV sub-model whose dead-stripe set matches this hit count.
        Falls back to the full surface if a warm regime fit is missing
        (can't happen for in-floor data — relabelling drops no rows — but
        keeps the live path total)."""
        if hits_made >= TAN_DEAD_AFTER_HITS:
            return self.all_dead or self.full
        if hits_made >= GRAY_DEAD_AFTER_HITS:
            return self.gray_dead or self.full
        return self.full

    def supports_pose(self, pose_y: float) -> bool:
        return self.full.supports_pose(pose_y)


def fit_regime_models(
    rows: list[tuple[float, float, float, float, float]],
    min_samples: int = MIN_SAMPLES,
) -> RegimeStripeModel | None:
    """Fit the three throw-count regimes (#51). Returns None below the
    sample floor (caller falls back to the static vy band).

    The full surface is optimised normally; the two dead-stripe surfaces
    reuse the full model's fitted kernel and skip hyperparameter search
    (optimize=False), so the gate adds ~0.02s rather than ~11s to startup
    (see fit_stripe_gp)."""
    full = fit_stripe_gp(rows, min_samples)
    if full is None:
        return None
    kernel = full._gp.kernel_
    gray_dead = fit_stripe_gp(
        relabel_for_hits(rows, GRAY_DEAD_AFTER_HITS),
        min_samples, kernel=kernel, optimize=False,
    )
    all_dead = fit_stripe_gp(
        relabel_for_hits(rows, TAN_DEAD_AFTER_HITS),
        min_samples, kernel=kernel, optimize=False,
    )
    return RegimeStripeModel(full, gray_dead, all_dead)
