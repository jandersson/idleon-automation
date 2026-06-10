"""E[stripe] model for the darts dy-band aim (#41 step 3).

GP regression over (wind_x, wind_y, dy_at_fire, pose_y) → expected
score increment, where misses contribute 0 and hits contribute their
stripe value (+1/+2/+3/+5). A single regression target deliberately
folds hit probability and stripe steering into one EV surface: a
headwind that cuts hit rate and a dy that lands gray both lower E[score],
and the fire gate should trade them off in one currency.

The live gate can't CHOOSE dy — passes arrive as the arm swings — so
the model's job is pass selection: given the current wind, which dy
values are worth firing on? `model_dy_band` scans a dy grid and returns
the near-argmax set, the wind-conditioned generalization of the static
calm-air band DY_AIM_LO..DY_AIM_HI in main.py.

See docs/predictors.md for the GP refresher; the hoops models in
common/predictor.py established the kernel conventions used here.
"""
from __future__ import annotations

import sqlite3

# Score values a hit can produce (Throwy Darts stripes). Kept in sync
# with main.STRIPE_COLOR_BY_INCREMENT by the unit tests.
VALID_INCREMENTS = frozenset({1, 2, 3, 5})

# Sample floor before the model replaces the static band. ~150 was the
# agreed bar (#41): enough dy spread from the aim-gated sessions and
# wind spread from the backfill to pin a 4D surface.
MIN_SAMPLES = 150

# Fire on any dy whose predicted EV is within this of the grid argmax.
# 0.5 = half a gray stripe; tighter would starve fires on flat surfaces,
# looser would readmit the up-swing-adjacent dys the band exists to skip.
EV_MARGIN = 0.5

# dy grid the band scan covers: observed at-fire dys live in roughly
# -16..0 (slow release passes cluster at -4..-12). Kept inside the
# data support — the GP mean-reverts in unexplored territory, which
# can fake a second EV peak at the grid edge.
DY_GRID = tuple(range(-16, 1))

# Training rows with |dy| beyond this are mask glitches (a centroid
# jump between the arm and some other moving white sprite), not arm
# motion — the recorded swing tops out around 30 px/poll.
DY_OUTLIER_ABS = 40


class StripeEvModel:
    """Wraps a fitted sklearn GP regressor over
    (wind_x, wind_y, dy, pose_y) → E[score increment]."""

    def __init__(self, rows, gp):
        self.rows = rows
        self._gp = gp

    def predict(self, wind_x: float, wind_y: float, dy: float, pose_y: float) -> float:
        import numpy as np

        X = np.array([[wind_x, wind_y, dy, pose_y]])
        return float(self._gp.predict(X)[0])

    def predict_grid(
        self, wind_x: float, wind_y: float, pose_y: float, dys
    ) -> list[float]:
        import numpy as np

        X = np.array([[wind_x, wind_y, float(dy), pose_y] for dy in dys])
        return [float(v) for v in self._gp.predict(X)]


def fetch_stripe_rows(
    db: sqlite3.Connection,
) -> list[tuple[float, float, float, float, float]]:
    """(wind_x, wind_y, dy, pose_y, score) training rows.

    Misses are score 0. Hits with an OCR'd increment outside
    VALID_INCREMENTS (or NULL) are dropped — the label is somewhere in
    1..5 but unknown, and guessing would bias the stripe axis.
    """
    cur = db.execute(
        """
        SELECT wind_x, wind_y, arm_centroid_dy_at_fire, release_pose_y,
               hit, score_increment
        FROM throws
        WHERE wind_x IS NOT NULL AND wind_y IS NOT NULL
          AND arm_centroid_dy_at_fire IS NOT NULL
          AND release_pose_y IS NOT NULL
          AND hit IS NOT NULL
        """
    )
    rows = []
    for wind_x, wind_y, dy, pose_y, hit, increment in cur.fetchall():
        if abs(dy) > DY_OUTLIER_ABS:
            continue
        if hit:
            if increment not in VALID_INCREMENTS:
                continue
            score = float(increment)
        else:
            score = 0.0
        rows.append((float(wind_x), float(wind_y), float(dy), float(pose_y), score))
    return rows


def fit_stripe_gp(
    rows: list[tuple[float, float, float, float, float]],
    min_samples: int = MIN_SAMPLES,
) -> StripeEvModel | None:
    """Fit the EV surface. Returns None below the sample floor.

    Anisotropic RBF length scales per feature: wind components are in
    mph (-10..+10, scale 3), dy in px/poll (-20..0, scale 4 — the band
    itself is 4 wide, so curvature at that scale must be expressible),
    pose_y in window px (scale 50, matching the hoops convention for
    sprite-position dims). WhiteKernel absorbs the outcome noise — the
    target is 0-or-stripe, so per-point variance is large by
    construction.
    """
    if len(rows) < min_samples:
        return None
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    X = np.array([[r[0], r[1], r[2], r[3]] for r in rows])
    y = np.array([r[4] for r in rows])
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(
            length_scale=[3.0, 3.0, 4.0, 50.0],
            length_scale_bounds=(0.5, 1e3),
        )
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-2, 1e2))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=4,
        random_state=0,
    )
    gp.fit(X, y)
    return StripeEvModel(rows, gp)


def model_dy_band(
    model: StripeEvModel,
    wind_x: float,
    wind_y: float,
    pose_y: float,
    dys=DY_GRID,
    margin: float = EV_MARGIN,
) -> tuple[set[int], int, float]:
    """The wind-conditioned fire band: the CONTIGUOUS run of dys around
    the grid argmax whose EV stays within `margin` of it. Contiguity
    matters: when the surface is low and flat (strong headwind), the
    GP's mean-reversion at the grid edges can poke back over the
    threshold, and a disconnected band member there would admit fires
    on dys no throw has ever sampled. Returns (band, best_dy, best_ev)."""
    evs = model.predict_grid(wind_x, wind_y, pose_y, dys)
    best_i = max(range(len(dys)), key=lambda i: evs[i])
    floor = evs[best_i] - margin
    band = {int(dys[best_i])}
    for i in range(best_i - 1, -1, -1):
        if evs[i] < floor:
            break
        band.add(int(dys[i]))
    for i in range(best_i + 1, len(dys)):
        if evs[i] < floor:
            break
        band.add(int(dys[i]))
    return band, int(dys[best_i]), evs[best_i]
