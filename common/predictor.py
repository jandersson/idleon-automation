"""Predictors for "where should the platform be at fire time?".

Most predictors here are fit on a list of past *makes* —
`(hoop_y, hoop_x, platform_y)` tuples — and expose
`predict(hoop_y, hoop_x) -> float` returning the predicted optimal
`platform_y` (i.e. the value the caller should aim `target_y` at).

Four implementations live here so they can be swapped at the call site:

- `KnnPredictor` — inverse-distance-weighted KNN over past makes.
  Adapts to local curvature in the (hoop_y, hoop_x) → optimal_py
  surface; recommended default.
- `BivariatePredictor` — closed-form OLS for
  `target_y = a*hoop_y + b*hoop_x + c`. Older model, kept here for
  A/B comparison and as a baseline.
- `GpPredictor` — Gaussian Process regression with anisotropic RBF
  kernel (sklearn). Returns a posterior mean *and* std; the std is
  what enables variance-driven gating and directional perturbation
  downstream.
- `TrajectoryKnnPredictor` — different shape of model. Fit on *every*
  shot (make or miss) as
  `(hoop_y, hoop_x, platform_y) -> ball_landing_x`, then at predict
  time scans candidate `platform_y` values within the empirical bob
  range and returns the one whose modelled landing is closest to the
  hoop. Lets the predictor learn from misses, not just makes — see
  GitHub issue #17.

`fit_knn` / `fit_bivariate` / `fit_gp` / `fit_trajectory_knn` are
factory helpers that take the rows and return a fitted predictor (or
None if too few samples). The first three take `fetch_makes` rows; the
trajectory one takes `fetch_shots` rows (4-tuples including
`ball_landing_x`).

For algorithm refreshers (KNN, OLS, GP), see `docs/predictors.md` (or
https://jandersson.github.io/idleon-automation/predictors.html for
the same content with rendered LaTeX + the interactive playground
linked from the top). Keep the doc in sync when adding/removing
predictors here.
"""
from typing import Protocol


class Predictor(Protocol):
    """Anything with `n` and `predict(hoop_y, hoop_x) -> float`."""
    n: int
    def predict(self, hoop_y: float, hoop_x: float) -> float: ...


class KnnPredictor:
    """KNN over past makes with inverse-distance weighting."""

    def __init__(self, points: list[tuple[float, float, float]], k: int = 5):
        # points: list of (hoop_y, hoop_x, platform_y) for past makes
        self.points = points
        self.k = min(k, len(points))

    @property
    def n(self) -> int:
        return len(self.points)

    def predict(self, hoop_y: float, hoop_x: float) -> float:
        distances = sorted(
            (((py - hoop_y) ** 2 + (px - hoop_x) ** 2) ** 0.5, target)
            for py, px, target in self.points
        )
        nearest = distances[: self.k]
        # Inverse-distance weighting (clamped at 1.0 to avoid blow-up
        # for exact-match hoops).
        weights = [1.0 / max(d, 1.0) for d, _ in nearest]
        total = sum(weights)
        return sum(w * t for w, (_, t) in zip(weights, nearest)) / total


class BivariatePredictor:
    """Linear plane: target_y = a*hoop_y + b*hoop_x + c, fit via OLS.

    Faster to evaluate than KNN and requires no point storage at predict
    time, but the linear assumption misses curvature in the optimal-py
    surface — which is why it needed hardcoded overrides for
    low-hoop_x regions before we switched to KNN.
    """

    def __init__(
        self,
        points: list[tuple[float, float, float]],
        a: float, b: float, c: float,
    ):
        self.points = points
        self.a = a
        self.b = b
        self.c = c

    @property
    def n(self) -> int:
        return len(self.points)

    def predict(self, hoop_y: float, hoop_x: float) -> float:
        return self.a * hoop_y + self.b * hoop_x + self.c


class GpPredictor:
    """Gaussian Process regression over past makes with an anisotropic
    RBF kernel. Wraps a fitted sklearn `GaussianProcessRegressor`.

    Exposes `predict` (posterior mean only, matching the Predictor
    Protocol) and `predict_with_std` (mean + std), so callers that want
    to gate or perturb based on uncertainty can reach for the std
    without instantiating a separate predictor.
    """

    def __init__(self, points: list[tuple[float, float, float]], gp):
        self.points = points
        self._gp = gp

    @property
    def n(self) -> int:
        return len(self.points)

    def predict(self, hoop_y: float, hoop_x: float) -> float:
        mu, _ = self.predict_with_std(hoop_y, hoop_x)
        return mu

    def predict_with_std(
        self, hoop_y: float, hoop_x: float
    ) -> tuple[float, float]:
        import numpy as np

        x = np.array([[hoop_y, hoop_x]])
        mu, std = self._gp.predict(x, return_std=True)
        return float(mu[0]), float(std[0])


class TrajectoryKnnPredictor:
    """3D KNN over (hoop_y, hoop_x, platform_y) -> ball_landing_x.

    The model maps "what shot was fired" to "where the ball ended up".
    To pick a target_y for a given hoop, scan platform_y across the
    empirical bob range and return the value whose predicted landing
    is closest to hoop_x — i.e. invert the trajectory model.

    Trains on every shot, not just makes, so misses contribute their
    landing_x as direct evidence of "this offset overshoots / undershoots
    at this hoop". Sparse-region behaviour is the same as KNN's: the k
    nearest neighbours dominate.
    """

    def __init__(
        self,
        points: list[tuple[float, float, float, float]],
        k: int = 5,
        py_scan_step: float = 5.0,
    ):
        # points: list of (hoop_y, hoop_x, platform_y, ball_landing_x)
        self.points = points
        self.k = min(k, len(points))
        self._step = py_scan_step
        # Scan range bounded by the empirical platform_y envelope so we
        # never predict for platform_y values we have zero data for.
        pys = [p[2] for p in points]
        self._py_lo = min(pys)
        self._py_hi = max(pys)

    @property
    def n(self) -> int:
        return len(self.points)

    def _predict_landing(
        self, hoop_y: float, hoop_x: float, platform_y: float
    ) -> float:
        distances = sorted(
            (
                ((py - hoop_y) ** 2 + (px - hoop_x) ** 2 + (pp - platform_y) ** 2) ** 0.5,
                lx,
            )
            for py, px, pp, lx in self.points
        )
        nearest = distances[: self.k]
        weights = [1.0 / max(d, 1.0) for d, _ in nearest]
        total = sum(weights)
        return sum(w * lx for w, (_, lx) in zip(weights, nearest)) / total

    def predict(self, hoop_y: float, hoop_x: float) -> float:
        """Return the platform_y whose modelled trajectory lands closest
        to hoop_x. Scans the empirical platform_y range in self._step
        increments."""
        best_py: float | None = None
        best_diff = float("inf")
        py = self._py_lo
        while py <= self._py_hi + 1e-9:
            pred_landing = self._predict_landing(hoop_y, hoop_x, py)
            diff = abs(pred_landing - hoop_x)
            if diff < best_diff:
                best_diff = diff
                best_py = py
            py += self._step
        # py_lo == py_hi degenerate case (all training shares one platform_y);
        # fall back to that single value rather than returning None.
        return float(best_py if best_py is not None else self._py_lo)


def fit_knn(
    rows: list[tuple[float, float, float]],
    k: int = 5,
    min_samples: int = 4,
) -> KnnPredictor | None:
    """Build a KnnPredictor from rows of (hoop_y, hoop_x, platform_y)."""
    if len(rows) < min_samples:
        return None
    points = [(float(r[0]), float(r[1]), float(r[2])) for r in rows]
    return KnnPredictor(points, k=k)


def fit_bivariate(
    rows: list[tuple[float, float, float]],
    min_samples: int = 4,
) -> BivariatePredictor | None:
    """Closed-form OLS for target_y = a*hoop_y + b*hoop_x + c.

    Returns None if the input isn't well-conditioned (singular normal
    matrix) or has too few samples. Solves the 3x3 normal equations
    via Cramer's rule.
    """
    if len(rows) < min_samples:
        return None
    points = [(float(r[0]), float(r[1]), float(r[2])) for r in rows]
    n = len(points)
    sx1 = sum(p[0] for p in points)
    sx2 = sum(p[1] for p in points)
    sy = sum(p[2] for p in points)
    sx1x1 = sum(p[0] * p[0] for p in points)
    sx2x2 = sum(p[1] * p[1] for p in points)
    sx1x2 = sum(p[0] * p[1] for p in points)
    sx1y = sum(p[0] * p[2] for p in points)
    sx2y = sum(p[1] * p[2] for p in points)
    M = [
        [sx1x1, sx1x2, sx1],
        [sx1x2, sx2x2, sx2],
        [sx1, sx2, n],
    ]
    v = [sx1y, sx2y, sy]
    sol = _solve_3x3(M, v)
    if sol is None:
        return None
    a, b, c = sol
    return BivariatePredictor(points, a, b, c)


def fit_gp(
    rows: list[tuple[float, float, float]],
    min_samples: int = 4,
) -> GpPredictor | None:
    """Fit a Gaussian Process to (hoop_y, hoop_x) → platform_y.

    Anisotropic RBF kernel (one length scale per dim, since hoop_y and
    hoop_x cover different ranges) × ConstantKernel for tunable signal
    variance, plus a WhiteKernel for the per-make platform_y detection
    jitter. Hyperparameters fit via marginal-likelihood maximisation
    with 4 restarts to dodge bad local optima.

    sklearn imports are lazy: bots that stick with KNN don't pay the
    sklearn import cost on startup.
    """
    if len(rows) < min_samples:
        return None
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    points = [(float(r[0]), float(r[1]), float(r[2])) for r in rows]
    X = np.array([[p[0], p[1]] for p in points])
    y = np.array([p[2] for p in points])

    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=[50.0, 50.0], length_scale_bounds=(1.0, 1e3))
        + WhiteKernel(noise_level=4.0, noise_level_bounds=(1e-2, 1e2))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=4,
    )
    gp.fit(X, y)
    return GpPredictor(points, gp)


def fit_trajectory_knn(
    rows: list[tuple[float, float, float, float]],
    k: int = 5,
    min_samples: int = 8,
    py_scan_step: float = 5.0,
) -> TrajectoryKnnPredictor | None:
    """Build a TrajectoryKnnPredictor from rows of
    (hoop_y, hoop_x, platform_y, ball_landing_x).

    Higher min_samples than the make-based predictors because the input
    space has one more dimension (platform_y) — sparse 3D coverage
    produces noisy KNN predictions. Returns None if too few samples or
    if the input list is empty.
    """
    if len(rows) < min_samples:
        return None
    points = [
        (float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in rows
    ]
    return TrajectoryKnnPredictor(points, k=k, py_scan_step=py_scan_step)


def _solve_3x3(M: list[list[float]], v: list[float]) -> tuple[float, float, float] | None:
    def det3(m: list[list[float]]) -> float:
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    D = det3(M)
    if abs(D) < 1e-9:
        return None
    out = []
    for i in range(3):
        Mi = [row[:] for row in M]
        for r in range(3):
            Mi[r][i] = v[r]
        out.append(det3(Mi) / D)
    return out[0], out[1], out[2]
