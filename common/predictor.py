"""Predictors for "where should the platform be at fire time?".

Most predictors here are fit on a list of past *makes* —
`(hoop_y, hoop_x, platform_y)` tuples — and expose
`predict(hoop_y, hoop_x) -> float` returning the predicted optimal
`platform_y` (i.e. the value the caller should aim `target_y` at).

Five implementations live here so they can be swapped at the call site:

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
- `TrajectoryGpPredictor` — same trajectory-regression shape as the
  KNN flavour, but uses a 3D GP for the landing model. Smoother fit
  than KNN in sparse 3D regions and exposes posterior variance for
  uncertainty-aware downstream logic.
- `TrajectoryRfPredictor` — same shape, with a sklearn random forest
  in place of GP/KNN. RFs handle interactions naturally and don't
  extrapolate (predict mean of training samples in each leaf), which
  is robust on small datasets but can produce step-function outputs.

`fit_knn` / `fit_bivariate` / `fit_gp` / `fit_trajectory_knn` /
`fit_trajectory_gp` are factory helpers that take the rows and return
a fitted predictor (or None if too few samples). The first three take
`fetch_makes` rows; the trajectory ones take `fetch_clean_trajectories`
rows (4-tuples including `ball_landing_x`) plus the make rows for the
envelope clamp.

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

    To prevent the scan from picking platform_y values at the bob extremes
    where the model surface is unconstrained (the failure mode from
    issue #19), `predict` first finds the M nearest *makes* in
    (hoop_y, hoop_x) and bounds the scan to their min/max platform_y
    plus a small margin. The trajectory model still chooses inside that
    envelope; it can't pick wild extrapolations. When no makes are
    provided (legacy callers, cold start), the scan falls back to the
    full empirical platform_y range.
    """

    def __init__(
        self,
        points: list[tuple[float, float, float, float]],
        makes: list[tuple[float, float, float]] | None = None,
        k: int = 5,
        py_scan_step: float = 5.0,
        envelope_k: int = 5,
        envelope_margin_px: float = 10.0,
    ):
        # points: list of (hoop_y, hoop_x, platform_y, ball_landing_x)
        # makes:  list of (hoop_y, hoop_x, platform_y) for the envelope clamp
        self.points = points
        self.makes = makes or []
        self.k = min(k, len(points))
        self._step = py_scan_step
        self._envelope_k = envelope_k
        self._envelope_margin = envelope_margin_px
        # Global platform_y envelope from the trained shots — used as the
        # outer bound when there are no nearby makes to scope to.
        pys = [p[2] for p in points]
        self._py_lo = min(pys)
        self._py_hi = max(pys)

    @property
    def n(self) -> int:
        return len(self.points)

    def _envelope(self, hoop_y: float, hoop_x: float) -> tuple[float, float]:
        """Scan bounds for a query hoop: M nearest makes' platform_y range
        plus a margin, clipped to the global empirical envelope. If no
        makes, fall back to the global envelope."""
        if not self.makes:
            return self._py_lo, self._py_hi
        nearest = sorted(
            self.makes,
            key=lambda m: (m[0] - hoop_y) ** 2 + (m[1] - hoop_x) ** 2,
        )[: self._envelope_k]
        pys = [m[2] for m in nearest]
        lo = max(min(pys) - self._envelope_margin, self._py_lo)
        hi = min(max(pys) + self._envelope_margin, self._py_hi)
        return lo, hi

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
        to hoop_x. Scan range is the nearby-makes envelope (or the global
        empirical envelope if no makes were provided)."""
        lo, hi = self._envelope(hoop_y, hoop_x)
        best_py: float | None = None
        best_diff = float("inf")
        py = lo
        while py <= hi + 1e-9:
            pred_landing = self._predict_landing(hoop_y, hoop_x, py)
            diff = abs(pred_landing - hoop_x)
            if diff < best_diff:
                best_diff = diff
                best_py = py
            py += self._step
        # Degenerate envelope (lo == hi, e.g. one make in the neighbourhood):
        # fall back to that single value rather than returning None.
        return float(best_py if best_py is not None else lo)


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


class TrajectoryGpPredictor:
    """Same shape as TrajectoryKnnPredictor but uses a 3D Gaussian
    Process for the (hoop_y, hoop_x, platform_y) -> ball_landing_x
    surface. Smoother fit than 3D KNN in sparse regions and exposes
    posterior variance via predict_landing_with_std for callers that
    want uncertainty-aware logic.

    Same interface contract: predict(hoop_y, hoop_x) returns the
    platform_y inside the nearby-makes envelope whose modelled landing
    is closest to hoop_x.
    """

    def __init__(
        self,
        points: list[tuple[float, float, float, float]],
        makes: list[tuple[float, float, float]] | None,
        gp,
        py_scan_step: float = 5.0,
        envelope_k: int = 5,
        envelope_margin_px: float = 10.0,
    ):
        self.points = points
        self.makes = makes or []
        self._gp = gp
        self._step = py_scan_step
        self._envelope_k = envelope_k
        self._envelope_margin = envelope_margin_px
        pys = [p[2] for p in points]
        self._py_lo = min(pys)
        self._py_hi = max(pys)

    @property
    def n(self) -> int:
        return len(self.points)

    def _envelope(self, hoop_y: float, hoop_x: float) -> tuple[float, float]:
        if not self.makes:
            return self._py_lo, self._py_hi
        nearest = sorted(
            self.makes,
            key=lambda m: (m[0] - hoop_y) ** 2 + (m[1] - hoop_x) ** 2,
        )[: self._envelope_k]
        pys = [m[2] for m in nearest]
        lo = max(min(pys) - self._envelope_margin, self._py_lo)
        hi = min(max(pys) + self._envelope_margin, self._py_hi)
        return lo, hi

    def _predict_landing_batch(
        self, hoop_y: float, hoop_x: float, platform_ys: list[float]
    ) -> list[float]:
        import numpy as np

        X = np.array([[hoop_y, hoop_x, py] for py in platform_ys])
        return list(self._gp.predict(X))

    def predict_landing_with_std(
        self, hoop_y: float, hoop_x: float, platform_y: float
    ) -> tuple[float, float]:
        """Posterior (mean, std) for the predicted ball_landing_x at the
        given (hoop, platform_y). Useful for uncertainty-aware logic."""
        import numpy as np

        X = np.array([[hoop_y, hoop_x, platform_y]])
        mu, std = self._gp.predict(X, return_std=True)
        return float(mu[0]), float(std[0])

    def predict(self, hoop_y: float, hoop_x: float) -> float:
        """Return the platform_y inside the nearby-makes envelope whose
        modelled trajectory lands closest to hoop_x."""
        py, _ = self.predict_with_std(hoop_y, hoop_x)
        return py

    def predict_with_std(
        self, hoop_y: float, hoop_x: float
    ) -> tuple[float, float]:
        """Same as predict() but also returns the posterior landing-x std
        at the chosen platform_y. Mirrors GpPredictor.predict_with_std's
        interface so callers can scale perturbation steps by predictor
        confidence regardless of which GP variant is fit."""
        lo, hi = self._envelope(hoop_y, hoop_x)
        scan_pys: list[float] = []
        py = lo
        while py <= hi + 1e-9:
            scan_pys.append(py)
            py += self._step
        if not scan_pys:
            return float(lo), 0.0
        landings = self._predict_landing_batch(hoop_y, hoop_x, scan_pys)
        best_py, _ = min(
            zip(scan_pys, landings),
            key=lambda pair: abs(pair[1] - hoop_x),
        )
        _, std = self.predict_landing_with_std(hoop_y, hoop_x, best_py)
        return float(best_py), float(std)


def fit_trajectory_knn(
    rows: list[tuple[float, float, float, float]],
    makes: list[tuple[float, float, float]] | None = None,
    k: int = 5,
    min_samples: int = 8,
    py_scan_step: float = 5.0,
) -> TrajectoryKnnPredictor | None:
    """Build a TrajectoryKnnPredictor from rows of
    (hoop_y, hoop_x, platform_y, ball_landing_x). Optionally pass `makes`
    (the same 3-tuples fetch_makes returns) so the predictor can clamp
    its scan to a nearby-makes envelope rather than the full bob range.

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
    make_points = (
        [(float(m[0]), float(m[1]), float(m[2])) for m in makes]
        if makes is not None
        else None
    )
    return TrajectoryKnnPredictor(
        points, makes=make_points, k=k, py_scan_step=py_scan_step,
    )


class TrajectoryRfPredictor:
    """Random-forest variant of the trajectory predictor family.

    Same external contract as TrajectoryKnnPredictor /
    TrajectoryGpPredictor: predict(hoop_y, hoop_x) returns the
    platform_y inside the nearby-makes envelope whose modelled landing
    is closest to hoop_x. Internally uses a sklearn
    RandomForestRegressor over (hoop_y, hoop_x, platform_y) →
    ball_landing_x.

    Tree ensembles handle feature interactions and noisy data well at
    small sample sizes — a complement to GP (smooth) and KNN (local).
    Trees produce step-function predictions, which means the scan
    can have plateaus in pred_landing(platform_y). The argmin still
    returns a single platform_y per call.
    """

    def __init__(
        self,
        points: list[tuple[float, float, float, float]],
        makes: list[tuple[float, float, float]] | None,
        rf,
        py_scan_step: float = 5.0,
        envelope_k: int = 5,
        envelope_margin_px: float = 10.0,
    ):
        self.points = points
        self.makes = makes or []
        self._rf = rf
        self._step = py_scan_step
        self._envelope_k = envelope_k
        self._envelope_margin = envelope_margin_px
        pys = [p[2] for p in points]
        self._py_lo = min(pys)
        self._py_hi = max(pys)

    @property
    def n(self) -> int:
        return len(self.points)

    def _envelope(self, hoop_y: float, hoop_x: float) -> tuple[float, float]:
        if not self.makes:
            return self._py_lo, self._py_hi
        nearest = sorted(
            self.makes,
            key=lambda m: (m[0] - hoop_y) ** 2 + (m[1] - hoop_x) ** 2,
        )[: self._envelope_k]
        pys = [m[2] for m in nearest]
        lo = max(min(pys) - self._envelope_margin, self._py_lo)
        hi = min(max(pys) + self._envelope_margin, self._py_hi)
        return lo, hi

    def predict(self, hoop_y: float, hoop_x: float) -> float:
        import numpy as np

        lo, hi = self._envelope(hoop_y, hoop_x)
        scan_pys: list[float] = []
        py = lo
        while py <= hi + 1e-9:
            scan_pys.append(py)
            py += self._step
        if not scan_pys:
            return float(lo)
        X = np.array([[hoop_y, hoop_x, py] for py in scan_pys])
        landings = self._rf.predict(X)
        best_py, _ = min(
            zip(scan_pys, landings),
            key=lambda pair: abs(pair[1] - hoop_x),
        )
        return float(best_py)


def fit_trajectory_rf(
    rows: list[tuple[float, float, float, float]],
    makes: list[tuple[float, float, float]] | None = None,
    min_samples: int = 12,
    n_estimators: int = 100,
    py_scan_step: float = 5.0,
    random_state: int = 0,
) -> TrajectoryRfPredictor | None:
    """Fit a RandomForestRegressor to (hoop_y, hoop_x, platform_y) →
    ball_landing_x. Returns None if too few samples.

    Same min_samples bar (12) as fit_trajectory_gp — both work in 3D
    with mixed make/miss data. Default n_estimators=100 (sklearn
    default); larger forests don't help much on n~200 and double
    fit time. random_state pinned for reproducible per-session output
    (otherwise the scan landing-prediction is jittery between bot
    restarts on the same data).
    """
    if len(rows) < min_samples:
        return None
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    points = [
        (float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in rows
    ]
    X = np.array([[p[0], p[1], p[2]] for p in points])
    y = np.array([p[3] for p in points])
    rf = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1,
    )
    rf.fit(X, y)
    make_points = (
        [(float(m[0]), float(m[1]), float(m[2])) for m in makes]
        if makes is not None
        else None
    )
    return TrajectoryRfPredictor(
        points, makes=make_points, rf=rf, py_scan_step=py_scan_step,
    )


def fit_trajectory_gp(
    rows: list[tuple[float, float, float, float]],
    makes: list[tuple[float, float, float]] | None = None,
    min_samples: int = 12,
    py_scan_step: float = 5.0,
) -> TrajectoryGpPredictor | None:
    """Fit a 3D Gaussian Process to (hoop_y, hoop_x, platform_y) →
    ball_landing_x. Returns None if too few samples.

    Higher min_samples than the make-only fit_gp (12 vs 4) because the
    input space has one more dimension and the noise level is higher
    (we include misses). Anisotropic RBF length scales [50, 50, 30]:
    the third dimension (platform_y) is the one we scan over at predict
    time, so we want a tighter scale there to capture local curvature
    of the launch-physics function.

    sklearn imports are lazy so bots that stick with KNN don't pay the
    sklearn import cost.
    """
    if len(rows) < min_samples:
        return None
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    points = [
        (float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in rows
    ]
    X = np.array([[p[0], p[1], p[2]] for p in points])
    y = np.array([p[3] for p in points])

    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=[50.0, 50.0, 30.0],
              length_scale_bounds=(1.0, 1e3))
        + WhiteKernel(noise_level=400.0, noise_level_bounds=(1e-1, 1e4))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=4,
    )
    gp.fit(X, y)
    make_points = (
        [(float(m[0]), float(m[1]), float(m[2])) for m in makes]
        if makes is not None
        else None
    )
    return TrajectoryGpPredictor(
        points, makes=make_points, gp=gp, py_scan_step=py_scan_step,
    )


class MakeProbPredictor:
    """P(make | hoop_y, hoop_x, platform_y, platform_vy) classifier (#38).

    Unlike the reach-regression predictors, this one has no censoring
    problem: every shot is a labeled sample (a structure clank is simply
    made=0), so it can learn the clank band's geometry that reach models
    structurally cannot see — at the band's hoops, reach is right and
    the make probability is what varies.

    It deliberately does NOT implement predict(hoop_y, hoop_x): firing
    direction and bob phase are embedded in the candidate (platform_y,
    platform_vy) pairs, which only the caller can supply (built
    empirically from the live platform sample buffer — each recent
    crossing of a y is a candidate carrying the vy actually observed
    there). Use score_candidates and take the argmax.
    """

    def __init__(self, points: list[tuple[float, float, float, float, int]], clf):
        self.points = points
        self._clf = clf

    @property
    def n(self) -> int:
        return len(self.points)

    def score_candidates(
        self,
        hoop_y: float,
        hoop_x: float,
        candidates: list[tuple[float, float]],
    ) -> list[float]:
        """P(make) for each candidate (platform_y, platform_vy) at the
        given hoop. Order matches the input."""
        import numpy as np

        X = np.array([[hoop_y, hoop_x, py, vy] for py, vy in candidates])
        return [float(p) for p in self._clf.predict_proba(X)[:, 1]]


def fit_make_prob(
    rows: list[tuple[float, float, float, float, int]],
    min_samples: int = 60,
    min_makes: int = 8,
) -> MakeProbPredictor | None:
    """Fit the make-probability classifier on
    (hoop_y, hoop_x, platform_y, platform_vy, made) rows.

    GP classifier with an anisotropic RBF: length scales mirror the
    trajectory GP's (hoop dims smooth at ~50px) with tighter platform_y
    and vy scales — the band's make window is narrow in both. Returns
    None below the sample/positive-label floors (a classifier without
    positives can only say "never fire").

    sklearn import is lazy, matching the other fit_* factories.
    """
    if len(rows) < min_samples or sum(r[4] for r in rows) < min_makes:
        return None
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    points = [
        (float(r[0]), float(r[1]), float(r[2]), float(r[3]), int(r[4]))
        for r in rows
    ]
    X = np.array([[p[0], p[1], p[2], p[3]] for p in points])
    y = np.array([p[4] for p in points])
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(
        length_scale=[50.0, 50.0, 25.0, 25.0],
        length_scale_bounds=(1.0, 1e3),
    )
    clf = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=1)
    clf.fit(X, y)
    return MakeProbPredictor(points, clf)


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
