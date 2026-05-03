"""Predictors for "where should the platform be at fire time?".

Each predictor is fit on a list of past makes — `(hoop_y, hoop_x,
platform_y)` tuples — and exposes a `predict(hoop_y, hoop_x) -> float`
method returning the predicted optimal `platform_y` (i.e. the value the
caller should aim `target_y` at).

Two implementations live here so they can be swapped at the call site:

- `KnnPredictor` — inverse-distance-weighted KNN over past makes.
  Adapts to local curvature in the (hoop_y, hoop_x) → optimal_py
  surface; recommended default.
- `BivariatePredictor` — closed-form OLS for
  `target_y = a*hoop_y + b*hoop_x + c`. Older model, kept here for
  A/B comparison and as a baseline.

`fit_knn` / `fit_bivariate` are factory helpers that take the rows and
return a fitted predictor (or None if too few samples).

KNN refresher (for the reader who took ML class a decade ago):
- "K-Nearest Neighbors". K is just a hyperparameter — the count of
  neighbors to consult. We use K=3 (set at the call site in
  minigames/hoops/main.py).
- For each new shot:
    1. Take the current (hoop_y, hoop_x) you need to aim at.
    2. Look up all past makes (already filtered by direction and
       non-clamped in fetch_makes).
    3. Find the K closest past makes by Euclidean distance in
       (hoop_y, hoop_x) space.
    4. Compute their offsets (platform_y - hoop_y) and average them
       weighted by 1 / distance, so closer neighbors count more
       (a tiny epsilon prevents division by zero on exact matches).
    5. The weighted-average offset → predicted platform_y =
       hoop_y + offset. Caller fires when actual platform_y crosses it.
- Tradeoff on K: too small → noisy (one bad past make swings the
  prediction); too large → over-smoothed (averages across regions
  whose physics differ). K=3 was where it settled empirically — see
  the comment in main.py near the fit_knn call.
- No training step beyond storing the points; "fit" is a misnomer —
  KNN is a lazy learner that just memorises and queries at predict
  time.

OLS refresher:
- "Ordinary Least Squares". Closed-form linear regression: pick
  (a, b, c) to minimise sum of (target_y - (a*hoop_y + b*hoop_x + c))^2
  over the training points. Fast, interpretable, but assumes the
  surface is globally planar — which the hoops physics isn't.
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
