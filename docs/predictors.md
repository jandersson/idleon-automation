# Predictors — refresher notes

Prediction algorithms used (or planned) in `common/predictor.py`,
explained for the reader who's a decade out of "machine learning
school". Concepts are spelled out as a refresher, not first-time
teaching.

> **Companion visualisation:**
> [predictors.html](https://jandersson.github.io/idleon-automation/predictors.html)
> on GitHub Pages — interactive heatmap of each predictor's
> predicted-offset surface over a synthetic 2D playfield. Switch
> predictors, scrub K and noise, hover for predicted vs ground truth.
> (For local viewing, open `docs/predictors.html` directly in a
> browser — no server or build.)

The shared problem: given the current `(hoop_y, hoop_x)`, predict the
optimal `platform_y` to fire at. Training data is a list of past makes
— `(hoop_y, hoop_x, platform_y)` rows from `shots.db` filtered to
clean, non-clamped successes via `fetch_makes`.

The surface we're modelling is the function
`f(hoop_y, hoop_x) → optimal_platform_y`. It's smooth (parabolic ball
physics) but non-linear and non-stationary (different regions of the
playfield have different effective dynamics).

---

## KNN — K-Nearest Neighbors (current default)

**The mental model:** look up the K most similar past situations,
average their answers.

**The algorithm:**
1. Take the query point `(hoop_y, hoop_x)`.
2. Compute Euclidean distance to every past make in `(hoop_y, hoop_x)` space.
3. Take the K closest (K=3 in our setup).
4. Compute their offsets `platform_y - hoop_y`.
5. Average them weighted by `1 / distance` (closer neighbours dominate).
   A tiny epsilon prevents division by zero on exact matches.
6. Return `predicted_platform_y = hoop_y + weighted_avg_offset`.

**Why K=3:** smaller K is more local. With K=5 the dense centre-of-court
data drifted predictions in sparse corners. K=3 lets nearby data
dominate. Tradeoff: too small → noisy (one bad past make swings the
prediction); too large → over-smoothed (averages across regions whose
physics differ).

**No "fit" in the conventional sense:** KNN is a lazy learner. The
"fit" call just stores the points. All the work happens at predict time.

**Connection to other names:** what we call "KNN with inverse-distance
weighting" is essentially **Nadaraya-Watson kernel regression** with a
1/d kernel and a hard cutoff at the K nearest. Same idea, different
naming traditions.

---

## Bivariate OLS — Ordinary Least Squares (baseline)

**The mental model:** fit one global plane to all past makes.

**The algorithm:** find `(a, b, c)` minimising
`Σ (target_y - (a·hoop_y + b·hoop_x + c))²` over all training points.
Closed-form via the normal equations.

**Why it's kept:** A/B comparison and as a baseline. Older model from
before the predictor split.

**Why it's weaker than KNN here:** the optimal-platform-y surface isn't
globally planar. A single global plane gets dragged by dense regions and
under-fits sparse ones.

---

## Gaussian Process Regression — planned upgrade

GP is the next predictor on TODO. This section is the refresher I wish
I'd had in school — written for a reader who remembers being confused
by it.

### The conceptual jump that makes GP click

In linear regression you fit *parameters* to data. In GP regression
you fit a *distribution over functions* to data.

That sounds abstract. The trick is:

**A function is, in practice, just a vector of its values at a finite
set of points.** If I tell you a probability distribution over what the
vector `[f(x₁), f(x₂), ..., f(x_n)]` jointly looks like, I've defined a
probability distribution over the function (restricted to those n
points). You never actually need to think about "infinite-dimensional
distributions" — every prediction you make is finite.

**A Gaussian Process is the choice to make that joint distribution
multivariate Gaussian.** That's literally the definition: for any finite
set of input points, the joint distribution of the function values is a
multivariate Gaussian.

### The covariance matrix encodes our beliefs about the function

A multivariate Gaussian needs a mean (usually taken as 0 — we model the
deviation from a baseline) and a covariance matrix `K`. The entry
`K[i,j]` answers: how much do we expect `f(x_i)` and `f(x_j)` to be
similar?

The *kernel function* `k(x_i, x_j)` computes this. The standard choice
is the squared-exponential / RBF kernel:

```
k(x_i, x_j) = σ² · exp( -‖x_i − x_j‖² / (2ℓ²) )
```

Properties to internalise:
- `k(x, x) = σ²` — the variance at any single point.
- `k(x_i, x_j) → 0` as the points get far apart.
- `ℓ` ("lengthscale") controls how fast similarity decays with distance.
  Small ℓ → wiggly, untrusting functions. Large ℓ → smooth, broadly
  generalising functions.

The intuition the kernel formalises: nearby inputs should produce
similar outputs.

### Conditioning on training data — the part that's actually confusing

You have training data `(x₁, y₁), ..., (x_n, y_n)` and want to predict
`y*` at a new point `x*`. Stack training outputs and the unknown
together:

```
[y_train ; y*]  ~  Normal( 0,  [K_train       K_*,train ;
                                K_*,train ᵀ   k(x*, x*)] )
```

Where `K_train` is the `n×n` kernel matrix among training inputs and
`K_*,train` is `n×1` (kernel values between `x*` and each training
input).

By the standard conditional-Gaussian formula,
`p(y* | y_train)` is also Gaussian with:

```
mean      μ*  = K_*,train ᵀ · K_train⁻¹ · y_train
variance  σ²* = k(x*, x*)  −  K_*,train ᵀ · K_train⁻¹ · K_*,train
```

That's it. One matrix inverse computed once at "fit" time. Two
matrix-vector products at predict time. No iterative optimisation.

### What the variance gives us

The big strategic difference from KNN: σ²* tells us *how confident the
model is at this query point*. It's high when `x*` is far from training
data (extrapolation) and low when it's surrounded by training points.

For the hoops bot this turns the predictor from a point estimator into
a strategic signal: **skip shots where predicted variance > threshold**.
A KNN/OLS predictor will happily extrapolate into a region it has zero
data on; GP refuses to pretend it knows.

### Hyperparameters and fitting

`σ²` and `ℓ` aren't user-set — they're fitted by maximising the
*marginal likelihood* of the training data:

```
log p(y | X, σ, ℓ) = − ½ yᵀ K_train⁻¹ y  −  ½ log|K_train|  −  (n/2) log 2π
```

The two data-dependent terms balance fit (first term, smaller is
better) and complexity (second term, simpler kernel preferred). Standard
gradient ascent or scipy.optimize handles this. Ten lines of code, or
use sklearn's `GaussianProcessRegressor` which does it for you.

### Connection to what we already have

GP can be thought of as "kernel regression where the weights are
computed by inverting the joint kernel matrix", which accounts for *all*
training points together rather than just the K nearest neighbours. Plus
you get the predictive variance for free.

If KNN with inverse-distance weighting is "Nadaraya-Watson with a hard
cutoff", GP is "Nadaraya-Watson done properly with a principled global
weighting and a posterior variance".

### Practical gotchas

- **O(n³) cost from the matrix inverse.** For our ~100s of training
  points, that's microseconds. Sparse approximations (inducing points,
  Nyström) exist if it ever explodes.
- **Numerical stability of K_train⁻¹.** Add a small jitter `ε·I` to the
  diagonal before inverting. sklearn does this automatically.
- **Kernel choice matters more than people pretend.** RBF is the safe
  default for smooth surfaces. Matérn is a good "I want some
  flexibility about smoothness" default. Periodic kernels for periodic
  signals (could matter if e.g. platform bob phase ever becomes a
  feature).
- **The 0-mean assumption is about the *deviation from a baseline*.**
  If your y values are far from 0, subtract the training-set mean
  before fitting and add it back at predict time. Or use a constant
  mean function. Forgetting this leads to predictions biased toward 0
  in extrapolation regions.
