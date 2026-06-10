# Predictors — algorithm notes

Prediction algorithms used (or planned) in `common/predictor.py`.
Concepts are spelled out at refresher depth: the assumed reader has
seen the material before but doesn't remember the specifics.

> **Companion visualisation:**
> [predictor_playground.html](https://jandersson.github.io/idleon-automation/predictor_playground.html)
> on GitHub Pages — interactive heatmap of each predictor's
> predicted-offset surface over a synthetic 2D playfield. Switch
> predictors, scrub K and noise, hover for predicted vs ground truth.
> (For local viewing, open `docs/predictor_playground.html` directly
> in a browser — no server or build.)

The original shared problem: given the current `(hoop_y, hoop_x)`,
predict the optimal `platform_y` to fire at. Training data is a list
of past makes — `(hoop_y, hoop_x, platform_y)` rows from `shots.db`
filtered to clean successes via `fetch_makes`.

That framing has since generalised (June 2026): both bots' real
decision is *"of the firing states available right now, which one do I
take?"* — hoops picks a `(platform_y, vy, direction)` crossing from the
live bob buffer, darts picks which swing pass to fire on via its
centroid-dy. The later sections cover the models that answer that
question; the early sections are still the right on-ramp.

The surface we're modelling is the function
\\(f(\text{hoop\_y}, \text{hoop\_x}) \to \text{optimal\_platform\_y}\\).
It's smooth (parabolic ball physics) but non-linear and non-stationary
(different regions of the playfield have different effective dynamics).

---

## KNN — K-Nearest Neighbors (current default)

**The mental model:** look up the K most similar past situations,
average their answers.

**The algorithm:**
1. Take the query point \\((\text{hoop\_y}, \text{hoop\_x})\\).
2. Compute Euclidean distance to every past make in
   \\((\text{hoop\_y}, \text{hoop\_x})\\) space.
3. Take the \\(K\\) closest (\\(K=3\\) in our setup).
4. Compute their offsets \\(\text{platform\_y} - \text{hoop\_y}\\).
5. Average them weighted by \\(1 / d_i\\) (closer neighbours dominate).
   A tiny \\(\epsilon\\) prevents division by zero on exact matches.
6. Return \\(\widehat{\text{platform\_y}} = \text{hoop\_y} + \overline{\text{offset}}\\).

In one line, the prediction is:

$$\hat{y}(x_*) = \frac{\sum_{i \in N_K(x_*)} \frac{1}{d_i} \, y_i}{\sum_{i \in N_K(x_*)} \frac{1}{d_i}}$$

where \\(N_K(x_\ast)\\) is the set of \\(K\\) nearest neighbours of the query and
\\(d_i = \\|x_\ast - x_i\\|\\).

**Why \\(K=3\\):** smaller \\(K\\) is more local. With \\(K=5\\) the dense
centre-of-court data drifted predictions in sparse corners. \\(K=3\\) lets
nearby data dominate. Tradeoff: too small → noisy (one bad past make
swings the prediction); too large → over-smoothed (averages across
regions whose physics differ).

**No "fit" in the conventional sense:** KNN is a lazy learner. The
"fit" call just stores the points. All the work happens at predict time.

**Connection to other names:** what we call "KNN with inverse-distance
weighting" is essentially **Nadaraya-Watson kernel regression** with a
\\(1/d\\) kernel and a hard cutoff at the \\(K\\) nearest. Same idea, different
naming traditions.

---

## Bivariate OLS — Ordinary Least Squares (baseline)

**The mental model:** fit one global plane to all past makes.

**The algorithm:** find \\((a, b, c)\\) minimising

$$\sum_i \left( t_i - \big( a \cdot h_{y,i} + b \cdot h_{x,i} + c \big) \right)^2$$

over all training points. Closed-form via the normal equations
(\\(\hat{\beta} = (X^\top X)^{-1} X^\top y\\)).

**Why it's kept:** A/B comparison and as a baseline. Older model from
before the predictor split.

**Why it's weaker than KNN here:** the optimal-platform-y surface isn't
globally planar. A single global plane gets dragged by dense regions and
under-fits sparse ones.

---

## Gaussian Process Regression — shipped (the default since May 2026)

This section is the refresher I wish I'd had in school — written for a
reader who remembers being confused by it. Status notes: `gp` has been
the default `PREDICTOR_KIND` since it landed; its posterior σ scales
the miss-sweep step sizes; and a trajectory-regression variant
(`trajectory_gp`, fit on every shot's flight arrival rather than makes
only) tested statistically *tied* with it over n=103 — the early
3-session window where it looked better was small-N luck (issue #23,
a lesson that now gates every promotion decision in this repo).

### The conceptual jump that makes GP click

In linear regression you fit *parameters* to data. In GP regression
you fit a *distribution over functions* to data.

That sounds abstract. The trick is:

**A function is, in practice, just a vector of its values at a finite
set of points.** If I tell you a probability distribution over what the
vector \\([f(x_1), f(x_2), \dots, f(x_n)]\\) jointly looks like, I've
defined a probability distribution over the function (restricted to
those \\(n\\) points). You never actually need to think about
"infinite-dimensional distributions" — every prediction you make is
finite.

**A Gaussian Process is the choice to make that joint distribution
multivariate Gaussian.** That's literally the definition: for any finite
set of input points, the joint distribution of the function values is a
multivariate Gaussian.

### The covariance matrix encodes our beliefs about the function

A multivariate Gaussian needs a mean (usually taken as \\(0\\) — we model the
deviation from a baseline) and a covariance matrix \\(K\\). The entry
\\(K_{ij}\\) answers: how much do we expect \\(f(x_i)\\) and \\(f(x_j)\\) to be
similar?

The *kernel function* \\(k(x_i, x_j)\\) computes this. The standard choice
is the squared-exponential / RBF kernel:

$$k(x_i, x_j) = \sigma^2 \, \exp\!\left( -\frac{\|x_i - x_j\|^2}{2 \ell^2} \right)$$

Properties to internalise:

- \\(k(x, x) = \sigma^2\\) — the variance at any single point.
- \\(k(x_i, x_j) \to 0\\) as the points get far apart.
- \\(\ell\\) ("lengthscale") controls how fast similarity decays with
  distance. Small \\(\ell\\) → wiggly, untrusting functions. Large \\(\ell\\) →
  smooth, broadly generalising functions.

The intuition the kernel formalises: nearby inputs should produce
similar outputs.

### Conditioning on training data — the part that's actually confusing

You have training data \\((x_1, y_1), \dots, (x_n, y_n)\\) and want to
predict \\(y_\ast \\) at a new point \\(x_\ast \\). Stack training outputs and the
unknown together:

$$\begin{bmatrix} \mathbf{y}_{\text{train}} \\ y_* \end{bmatrix} \;\sim\; \mathcal{N}\!\left( \mathbf{0}, \;\begin{bmatrix} K_{\text{train}} & K_{*,\text{train}} \\ K_{*,\text{train}}^{\top} & k(x_*, x_*) \end{bmatrix} \right)$$

Where \\(K_{\text{train}}\\) is the \\(n \times n\\) kernel matrix among
training inputs and \\(K_{\ast,\text{train}}\\) is the \\(n \times 1\\) vector of
kernel values between \\(x_\ast \\) and each training input.

By the standard conditional-Gaussian formula,
\\(p(y_\ast \mid \mathbf{y}_{\text{train}})\\) is also Gaussian with:

$$\mu_* \;=\; K_{*,\text{train}}^{\top} \, K_{\text{train}}^{-1} \, \mathbf{y}_{\text{train}}$$

$$\sigma_*^2 \;=\; k(x_*, x_*) \;-\; K_{*,\text{train}}^{\top} \, K_{\text{train}}^{-1} \, K_{*,\text{train}}$$

That's it. One matrix inverse computed once at "fit" time. Two
matrix-vector products at predict time. No iterative optimisation.

### What the variance gives us

The big strategic difference from KNN: \\(\sigma_\ast^2\\) tells us *how
confident the model is at this query point*. It's high when \\(x_\ast \\) is
far from training data (extrapolation) and low when it's surrounded by
training points.

For the hoops bot this turns the predictor from a point estimator into
a strategic signal: **skip shots where predicted variance > threshold**.
A KNN/OLS predictor will happily extrapolate into a region it has zero
data on; GP refuses to pretend it knows.

### Hyperparameters and fitting

\\(\sigma^2\\) and \\(\ell\\) aren't user-set — they're fitted by maximising
the *marginal likelihood* of the training data:

$$\log p(\mathbf{y} \mid X, \sigma, \ell) \;=\; -\tfrac{1}{2} \, \mathbf{y}^{\top} K_{\text{train}}^{-1} \mathbf{y} \;-\; \tfrac{1}{2} \log |K_{\text{train}}| \;-\; \tfrac{n}{2} \log 2\pi$$

The two data-dependent terms balance fit (first term, smaller is
better) and complexity (second term, simpler kernel preferred). Standard
gradient ascent or `scipy.optimize` handles this. Ten lines of code, or
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

- **\\(O(n^3)\\) cost from the matrix inverse.** For our ~100s of training
  points, that's microseconds. Sparse approximations (inducing points,
  Nyström) exist if it ever explodes.
- **Numerical stability of \\(K_{\text{train}}^{-1}\\).** Add a small
  jitter \\(\epsilon I\\) to the diagonal before inverting. sklearn does
  this automatically.
- **Kernel choice matters more than people pretend.** RBF is the safe
  default for smooth surfaces. Matérn is a good "I want some
  flexibility about smoothness" default. Periodic kernels for periodic
  signals (could matter if e.g. platform bob phase ever becomes a
  feature).
- **The 0-mean assumption is about the *deviation from a baseline*.**
  If your \\(y\\) values are far from \\(0\\), subtract the training-set mean
  before fitting and add it back at predict time. Or use a constant
  mean function. Forgetting this leads to predictions biased toward
  \\(0\\) in extrapolation regions.

---

## GP Classification — `make_prob` (hoops, June 2026)

The same "distribution over functions" idea, pointed at a yes/no
question: \\(P(\text{make} \mid \text{hoop\_y}, \text{hoop\_x}, \text{platform\_y}, \text{platform\_vy})\\).

### Why classification, when we had a perfectly good regressor

Because of **censoring** — the failure mode that killed the reach-
regression approach at the clank band. The trajectory predictors model
"how far does the flight reach", but when the ball clips the rim front
and bounces back, its *unobstructed* reach is unknowable: the structure
censored the measurement. Those rows must be excluded from a reach
model — which means the model literally cannot see the hazard, and will
happily steer into the clank zone its training data was filtered of.

Classification has no such problem. A clank is simply `made = 0`. Every
shot is a labeled sample; nothing is censored; the wild exploration
shots are *more* useful, not less. When your outcome is "did the thing
work", model that directly.

### How GP classification differs from GP regression (refresher depth)

You can't put a Gaussian likelihood on a 0/1 label. The standard
construction:

1. Keep the GP exactly as before, but over a **latent function**
   \\(f(x)\\) you never observe.
2. Squash it through a link: \\(P(y{=}1 \mid x) = \sigma(f(x))\\), with
   \\(\sigma\\) the logistic (or probit) function.
3. The posterior over \\(f\\) is no longer Gaussian (the Bernoulli
   likelihood breaks conjugacy), so you approximate it — sklearn's
   `GaussianProcessClassifier` uses the **Laplace approximation**: find
   the posterior mode, fit a Gaussian there, carry on as if.

Mental model: same prior over smooth functions, but now the function is
"log-odds of success over the state space", and training bends it up
around observed makes and down around misses.

### The part that's actually new: candidates, not queries

The regression predictors answer "what platform_y should I want?". The
classifier answers a better-shaped question: "here are the firing
states the *live platform motion* will actually offer me in the next
few seconds — score each." Candidates are built empirically from the
bob buffer (each recently-swept y carries the velocity observed there;
the bob is periodic, so that's the expected velocity of the next
crossing), and the bot fires at the argmax. Direction comes along for
free — a candidate knows whether it's an up- or down-crossing — which
let the model out-vote the hand-written direction policy in its first
session, correctly.

First results (n=29 over two sessions): 72%, with the 95% CI clear of
the previous policy stack's 34.8%. Promotion to default pending the
~50-shot mark, because issue #23's ghost sits in on every one of these
meetings.

---

## Next: E[stripe] for darts — GP regression again, different cleverness

Darts turned out to have exactly one steerable axis: *where in the
release pass the click lands* (measured by `centroid_dy`) sets the
launch angle, which sets arc height, which sets the board stripe —
flat passes hit the mid-board red/bullseye stripes at ~5× the baseline
rate. Wind bends the same arc, in 2D (the indicator shows speed and a
direction arrow that ranges over the full compass).

The planned model (issue #41): \\(E[\text{stripe value} \mid \text{wind\_x}, \text{wind\_y}, \text{dy}, \text{pose\_y}]\\).
Two design choices worth recording:

- **Expectation over probability.** \\(P(\text{bullseye})\\) sounds
  natural but bullseyes are ~30 positives all-time — a classifier
  starves. Stripe value (0–5) labels every throw, and chasing the
  expectation chases bullseyes anyway, since the 5 dominates it.
  (Compare with hoops above, where classification won — the deciding
  question is *which label is dense*, not which sounds fancier.)
- **Wind as vector components, not (speed, direction).** Angles
  wrap around (359° ≈ 1°), and an RBF kernel has no idea — it sees
  maximal distance where the physics sees near-identity. Decompose
  instead: \\(\text{wind\_x} = s\cos\theta\\), \\(\text{wind\_y} = s\sin\theta\\) (screen convention: y-positive = down, matching every
  other coordinate in the repo). Speed isn't lost — it's the norm of
  the vector — and each component acts on a flight axis we measure:
  wind_y presses the arc (the stripe axis), wind_x stretches range.
  When the GP fits, its gradient along wind_y should read like a
  physics textbook; if it doesn't, the parse is wrong and we'll know
  before trusting it with throws.

Until the model has its ~150 throws across two wind tiers, a static
dy band ([-8, -5], with an ε-exploration throw every 8th) does the
aiming and generates the training data — the same
policy-first-model-second sequence that worked for hoops.
