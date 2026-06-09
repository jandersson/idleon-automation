"""Hoops bot: main loop and per-minigame config.

The targeting/outcome design (bounce-aware arrival, per-hoop firing
direction, velocity instrumentation, respawn-corrected make detection,
free-shot exploration) is the product of the June 2026 miss
investigation — rationale and evidence in docs/hoops_findings.md and
issue #37.
"""
import os
import random
import time
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay, check_failsafe
from common.monitor import make_shot_dir, save_frame, save_meta
from common.regions import get_region
from common.session_log import session_log
from common.git_info import current_code_commit
from minigames.hoops.shot_log import open_db, log_shot, set_make, fetch_makes, fetch_clean_trajectories, CLEAN_MAKE_TOLERANCE, MAX_BACK_DRIFT_PX
from common.predictor import fit_knn, fit_bivariate, fit_gp, fit_trajectory_knn, fit_trajectory_gp, fit_trajectory_rf
from common.auto_commit import commit_file_if_changed
from minigames.hoops.review_nag import maybe_print_nag
from common.ball_trajectory import analyse_shot_dir
from common.score_ocr import read_score as _read_score_tesseract
from common.score_template_ocr import make_score_reader
from common.window import get_bounds, WindowNotFoundError
from minigames.hoops.detector import find_rim, find_platform, find_game_over, find_game_prompt, score_region, score_changed

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
SHOT_DB_PATH = _HERE / "assets" / "shots.db"


# Template-based score reader (replaces tesseract for the common case;
# falls back when a digit isn't yet in the template library). Templates
# bootstrap from past clean OCR reads via
# scripts/bootstrap_digit_templates.py. Closes #15.
_template_reader = make_score_reader(_HERE / "assets" / "digit_templates")


def read_score(crop):
    """Read the hoops score from a region crop. Template OCR first
    (deterministic), tesseract fallback for digits not yet captured."""
    val = _template_reader(crop)
    if val is not None:
        return val
    return _read_score_tesseract(crop)

WINDOW_TITLE = "Legends Of Idleon"
POLL_INTERVAL = 0.005  # Tight loop: each find_platform call already takes
                       # ~15-30ms (multi-scale matching), so this sleep is
                       # mostly irrelevant — we're CPU-bound on the matcher.
                       # Lowered from 0.02 anyway to make sure sleep isn't
                       # the bottleneck on cycles where matching is fast.

# Offset is learned from past shots in shots.db at session start (see the
# fit_* factories in common.predictor).
#
# Cold-start fallback: when there are <3 confirmed makes for the active
# REQUIRED_DIRECTION, use this constant offset. Picked from the cluster of
# makes in dir=up data (platform_y at make ≈ hoop_y + 5..20).
COLD_START_OFFSET = 20

def _predicted_offset(hoop_y: int, hoop_x: int, predictor) -> int:
    """KNN-predicted offset for the given hoop position. Cold-start
    constant if no predictor is fitted yet."""
    if predictor is None:
        return COLD_START_OFFSET
    target_y = predictor.predict(hoop_y, hoop_x)
    return int(round(target_y - hoop_y))


def _predicted_std(hoop_y: int, hoop_x: int, predictor) -> float | None:
    """Posterior std of the prediction, or None if the predictor doesn't
    expose one. Used to scale perturbation magnitudes by confidence:
    sparse-region predictions (high σ) take bigger first steps."""
    if predictor is None or not hasattr(predictor, "predict_with_std"):
        return None
    try:
        _, sigma = predictor.predict_with_std(hoop_y, hoop_x)
        return float(sigma)
    except Exception:
        return None


def _compute_offset(hoop_y: int, hoop_x: int, predictor) -> int:
    """Compute the offset the bot will fire with.

    With a predictor: KNN over past makes returns the predicted optimal
    platform_y; offset = target_y - hoop_y. The KNN approach naturally
    adapts to local curvature in the (hoop_y, hoop_x) → optimal_py
    surface, so we no longer need hardcoded overrides for regions
    where a linear fit was wrong.

    Cold start: constant offset until we have enough makes for KNN.
    """
    return _predicted_offset(hoop_y, hoop_x, predictor)


# When a hoop position is missed, sweep offsets around the predicted value
# on subsequent shots. Hoop only respawns on makes, so without perturbation
# we'd burn every life trying the same wrong offset.
#
# First 9 entries: ±32 in 8px steps — dense around the predicted value
# (where most makes live). Beyond that: jump to ±48, ±64, ±80 in larger
# steps to cover stuck regions whose make zone is outside the predicted
# neighborhood. (617, 372) has been swept across 56px of fired_py without
# a make — the make zone there is genuinely far from where the regression
# predicts.
# Margin (px) inside the platform's observed bob range that target_y must
# stay within after perturbation. Prevents a stuck loop where target_y
# lands exactly at the bob extreme — at the apex the platform only kisses
# that y for a single ephemeral sample, which the bot reliably misses
# with the tight Y_TOLERANCE. Observed 2026-05-09 session 20:18:44.
PERTURBATION_BOB_MARGIN = 5  # px


def _estimate_bob_period_ms(
    samples: list[tuple[float, int, int]],
    min_prominence_px: int = 8,
    window: int = 5,
) -> int | None:
    """Mean ms between consecutive bob peaks in (timestamp, px, py) samples.

    Detects local maxima in the py series (largest y is the bottom of
    the bob, since y grows downward) and averages the time deltas
    between consecutive peaks. Returns None when fewer than 2 distinct
    peaks are visible — typically because the buffer is too short or
    the platform isn't moving (e.g. clamped at an extreme).

    `min_prominence_px` rejects spurious peaks created by detector
    jitter: a real peak must rise at least N px from the local minima
    on either side within `window` samples.
    """
    if len(samples) < 5:
        return None
    pys = [s[2] for s in samples]
    n = len(pys)
    peak_times: list[float] = []
    for i in range(1, n - 1):
        if pys[i] > pys[i - 1] and pys[i] >= pys[i + 1]:
            left = pys[max(0, i - window): i]
            right = pys[i + 1: min(n, i + window + 1)]
            left_min = min(left) if left else pys[i]
            right_min = min(right) if right else pys[i]
            prominence = pys[i] - max(left_min, right_min)
            if prominence >= min_prominence_px:
                peak_times.append(samples[i][0])
    if len(peak_times) < 2:
        return None
    deltas = [peak_times[k + 1] - peak_times[k] for k in range(len(peak_times) - 1)]
    return int(sum(deltas) / len(deltas) * 1000)


def _platform_velocity(
    samples: list[tuple[float, int, int]],
    window_s: float = 1.5,
    min_span_s: float = 0.2,
) -> float | None:
    """Platform vertical velocity (px/s, positive = downward) at the time
    of the newest sample, from the (timestamp, px, py) buffer.

    Least-squares slope of py over the samples inside `window_s` of the
    newest timestamp. The window is wide because the main loop's actual
    sample cadence is ~200-1000ms per tick (observed session 2026-06-09
    20:06 — fire-window diag showed 3-17 cycles per 3s), far slower than
    the 15-30ms the POLL_INTERVAL comment suggests; a 300ms window
    usually held a single sample and the helper always returned None.
    At 1.5s the window is still ~15% of the ~10s bob period, so the
    slope is a meaningfully local velocity.

    Requires at least 2 samples spanning at least `min_span_s` — slopes
    over shorter spans are dominated by the ±1-2px detector jitter.
    """
    if not samples:
        return None
    t_end = samples[-1][0]
    recent = [(t, py) for t, _, py in samples if t_end - t <= window_s]
    if len(recent) < 2 or (recent[-1][0] - recent[0][0]) < min_span_s:
        return None
    n = len(recent)
    mean_t = sum(t for t, _ in recent) / n
    mean_y = sum(y for _, y in recent) / n
    denom = sum((t - mean_t) ** 2 for t, _ in recent)
    if denom == 0:
        return None
    slope = sum((t - mean_t) * (y - mean_y) for t, y in recent) / denom
    return slope


# Sweep used to continue -48, 48, -64, 64, -80, 80 — trimmed 2026-06-09
# (#37): across 269 logged shots every perturbed shot beyond ±32 landed
# >120px from the hoop with zero makes. Combined with σ scaling (up to
# 3×) the old tail produced ±240px swings — guaranteed misses.
PERTURBATION_SEQUENCE = [
    0, -8, 8, -16, 16, -24, 24, -32, 32,
]

# Hard cap on |perturbation| after σ scaling. shots.db shows landing
# position responds incoherently to offset changes beyond ~±32px (#37);
# revisit once platform_vy data settles the velocity question.
PERTURBATION_MAX = 32

# A miss whose ball still arrived within this many px of hoop_x was a
# launch/rim outcome, not an aim error — shots arriving in this band make
# at 64%, so the sweep must NOT step away from a target that produced one.
IN_BAND_RESIDUAL_PX = 60


def _perturbation_for(miss_count: int, sigma: float | None = None) -> int:
    """Pixels to add to the predicted offset on the (miss_count+1)-th shot at
    a given hoop. miss_count=0 → no perturbation; subsequent misses cycle
    through the sweep, with magnitudes scaled by predictor uncertainty so a
    high-σ (sparse-region) prediction takes bigger steps to reach the make
    zone before lives run out. The scaled value is clamped to
    ±PERTURBATION_MAX.

    σ buckets chosen 2026-05-23 from a (710, 415) failure mode where the
    predictor was ~12px off and ±8 perturbations were too narrow to walk
    there in the 2-life window. Thresholds rough — tighten when we have
    more σ data per outcome.
    """
    # Past the end of the sweep, cycle through it again rather than pin at
    # the last entry — pinning fired 25 IDENTICAL +32 shots at one hoop
    # (session 2026-06-09 20:43) where the outcome was pure velocity luck.
    # Cycling keeps sampling the whole neighborhood, which both varies the
    # bob phase the platform crosses each target at and feeds the predictor
    # diverse rows instead of duplicates.
    base = PERTURBATION_SEQUENCE[miss_count % len(PERTURBATION_SEQUENCE)]
    if sigma is None or sigma < 10:
        scale = 1.0
    elif sigma < 20:
        scale = 1.5
    elif sigma < 40:
        scale = 2.0
    else:
        scale = 3.0
    scaled = int(round(base * scale))
    return max(-PERTURBATION_MAX, min(PERTURBATION_MAX, scaled))


def _shot_arrival_x(
    ball_x_at_rim: int | None,
    peak_x: int | None,
    landing_x: int | None,
) -> int | None:
    """Best estimate of where the shot arrived at the hoop, bounce-aware.

    When the ball deflects backward off the hoop structure (peak_x well
    past landing_x), ball_x_at_rim_height catches the post-bounce descent
    and reads as a huge short miss. Observed session 2026-06-09 20:59:
    9 of 14 shots at hoop (579,350) had peak_x within 26px of hoop_x —
    on-target clanks — but rim-height residuals of -111..-325 drove the
    sweep away from correct aim. The rightmost x the ball reached is the
    honest arrival measure for those; for unobstructed flights
    (peak ≈ landing) the rim-height crossing stays the better estimate.
    """
    if (
        peak_x is not None
        and landing_x is not None
        and (peak_x - landing_x) > MAX_BACK_DRIFT_PX
    ):
        return peak_x
    return ball_x_at_rim


# An in-band miss is worth re-firing (rim luck converts at ~64%), but only
# a couple of times: at clank-prone hoops the flat-arc arrival reads
# "in-band" every single shot and never drops — session 2026-06-09 21:23
# re-fired the same target through 20 holds across a 54-shot loop at
# (691,550) before the user killed it. After this many consecutive holds
# without a make, assume the geometry (not luck) is the problem and let
# the sweep move.
MAX_CONSECUTIVE_HOLDS = 2


def _sweep_should_advance(arrival_x: int | None, hoop_x: int | None) -> bool:
    """After a miss, decide whether the perturbation sweep steps to the
    next entry. False when the ball arrived within IN_BAND_RESIDUAL_PX of
    hoop_x: the aim was right and the miss was rim luck, so the next shot
    re-fires the same target. `arrival_x` should come from
    _shot_arrival_x so structure clanks aren't misread as short misses.
    No trajectory data → advance (can't tell a mislaunch from a
    near-miss, and standing still forever on an unmeasured target risks
    a stuck loop).
    """
    if arrival_x is None or hoop_x is None:
        return True
    return abs(arrival_x - hoop_x) > IN_BAND_RESIDUAL_PX


def _explore_target(
    ys: list[int],
    margin: int = PERTURBATION_BOB_MARGIN,
    rng=random,
) -> int | None:
    """Uniform-random target inside the observed bob range (margin-inset),
    or None when the range is degenerate.

    Free-shot exploration: while the pre-game "Make a shot to start"
    prompt is up, misses cost nothing — so after the first
    (predictor-aimed) free shot misses, the bot samples targets across
    the whole bob range instead of grinding the predictor's ±32px
    neighborhood. Each sample fires at a different bob phase, sweeping
    (platform_y, platform_vy) space and feeding the make-probability
    model (#37) far faster than live play can.
    """
    lower = min(ys) + margin
    upper = max(ys) - margin
    if upper <= lower:
        return None
    return rng.randint(lower, upper)


def _sweep_step(
    arrival_x: int | None,
    hoop_x: int | None,
    consecutive_holds: int,
) -> tuple[bool, int]:
    """Combine the in-band hold rule with the hold cap. Returns
    (advance, new_consecutive_holds)."""
    if (
        _sweep_should_advance(arrival_x, hoop_x)
        or consecutive_holds >= MAX_CONSECUTIVE_HOLDS
    ):
        return True, 0
    return False, consecutive_holds + 1


# Accepted window around target_y (pixels) when deciding to fire. Was 2,
# bumped to 6 — at 2, the in_window branch almost never fires and we rely
# entirely on "crossed", which catches the cross between samples and
# reports a sample-after-cross platform_y that's variably 0-20px past
# target. Wider in_window catches more shots near target with consistent
# platform_y, reducing variance in fire timing.
Y_TOLERANCE = 6

# Which predictor implementation to use for the make-zone target_y:
#   "knn"             — KNN over past makes, inverse-distance weighted
#                       (k=3). Adapts to local curvature; recommended
#                       baseline.
#   "bivariate"       — linear OLS for target_y = a*hoop_y + b*hoop_x + c.
#                       Faster but biased in regions far from the
#                       training-data cluster (needed hardcoded overrides
#                       historically).
#   "gp"              — Gaussian Process regression (sklearn, anisotropic
#                       RBF). Same interface as KNN/bivariate via
#                       predict(); also exposes predict_with_std() so
#                       future logic can gate or perturb based on
#                       uncertainty. See docs/predictors.md.
#   "trajectory_knn"  — 3D KNN on (hoop_y, hoop_x, platform_y) → ball_
#                       landing_x, fit on every shot (make or miss). At
#                       predict time scans candidate platform_y values
#                       and picks the one whose modelled landing_x is
#                       closest to hoop_x. Lets the predictor learn from
#                       misses as well as makes. See GitHub issue #17.
#   "trajectory_gp"   — Same trajectory-regression shape as
#                       trajectory_knn but uses a 3D GP for the landing
#                       surface. Smoother fit than KNN in sparse 3D
#                       regions; exposes posterior variance for future
#                       uncertainty-aware logic. See GitHub issue #21.
#   "trajectory_rf"   — Same shape with a sklearn random forest. Trees
#                       handle interactions naturally and are robust
#                       on noisy small data; complementary to GP
#                       (smooth) and KNN (local).
#
# The launcher's Bots tab exposes a Predictor dropdown that sets the
# HOOPS_PREDICTOR_KIND env var; this constant is the default when the
# env var is unset (e.g. running `uv run hoops` directly).
PREDICTOR_KIND = os.environ.get("HOOPS_PREDICTOR_KIND", "gp")

# Required direction of platform motion to fire. "up", "down", or "any".
# Back to "up" after dir=down sweep on hoop_y=448: offsets 60->10 all hit
# the front of the rim (5px short, plateaued). Hypothesis: launch inherits
# platform velocity, so dir=down imparts downward bias → flat arc. dir=up
# adds upward bias → higher arc → more time aloft → more horizontal range.
# Earlier "dir=up overshot" note was at hoop_y=337 (much higher hoop, easier
# to overshoot); at lower hoops it should land closer to right.
REQUIRED_DIRECTION = "up"

# Per-hoop direction override: at very low hoops, dir=up is PROVEN futile —
# a 56-shot exploration sweep at (691,550) (session 2026-06-09 21:41)
# covered the full bob range (platform_y 294-502, vy -175..+66) and every
# shot landed at least ~68px past the hoop; the only launch state that even
# reaches the hoop's x is the bob-bottom flat arc, which clanks off the
# structure. dir=down imparts downward bias → flatter, shorter arc — the
# 2026-05 dir=down sweep plateaued "5px short of the rim front" at
# hoop_y=448, and SHORTER is exactly what y>=530 hoops need. Threshold set
# from the sessions where up was mapped futile (hoop_y 534-550); the
# 480-529 band stays on "up" until data says otherwise.
LOW_HOOP_DOWN_THRESHOLD = 530

# The near "clank band" (x<=640) briefly fired dir=down too
# (2026-06-09 b757b85, reverted 2026-06-10): a 39-shot exploration sweep
# at (631,326) reached the hoop from the whole bob range without one
# make (peaks 531-733, 0/39), and the all-time record says dir=up DOES
# make in the band (~18%, 36 makes incl. at (630,325)/(634,322)) — the
# 2/36 that justified the override was the vy-instrumented slice only.
# Band clanks are probabilistic under dir=up, not categorical; that's
# #38's job, with both direction curves now mapped for free.


def _required_direction_for(hoop_y: int, hoop_x: int) -> str:
    """Direction policy: dir=down at very low hoops (exhaustively
    validated in both directions — see LOW_HOOP_DOWN_THRESHOLD), the
    default direction everywhere else."""
    if hoop_y >= LOW_HOOP_DOWN_THRESHOLD:
        return "down"
    return REQUIRED_DIRECTION

# Click position is fixed at the hoop (hoop_x, hoop_y). Earlier experiments
# (May 3 click_sweep + click_extreme) confirmed click position has no
# measurable effect on ball trajectory — the only thing that matters is
# the platform's position at click time (driven by the offset). Keeping
# click-at-hoop as a sane default; could click anywhere with the same
# result.

# Wait after clicking so the ball can travel, land, and the score animation
# completes. Was 2.0 — but observed ball arrival at the rim is ~2.9s and the
# score updates after that. With 2.0s the post_shot snapshot consistently
# captured the OLD score, so every make was logged as miss (the next shot's
# pre_shot would show the updated score, but we don't compare across shots).
POST_SHOT_COOLDOWN = 4.0

# Monitor mode: per-shot subfolder under assets/monitor/ with pre/post-shot
# screenshots, all flight frames captured during the post-shot cooldown (so
# we can see the full ball flight for offline review and ball-template
# extraction), and a meta.txt with shot details. Heavyweight (~200KB per
# shot) but invaluable for tuning offline. Toggle off in production.
MONITOR_MODE = True
MONITOR_DIR = _HERE / "assets" / "monitor"

# Capture flight frames during the post-shot cooldown. Used to record the
# actual ball trajectory for offline offset tuning and the trajectory
# predictors.
MONITOR_FLIGHT = True
FLIGHT_POLL = 0.05

# Score region is loaded fresh from regions.json each shot (using current
# window dims) so it survives the user resizing the game window between runs.
# Pick via hoops-pick-score-region (writes regions.json directly).


def _capture_score_region(left: int, top: int, width: int, height: int):
    region = get_region(_HERE, "score", width, height)
    if region is None:
        return None
    frame = grab_region(left, top, width, height)
    return score_region(
        frame,
        region["left"],
        region["top"],
        region["width"],
        region["height"],
    )


def _capture_lives_region(left: int, top: int, width: int, height: int):
    region = get_region(_HERE, "lives", width, height)
    if region is None:
        return None
    frame = grab_region(left, top, width, height)
    return score_region(  # same crop+gray helper, name unfortunate
        frame,
        region["left"],
        region["top"],
        region["width"],
        region["height"],
    )


# Bumped from 60 → 80 (2026-05-09) after a confirmed bank-shot make at
# hoop=(674,447) was rejected with ball_x_at_rim = 750 (76 px past
# hoop_x). Swishes land ~30–40 px from hoop_x, so 60 was tuned for the
# swish case; bank shots routinely come down 60–80 px off-center but
# still drop through. 80 matches CLEAN_MAKE_TOLERANCE in
# the hoops shot log — the in-bot check and the post-hoc clean-make
# classifier now use the same threshold. Closes #9.
TRAJECTORY_MAKE_TOLERANCE = 80  # px


def _log_shot_result(
    stats: dict,
    before,
    after,
    score_before_int: int | None = None,
    score_after_int: int | None = None,
    ball_x_at_rim: int | None = None,
    hoop_x: int | None = None,
    prompt_disappeared: bool = False,
    prompt_visible_before: bool = False,
) -> tuple[bool | None, float | None, int | None]:
    """Print the score diff line and update stats. Returns
    (changed, diff, computed_increment) or (None, None, None) if either
    score snapshot was missing.

    Make detection now anchors on the running session score (in `stats`,
    keys "session_score") rather than requiring a per-shot OCR'd `sb`.
    Pre-shot OCR fails surprisingly often (animation residuals); but
    post-shot OCR + the previous shot's confirmed `sa` is enough to
    derive the increment.

    Rules:
      - If `score_after_int` is None: count as miss (we can't confirm).
      - If `score_after_int <= session_score`: miss (no progress).
      - If `score_after_int - session_score` is 1 or 2 AND trajectory
        plausible: MAKE (+1 or +2). Update session_score.
      - If increment is >2 (e.g. OCR misread "8" as "17"): reject as
        misread — almost certainly a tesseract error, since you can't
        score >2 in a single shot.
    """
    if before is None or after is None:
        return None, None, None
    _, diff = score_changed(before, after)
    if prompt_visible_before:
        # The "Make a shot to start" prompt is up — the game hasn't
        # started, so the score is definitionally 0 and any pre-shot OCR
        # read is noise. Trusting it poisons the anchor for the whole
        # game: observed 2026-06-09 20:13, a pre-read of 7 on a fresh
        # game made the real post-shot score of 1 look like increment
        # -6 on every later shot, so no make could ever be detected.
        stats["session_score"] = 0
    elif score_before_int is not None and score_before_int > stats.get("session_score", 0):
        # Re-anchor session_score from this shot's pre-OCR if it's higher
        # than what we knew. Catches off-bot scoring (user manually clicks
        # the game between shots, mid-session) so we don't credit the bot
        # for the user's points on the next shot.
        stats["session_score"] = score_before_int
    session_score = stats.get("session_score", 0)
    inferred_increment: int | None = None
    if score_after_int is not None:
        inferred_increment = score_after_int - session_score

    plausible_inc = inferred_increment is not None and 1 <= inferred_increment <= 2
    trajectory_plausible = (
        ball_x_at_rim is None
        or hoop_x is None
        or abs(ball_x_at_rim - hoop_x) <= TRAJECTORY_MAKE_TOLERANCE
    )
    # Strict per-shot make credit (this is what the predictor sees).
    # `prompt_disappeared` is the strongest signal — the in-game "Make a
    # shot to start" prompt only clears on a real make, so it overrides
    # the trajectory check on the first shot of a fresh game. Useful when
    # bank shots come down further off-center than TRAJECTORY_MAKE_TOLERANCE
    # allows, or when ball_x_at_rim measured the upward crossing
    # (overshoot pattern from issue #17 prereqs).
    changed = plausible_inc and (trajectory_plausible or prompt_disappeared)

    # Always advance the session score anchor when sa is higher — even if
    # the per-shot increment is too big to attribute to this single shot.
    # This way the live score display matches the in-game score, and
    # subsequent shots get accurate increment math, even when OCR dropped
    # out for some shots in between.
    if score_after_int is not None and score_after_int > session_score:
        stats["session_score"] = score_after_int

    stats["attempts"] += 1
    if changed:
        stats["makes"] += 1
        if prompt_disappeared and not trajectory_plausible:
            label = f"MAKE (+{inferred_increment}) [prompt cleared — bank shot accepted]"
        else:
            label = f"MAKE (+{inferred_increment})"
    elif plausible_inc and not trajectory_plausible:
        label = "miss [OCR said make but ball overshot]"
    elif inferred_increment is not None and inferred_increment > 2:
        label = f"miss [OCR jump +{inferred_increment} — multiple makes lost to OCR drop]"
    else:
        label = "miss"
    sa_str = f", sa={score_after_int}" if score_after_int is not None else ", sa=?"
    print(f"  [score] {label} (diff={diff:.1f}{sa_str}, score={stats['session_score']}) | confident makes {stats['makes']}/{stats['attempts']}")
    return changed, diff, inferred_increment


def _classify_clean_make(
    made: bool | None,
    landing: int | None,
    peak_x: int | None,
    hoop_x: int,
) -> tuple[int | None, str | None]:
    """Classify a shot's clean_make value for predictor training.

    A make is "clean" only if the trajectory says the ball flew to the
    hoop on aim: landing close to hoop_x, and no large backward drift
    (peak_x far past landing_x = pole/backboard ricochet). A make that
    fails these still counts for score — it just doesn't teach the
    predictor, because the platform_y that produced it steered the ball
    somewhere else and luck brought it in.

    Prompt-confirmed makes get no exemption (changed 2026-06-09, #37).
    The prompt clearing is ground truth that the shot SCORED, but says
    nothing about whether the aim was right: a first-shot pole-tip
    ricochet (peak_x 252px past landing, session 20:13 shot 1) would
    have trained the predictor that the pole-hit zone is a make target.
    Bank shots with modest drift pass the filters on their own merits.

    No landing data → trust the make (nothing to judge against).
    Returns (clean_value, reject_reason); reason is non-None only when
    a make was rejected.
    """
    if made is None:
        return None, None
    if not made:
        return 0, None
    if landing is None:
        return 1, None
    if abs(landing - hoop_x) > CLEAN_MAKE_TOLERANCE:
        return 0, (
            f"ball_landing_x={landing} vs hoop_x={hoop_x} "
            f"(Δ={abs(landing - hoop_x)}px > {CLEAN_MAKE_TOLERANCE}px) — likely rattle-in"
        )
    if peak_x is not None and (peak_x - landing) > MAX_BACK_DRIFT_PX:
        return 0, (
            f"ball_peak_x={peak_x} vs ball_landing_x={landing} "
            f"(drift={peak_x - landing}px > {MAX_BACK_DRIFT_PX}px) — backboard/rim bounce"
        )
    return 1, None


# The hoop only respawns (teleports to a new position) on a make. When the
# newly detected hoop is far from the one the last shot was fired at, but
# that shot was logged a miss, the log is wrong — OCR dropped out (observed
# twice in three sessions, 2026-06-09: a real make logged as miss because
# score_after OCR returned None). The respawn is near-ground-truth; the
# guards below keep the two known false-positive sources out:
RESPAWN_MIN_MOVE_PX = 30   # score>=10 hoop drift is gradual; a respawn
                           # teleports. Detector jitter is ±2px.
RESPAWN_MIN_CONF = 0.9     # don't trust a marginal rim match (extreme low
                           # spawns near the EXIT button match ~0.7) to
                           # rewrite history.
RESPAWN_MAX_SCORE = 10     # at 10+ the hoop drifts between shots by design;
                           # disable the signal entirely there.


def _respawn_implies_make(
    prev_key: tuple[int, int] | None,
    new_key: tuple[int, int],
    last_made,
    session_score: int,
    new_conf: float,
) -> bool:
    """True when a newly detected hoop position proves the previous shot
    (logged as miss / unknown) actually made. See guard rationale above."""
    if prev_key is None or last_made:
        return False
    if session_score >= RESPAWN_MAX_SCORE:
        return False
    if new_conf < RESPAWN_MIN_CONF:
        return False
    dx = abs(new_key[0] - prev_key[0])
    dy = abs(new_key[1] - prev_key[1])
    return max(dx, dy) > RESPAWN_MIN_MOVE_PX


# Backstop for any fire-gate deadlock: if a target has been set this long
# without a shot, drop it and re-detect/recompute. The known deadlock is a
# target at the exact bob extreme on the session's first hoop (see
# _retarget_within_bob), but the watchdog also catches modes we haven't met.
STALE_TARGET_TIMEOUT_S = 60.0


def _retarget_within_bob(
    target_y: int,
    ys: list[int],
    margin: int = PERTURBATION_BOB_MARGIN,
) -> int:
    """Pull a target that sits inside the observed bob range but within
    `margin` of an extreme back to margin-distance from that extreme.

    At the extremes the platform is at its turnaround: it never crosses
    the target (no sign-flip trigger) and the direction history straddles
    the reversal, reading flat/down — so a dir=up gate never opens.
    Observed 2026-06-09 20:37: first hoop of the session, target_y=509 =
    exact bob bottom, bot hovered >60s without firing. The detection-time
    cap can't cover that case (no bob samples exist yet at first
    detection).

    Targets OUTSIDE the range are left alone — the clamp path handles
    those with a direction bypass and widened tolerance.
    """
    ymin, ymax = min(ys), max(ys)
    upper, lower = ymax - margin, ymin + margin
    if upper < lower or not (ymin <= target_y <= ymax):
        return target_y
    return min(max(target_y, lower), upper)


def _make_monitor_dir(throw_idx: int) -> Path:
    return make_shot_dir(MONITOR_DIR, throw_idx, prefix="shot")


def _direction(history: deque) -> str:
    if len(history) < 2:
        return "any"
    delta = history[-1] - history[0]
    if delta < -1:
        return "up"  # screen y decreases upward
    if delta > 1:
        return "down"
    return "flat"


@dataclass
class _HoopTracking:
    """Per-hoop state: which hoop we're shooting at and how the
    miss-driven sweep is doing there. Reset whenever the hoop moves."""
    key: tuple[int, int] | None = None
    misses: int = 0   # sweep position (advances on way-off misses)
    holds: int = 0    # consecutive in-band re-fires since the sweep moved
    shots: int = 0    # total fired at this hoop (gates free-shot exploration)

    def reset_for_new_hoop(self, key: tuple[int, int] | None) -> None:
        self.key = key
        self.misses = 0
        self.holds = 0
        self.shots = 0


@dataclass
class _Target:
    """Everything _select_target decided for the current hoop."""
    target_y: int
    offset: int
    perturbation: int
    predicted_offset: int
    target_source: str   # "predictor" | "sweep" | "explore"
    required_dir: str
    predictor_fit: bool  # whether a fitted predictor (vs cold start) aimed this


def _maybe_retro_make(
    shot_db,
    shot_stats: dict,
    last_shot: dict | None,
    prev_key: tuple[int, int] | None,
    new_key: tuple[int, int],
    hoop_conf: float,
) -> dict | None:
    """Hoop moved: if the respawn proves the last logged shot actually
    made (see _respawn_implies_make), retro-correct its row and the
    session stats. Returns the new last_shot value (None once consumed)."""
    if last_shot is None or not _respawn_implies_make(
        prev_key, new_key, last_shot["made"],
        shot_stats.get("session_score", 0), hoop_conf,
    ):
        return last_shot
    clean_value, _ = _classify_clean_make(
        True, last_shot["landing"], last_shot["peak"], last_shot["hoop_x"],
    )
    set_make(shot_db, last_shot["id"], 1, clean_value, "respawn")
    shot_stats["makes"] += 1
    # Keep the anchor in step with the real score so the NEXT make's
    # increment doesn't get double-attributed — unless post-shot OCR
    # already advanced it (make was rejected on other grounds but the
    # score was seen).
    if not last_shot["anchor_advanced"]:
        shot_stats["session_score"] += 1
    print(
        f"  [respawn] hoop {last_shot['hoop_key']} -> {new_key} after an unconfirmed shot — "
        f"retro-marking shot id={last_shot['id']} as MAKE (clean={clean_value}) | "
        f"confident makes {shot_stats['makes']}/{shot_stats['attempts']}"
    )
    return None


def _select_target(
    frame,
    hoop_x: int,
    hoop_y: int,
    hoop_conf: float,
    predictors: dict,
    tracking: _HoopTracking,
    range_samples,
) -> _Target:
    """Pick this hoop's firing direction, predictor, and target_y —
    predictor aim plus the miss-driven sweep, overridden by free-shot
    exploration while the start prompt is up, then capped inside the
    observed bob range."""
    required_dir = _required_direction_for(hoop_y, hoop_x)
    predictor = predictors.get(required_dir)
    base_offset = _compute_offset(hoop_y, hoop_x, predictor)
    predicted_offset_value = _predicted_offset(hoop_y, hoop_x, predictor)
    predictor_sigma = _predicted_std(hoop_y, hoop_x, predictor)
    perturbation = _perturbation_for(tracking.misses, predictor_sigma)
    offset = base_offset + perturbation
    target_y = hoop_y + offset
    target_source = "sweep" if perturbation else "predictor"
    # Free-shot exploration: pre-game misses cost no lives, so once the
    # predictor-aimed first shot has missed, sample targets across the
    # whole bob range instead of grinding the predictor's neighborhood —
    # every free shot becomes a (platform_y, vy) → outcome training row
    # (#37). The prompt check costs one template match, only on shots
    # where the target is being (re)computed.
    if (
        tracking.shots >= 1
        and len(range_samples) >= 40
        and find_game_prompt(frame)[0]
    ):
        explore_y = _explore_target([p[2] for p in range_samples])
        if explore_y is not None:
            target_y = explore_y
            offset = target_y - hoop_y
            perturbation = offset - predicted_offset_value
            target_source = "explore"
    # Cap target_y inside the platform's observed bob range with a small
    # margin so the platform passes through (rather than only kissing at
    # the apex). Without this, a perturbation that lands target_y exactly
    # at ymin/ymax produces a stuck loop: the dir=up + position-matched
    # sample is ephemeral at the bob extreme, and the bot spins waiting
    # for it (observed 2026-05-09 in session 20:18:44, perturb +16 →
    # target=510, bob ymax=510, never fired).
    cap_tag = ""
    if len(range_samples) >= 40:
        ys_seen = [p[2] for p in range_samples]
        ymin_seen, ymax_seen = min(ys_seen), max(ys_seen)
        upper = ymax_seen - PERTURBATION_BOB_MARGIN
        lower = ymin_seen + PERTURBATION_BOB_MARGIN
        if upper >= lower and target_y > upper:
            cap_tag = f" [capped target_y {target_y}->{upper}, bob ymax={ymax_seen}]"
            target_y = upper
            offset = target_y - hoop_y
        elif upper >= lower and target_y < lower:
            cap_tag = f" [capped target_y {target_y}->{lower}, bob ymin={ymin_seen}]"
            target_y = lower
            offset = target_y - hoop_y
    if target_source == "explore":
        tag = f", explore (free shot, {tracking.shots} fired at this hoop)"
    else:
        tag = f", perturb={perturbation:+d} (miss #{tracking.misses})" if perturbation else ""
    override_tag = f" [override: predicted={predicted_offset_value}]" if base_offset != predicted_offset_value else ""
    sigma_tag = f" (σ={predictor_sigma:.1f})" if predictor_sigma is not None else ""
    dir_tag = f", dir={required_dir}" if required_dir != REQUIRED_DIRECTION else ""
    print(f"Hoop rim at ({hoop_x},{hoop_y}) (conf={hoop_conf:.2f}){sigma_tag}, offset={offset}{tag}{override_tag}{cap_tag}{dir_tag}, target launch y={target_y}")
    return _Target(
        target_y=target_y,
        offset=offset,
        perturbation=perturbation,
        predicted_offset=predicted_offset_value,
        target_source=target_source,
        required_dir=required_dir,
        predictor_fit=predictor is not None,
    )


@dataclass
class _FiredShot:
    """Everything known at fire time about a shot that was just clicked.
    _record_shot_outcome fills in the rest (score, trajectory, verdict)."""
    left: int
    top: int
    width: int
    height: int
    shot_idx: int
    fired_at: str
    time_since_last_shot_ms: int | None
    prompt_visible_before: bool
    score_before: object
    lives_before: object
    shot_dir: Path | None
    hoop_x: int
    hoop_y: int
    hoop_conf: float
    hoop_scale: float
    px: int
    py: int
    clamped: bool
    eff_target_y: int
    direction: str
    target: _Target


def _record_shot_outcome(
    shot: _FiredShot,
    shot_db,
    shot_stats: dict,
    tracking: _HoopTracking,
    range_samples,
    session_started: str,
    code_commit: str | None,
) -> dict:
    """Post-click bookkeeping: capture the flight, wait out the cooldown,
    read the score, classify the outcome, persist the row, and update the
    per-hoop sweep state. Returns the last_shot record the respawn
    corrector needs. The click has already happened — nothing here is
    latency-sensitive."""
    left, top, width, height = shot.left, shot.top, shot.width, shot.height
    if MONITOR_FLIGHT and shot.shot_dir is not None:
        flight_deadline = time.time() + POST_SHOT_COOLDOWN
        flight_idx = 0
        while time.time() < flight_deadline:
            check_failsafe()
            flight_idx += 1
            f = grab_region(left, top, width, height)
            bgr = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(str(shot.shot_dir / f"flight_{flight_idx:03d}.png"), bgr)
            time.sleep(FLIGHT_POLL)
    else:
        time.sleep(POST_SHOT_COOLDOWN)
    score_after = _capture_score_region(left, top, width, height)
    lives_after = _capture_lives_region(left, top, width, height)
    # Re-check the prompt only if it was up before the shot — cheap to
    # skip the post grab + template match entirely when there's no
    # signal to derive.
    prompt_disappeared = False
    if shot.prompt_visible_before:
        post_full_frame = grab_region(left, top, width, height)
        prompt_disappeared = not find_game_prompt(post_full_frame)[0]
    # OCR the actual score numbers — used as the primary make signal
    # (the diff-based heuristic was noisy on the wider score region).
    # None if tesseract binary isn't installed. `score_increment` here is
    # just the per-shot pre→post diff for diagnostic logging; the actual
    # make-detection logic in _log_shot_result uses the running
    # session_score anchor, which is much more robust to pre-shot OCR
    # failures.
    score_before_int = read_score(shot.score_before) if shot.score_before is not None else None
    score_after_int = read_score(score_after) if score_after is not None else None
    # Ball trajectory metrics from the captured flight frames — done
    # BEFORE _log_shot_result so we can cross-check OCR against physics:
    # a "make" with the ball flying 100+ px past the hoop is almost
    # certainly a tesseract misread.
    trajectory: dict = {
        "ball_apex_y": None,
        "ball_peak_x": None,
        "ball_x_at_rim_height": None,
        "ball_landing_x": None,
        "ball_flight_ms": None,
    }
    if shot.shot_dir is not None and MONITOR_FLIGHT:
        try:
            trajectory = analyse_shot_dir(shot.shot_dir, shot.hoop_x, shot.hoop_y)
        except Exception as e:
            print(f"  [trajectory] analysis failed (non-fatal): {e}")
    anchor_before = shot_stats.get("session_score", 0)
    made, score_diff, score_increment = _log_shot_result(
        shot_stats, shot.score_before, score_after,
        score_before_int=score_before_int,
        score_after_int=score_after_int,
        ball_x_at_rim=trajectory["ball_x_at_rim_height"],
        hoop_x=shot.hoop_x,
        prompt_disappeared=prompt_disappeared,
        prompt_visible_before=shot.prompt_visible_before,
    )
    # Clean-make filter for predictor training — see _classify_clean_make.
    # The prompt-disappeared signal deliberately plays no role here: it
    # confirms the MAKE (handled in _log_shot_result), not the aim.
    clean_make_value, clean_reject_reason = _classify_clean_make(
        made,
        trajectory.get("ball_landing_x"),
        trajectory.get("ball_peak_x"),
        shot.hoop_x,
    )
    if clean_reject_reason:
        print(f"  [clean] make rejected for predictor: {clean_reject_reason}")
    # Compute lives_diff up-front so we can persist it on the shot row.
    # Only meaningful when the lives region is visibly populated (during
    # pre-game or "Make a shot to start" screens it's blank, std() near
    # zero).
    lives_diff_value: float | None = None
    if shot.lives_before is not None and lives_after is not None:
        if float(shot.lives_before.std()) >= 5.0 or float(lives_after.std()) >= 5.0:
            _, lives_diff_value = score_changed(shot.lives_before, lives_after)
    # Capture bob range from the recent platform samples for #16 and the
    # bob period for #13.
    bob_ymin = bob_ymax = None
    if len(range_samples) >= 10:
        ys_at_fire = [p[2] for p in range_samples]
        bob_ymin = int(min(ys_at_fire))
        bob_ymax = int(max(ys_at_fire))
    bob_period_ms_value = _estimate_bob_period_ms(list(range_samples))
    # Velocity at fire: range_samples hasn't been appended to since the
    # sample that triggered the shot, so the window ends exactly at fire
    # time.
    platform_vy_value = _platform_velocity(list(range_samples))
    shot_row_id = log_shot(
        shot_db,
        session_started=session_started,
        shot_idx=shot.shot_idx,
        fired_at=shot.fired_at,
        hoop_x=shot.hoop_x,
        hoop_y=shot.hoop_y,
        hoop_conf=float(shot.hoop_conf),
        platform_x=int(shot.px),
        platform_y=int(shot.py),
        offset=int(shot.target.offset),
        target_y=int(shot.target.target_y),
        eff_target_y=int(shot.eff_target_y),
        clamped=int(bool(shot.clamped)),
        direction=shot.direction,
        required_direction=shot.target.required_dir,
        score_diff=float(score_diff) if score_diff is not None else None,
        made=int(bool(made)) if made is not None else None,
        shot_dir=str(shot.shot_dir) if shot.shot_dir is not None else None,
        perturbation=int(shot.target.perturbation),
        lives_diff=lives_diff_value,
        ball_apex_y=trajectory["ball_apex_y"],
        ball_peak_x=trajectory["ball_peak_x"],
        ball_x_at_rim_height=trajectory["ball_x_at_rim_height"],
        ball_landing_x=trajectory["ball_landing_x"],
        window_w=int(width),
        window_h=int(height),
        score_before_int=score_before_int,
        score_after_int=score_after_int,
        score_increment=score_increment,
        clean_make=clean_make_value,
        predicted_offset=int(shot.target.predicted_offset),
        code_commit=code_commit,
        predictor_kind=PREDICTOR_KIND if shot.target.predictor_fit else None,
        rim_match_scale=float(shot.hoop_scale) if shot.hoop_scale else None,
        time_since_last_shot_ms=shot.time_since_last_shot_ms,
        ball_flight_ms=trajectory.get("ball_flight_ms"),
        bob_ymin=bob_ymin,
        bob_ymax=bob_ymax,
        bob_period_ms=bob_period_ms_value,
        platform_vy=platform_vy_value,
        prompt_up=int(bool(shot.prompt_visible_before)),
        target_source=shot.target.target_source,
    )
    # Update perturbation tracking. Made → reset (next hoop will be in a
    # new position anyway). Miss → advance the sweep, unless the ball
    # arrived in-band (aim was right; re-fire the same target rather
    # than stepping away from it).
    if made:
        tracking.reset_for_new_hoop(None)
    else:
        arrival_x = _shot_arrival_x(
            trajectory["ball_x_at_rim_height"],
            trajectory["ball_peak_x"],
            trajectory["ball_landing_x"],
        )
        advance, tracking.holds = _sweep_step(arrival_x, shot.hoop_x, tracking.holds)
        if advance:
            tracking.misses += 1
        else:
            resid = arrival_x - shot.hoop_x
            bounce_tag = (
                " [bounce-aware: peak_x]"
                if arrival_x == trajectory["ball_peak_x"]
                and arrival_x != trajectory["ball_x_at_rim_height"]
                else ""
            )
            print(f"  [perturb] miss but ball arrived {resid:+d}px from hoop_x{bounce_tag} — re-firing same target (hold {tracking.holds}/{MAX_CONSECUTIVE_HOLDS})")
    # Print the lives tick if it happened (signal for bot watching).
    if lives_diff_value is not None and lives_diff_value > 3:
        print(f"  [lives] counter ticked down (diff={lives_diff_value:.1f})")
    if shot.shot_dir is not None:
        post_frame = grab_region(left, top, width, height)
        save_frame(shot.shot_dir / "post_shot.png", post_frame)
        save_meta(
            shot.shot_dir / "meta.txt",
            hoop=f"({shot.hoop_x},{shot.hoop_y})",
            platform=f"({shot.px},{shot.py})",
            offset=shot.target.offset,
            target_y=shot.target.target_y,
            eff_target_y=shot.eff_target_y,
            clamped=shot.clamped,
            direction=shot.direction,
        )
    # Remember the row so a hoop respawn before the next shot can
    # retroactively correct an OCR-missed make.
    return {
        "id": shot_row_id,
        "made": made,
        "landing": trajectory["ball_landing_x"],
        "peak": trajectory["ball_peak_x"],
        "hoop_x": shot.hoop_x,
        "hoop_key": (shot.hoop_x, shot.hoop_y),
        "anchor_advanced": shot_stats.get("session_score", 0) > anchor_before,
    }


REPO_ROOT = _HERE.parent.parent
SNAPSHOT_REL = "minigames/hoops/assets/shots_snapshot.json"


def run():
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        session_started = datetime.now().isoformat(timespec="seconds")
        code_commit = current_code_commit(REPO_ROOT)
        if code_commit:
            print(f"Code commit: {code_commit}")
        shot_db = open_db(SHOT_DB_PATH)
        try:
            # One predictor per firing direction: training rows are
            # direction-filtered (launch physics differ between up- and
            # down-crossing fires), and the direction policy is per-hoop
            # (see _required_direction_for). The "down" fit usually
            # cold-starts until low-hoop sessions accumulate dir=down rows.
            predictors = {
                d: _fit_predictor(shot_db, d) for d in ("up", "down")
            }
            _run_inner(session_started, shot_db, predictors, code_commit)
        finally:
            shot_db.close()
            _refresh_and_commit_snapshot()


def _fit_predictor(shot_db, direction: str):
    """Fit the PREDICTOR_KIND model on rows for one firing direction.
    Returns None (cold start) when too few samples exist."""
    if PREDICTOR_KIND in ("trajectory_knn", "trajectory_gp", "trajectory_rf"):
        # All trajectory predictors learn from every shot (make or miss)
        # and use the same input schema (4-tuple with arrival_x) and
        # envelope clamp from nearby makes.
        rows = fetch_clean_trajectories(shot_db, direction)
        makes = fetch_makes(shot_db, direction)
        if PREDICTOR_KIND == "trajectory_knn":
            predictor = fit_trajectory_knn(rows, makes=makes, k=5)
        elif PREDICTOR_KIND == "trajectory_gp":
            predictor = fit_trajectory_gp(rows, makes=makes)
        else:
            predictor = fit_trajectory_rf(rows, makes=makes)
    else:
        rows = fetch_makes(shot_db, direction)
        if PREDICTOR_KIND == "knn":
            # k=3 (was 5): smaller K is more local. With k=5 the
            # predictor over-smoothed in regions with sparse training
            # data — a single nearby make at hoop=(593,390) with
            # offset=78 got diluted to offset=23 by 4 distant
            # neighbours. k=3 lets the local data dominate more.
            predictor = fit_knn(rows, k=3)
        elif PREDICTOR_KIND == "bivariate":
            predictor = fit_bivariate(rows)
        elif PREDICTOR_KIND == "gp":
            predictor = fit_gp(rows)
        else:
            raise ValueError(f"Unknown PREDICTOR_KIND {PREDICTOR_KIND!r}")
    if predictor is None:
        print(f"No predictor fit ({PREDICTOR_KIND!r}, too few samples in dir={direction!r}); using cold-start offset={COLD_START_OFFSET}")
    else:
        sample_kind = "shots" if PREDICTOR_KIND in ("trajectory_knn", "trajectory_gp", "trajectory_rf") else "makes"
        print(f"{PREDICTOR_KIND.upper()} target predictor (dir={direction!r}, n={predictor.n} {sample_kind})")
    return predictor


def _refresh_and_commit_snapshot() -> None:
    """Regenerate shots_snapshot.json from the DB, then commit + push (if in
    window). Best-effort — failures are reported but don't propagate."""
    try:
        from scripts import dump_shots
        dump_shots.main()
    except Exception as e:
        print(f"  [snapshot] dump failed (non-fatal): {e}")
        return
    commit_file_if_changed(
        REPO_ROOT,
        SNAPSHOT_REL,
        "Hoops: refresh shots_snapshot.json (auto)",
    )
    # Surface a nudge if many sessions have been played since the last
    # code-level review — keeps human-in-the-loop without external comms.
    maybe_print_nag(REPO_ROOT, SHOT_DB_PATH, SNAPSHOT_REL)


def _run_inner(session_started: str, shot_db, predictors: dict, code_commit: str | None):
    print(f"Hoops bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    platform_history: deque[int] = deque(maxlen=5)
    target: _Target | None = None
    target_set_at: float = 0.0  # for the stale-target watchdog
    hoop_x: int | None = None
    hoop_y: int | None = None
    hoop_conf_last: float = 0.0  # carries to shot_log row
    hoop_scale_last: float = 0.0  # rim_match_scale, carries to shot_log row
    last_fired_at_ms: float | None = None  # for time_since_last_shot_ms
    # range_samples is (timestamp, px, py) — the timestamp lets us
    # estimate bob_period_ms via _estimate_bob_period_ms.
    hoop_missing_since: float | None = None  # for exit-when-stuck
    range_samples: deque[tuple[float, int, int]] = deque(maxlen=200)  # (ts, px, py) for range + period diagnostics
    last_range_log = time.time()
    # Fire-window diagnostics. Reset every range-log window so we see a rate
    # rather than a monotonically-growing total. y_open = py near target_y.
    cycles_window = 0
    y_open_window = 0
    prev_py: int | None = None
    shot_stats: dict = {"makes": 0, "attempts": 0, "session_score": 0}
    # Per-hoop sweep state — reset when the hoop position changes (after
    # a make, the game spawns a new hoop).
    tracking = _HoopTracking()
    # State of the most recent logged shot, kept so a hoop respawn can
    # retroactively correct a make the OCR failed to confirm.
    last_shot: dict | None = None

    while True:
        check_failsafe()
        try:
            left, top, width, height = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        frame = grab_region(left, top, width, height)

        # Stop cleanly when the trial ends.
        is_over, go_conf = find_game_over(frame)
        if is_over:
            print(f"Game over detected (conf={go_conf:.2f}). Final session: {shot_stats['makes']}/{shot_stats['attempts']} makes.")
            return

        if target is not None and time.time() - target_set_at > STALE_TARGET_TIMEOUT_S:
            print(
                f"  [stale] no shot for {STALE_TARGET_TIMEOUT_S:.0f}s at target_y={target.target_y} — "
                f"re-detecting hoop and recomputing target"
            )
            target = None

        if target is None:
            hoop_pos, hoop_conf, hoop_scale = find_rim(frame)
            if hoop_pos is None:
                if hoop_missing_since is None:
                    hoop_missing_since = time.time()
                    # Save the very first frame where the rim isn't found,
                    # for offline review. Goes to assets/diagnostics/.
                    diag_dir = _HERE / "assets" / "diagnostics"
                    diag_dir.mkdir(parents=True, exist_ok=True)
                    diag_path = diag_dir / f"missing_{datetime.now():%Y%m%d_%H%M%S}.png"
                    save_frame(diag_path, frame)
                    print(f"Saved diagnostic frame to {diag_path}")
                elapsed = time.time() - hoop_missing_since
                print(f"Rim not found (best confidence={hoop_conf:.2f}, missing for {elapsed:.0f}s)")
                # 5s (was 20): if the rim is gone for more than a few seconds,
                # we're almost certainly out of the minigame — either the
                # game-over screen is up (template may not match the higher-
                # score variant) or a stray click has dropped us back to the
                # world view. Either way, 20s of dead-time polling is wasted.
                # See diagnostics/missing_20260523_230553.png for the canonical
                # "back in world" failure that motivated tightening.
                if elapsed > 5:
                    print("Rim invisible for >5s — bailing out (likely game over or exited minigame).")
                    return
                time.sleep(0.2)
                continue
            hoop_missing_since = None
            hoop_x, hoop_y = hoop_pos
            hoop_conf_last = hoop_conf
            hoop_scale_last = hoop_scale
            # Reset miss tracking if the hoop has moved (i.e., last shot made
            # and a new hoop spawned). Use ±2px tolerance to absorb detector
            # jitter — a hoop "in the same place" matches if both x and y are
            # within a couple of pixels.
            new_key = (hoop_x, hoop_y)
            if tracking.key is None or abs(new_key[0] - tracking.key[0]) > 2 or abs(new_key[1] - tracking.key[1]) > 2:
                last_shot = _maybe_retro_make(
                    shot_db, shot_stats, last_shot, tracking.key, new_key, hoop_conf,
                )
                tracking.reset_for_new_hoop(new_key)
            target = _select_target(
                frame, hoop_x, hoop_y, hoop_conf, predictors, tracking, range_samples,
            )
            target_set_at = time.time()

        platform_pos, platform_conf = find_platform(frame)
        if platform_pos is None:
            time.sleep(POLL_INTERVAL)
            continue

        px, py = platform_pos
        platform_history.append(py)
        range_samples.append((time.time(), px, py))

        if time.time() - last_range_log > 3.0 and range_samples:
            xs = [p[1] for p in range_samples]
            ys = [p[2] for p in range_samples]
            y_pct = (100 * y_open_window / cycles_window) if cycles_window else 0
            print(
                f"  [diag] platform last 200 samples: x={min(xs)}..{max(xs)}, "
                f"y={min(ys)}..{max(ys)}, target_y={target.target_y} | "
                f"fire-window {cycles_window} cyc: y={y_pct:.0f}%"
            )
            last_range_log = time.time()
            cycles_window = 0
            y_open_window = 0

        # Re-apply the bob-margin cap now that bob samples exist. The cap at
        # hoop-detection time can't fire on the session's first hoop (no
        # samples yet) — a target at the exact bob extreme deadlocks the
        # fire gate (see _retarget_within_bob).
        if len(range_samples) >= 40:
            ys_seen = [p[2] for p in range_samples]
            new_target = _retarget_within_bob(target.target_y, ys_seen)
            if new_target != target.target_y:
                print(
                    f"  [cap] target_y {target.target_y} within {PERTURBATION_BOB_MARGIN}px of bob extreme "
                    f"({min(ys_seen)}..{max(ys_seen)}) -> {new_target}"
                )
                target.target_y = new_target
                target.offset = target.target_y - hoop_y

        # Clamp target_y to platform's observed reachable range. If the hoop
        # repositions outside the platform's bob, we still want to fire — even
        # a miss forces the hoop to reposition, breaking the deadlock.
        effective_target_y = target.target_y
        clamped = False
        if len(range_samples) >= 40:
            ys = [p[2] for p in range_samples]
            ymin, ymax = min(ys), max(ys)
            if target.target_y > ymax:
                effective_target_y = ymax
                clamped = True
            elif target.target_y < ymin:
                effective_target_y = ymin
                clamped = True

        # Either py is inside the tolerance window, OR the platform crossed the
        # target between this sample and the previous one (catches the fast
        # mid-bob region where ±tolerance is narrower than per-sample movement).
        # When clamped, target_y is unreachable; the platform only kisses
        # effective_target_y (its bob max) for a single sample per cycle and we
        # easily miss it with the tight tolerance — so widen specifically for
        # the clamped case.
        active_tolerance = max(Y_TOLERANCE, 6) if clamped else Y_TOLERANCE
        in_window = abs(py - effective_target_y) <= active_tolerance
        crossed = prev_py is not None and (
            (prev_py - effective_target_y) * (py - effective_target_y) < 0
        )
        # Fire-window diagnostic counter (per-cycle, reset each 3s log).
        cycles_window += 1
        if in_window or crossed:
            y_open_window += 1
            # Direction from the actual crossing if available, else from history.
            if crossed and prev_py is not None:
                direction = "up" if py < prev_py else "down"
            else:
                direction = _direction(platform_history)
            # When clamped, the platform only crosses the target at a bob extreme,
            # where direction is whatever-it-just-was → never matches. Bypass.
            direction_ok = clamped or target.required_dir == "any" or direction == target.required_dir
            if direction_ok:
                tag = " [crossed]" if crossed and not in_window else ""
                if clamped:
                    tag += " [clamped]"
                print(f"Platform at ({px},{py}) (target_y={target.target_y}, dir={direction}) — shooting{tag}")
                # Fire immediately. Bookkeeping (disk writes, extra grabs,
                # randomized delay) happens after the click — anything done
                # before it shifts platform_y away from what the cross-detector
                # just sampled, biasing every shot by tens of pixels.
                shot_idx = shot_stats["attempts"] + 1
                tracking.shots += 1
                fired_at = datetime.now().isoformat(timespec="seconds")
                fired_at_ms = time.time() * 1000.0
                time_since_last_shot_ms = (
                    int(fired_at_ms - last_fired_at_ms)
                    if last_fired_at_ms is not None else None
                )
                last_fired_at_ms = fired_at_ms
                # Detect the "Make a shot to start the game!" prompt
                # *before* firing — it's the in-game signal that this is
                # the first shot of a fresh game. If the prompt was up
                # before and is gone after, that's a make regardless of
                # OCR or trajectory checks.
                prompt_visible_before = find_game_prompt(frame)[0]
                # Click at the center of the game window. Click coordinates
                # have no measurable effect on aim (verified 2026-05-03 via
                # CLICK_STRATEGY experiments — see docs/cleanup_backlog.md
                # "Hoops: click-position machinery"); the only thing that
                # determines launch trajectory is platform_y at click time.
                # Using a fixed center-of-window target instead of the
                # per-shot (hoop_x, hoop_y) means click position is constant
                # across shots — easier to reason about, removes the
                # implicit dependency on the rim detector for the click
                # target, and eliminates corner-case "click at extreme
                # screen position" risks.
                click(left + width // 2, top + height // 2)
                # Per-shot monitor folder: we'll save pre/post-shot screenshots
                # plus all flight frames captured during the cooldown.
                shot_dir = _make_monitor_dir(shot_idx) if MONITOR_MODE else None
                if shot_dir is not None:
                    save_frame(shot_dir / "pre_shot.png", frame)
                # Score/lives "before" are captured post-click: the in-game
                # score only updates after the ball lands (~2.9s later), so
                # the region still shows the pre-shot value.
                score_before = _capture_score_region(left, top, width, height)
                lives_before = _capture_lives_region(left, top, width, height)
                random_delay(10, 40)
                last_shot = _record_shot_outcome(
                    _FiredShot(
                        left=left, top=top, width=width, height=height,
                        shot_idx=shot_idx,
                        fired_at=fired_at,
                        time_since_last_shot_ms=time_since_last_shot_ms,
                        prompt_visible_before=prompt_visible_before,
                        score_before=score_before,
                        lives_before=lives_before,
                        shot_dir=shot_dir,
                        hoop_x=hoop_x, hoop_y=hoop_y,
                        hoop_conf=hoop_conf_last, hoop_scale=hoop_scale_last,
                        px=px, py=py,
                        clamped=clamped,
                        eff_target_y=effective_target_y,
                        direction=direction,
                        target=target,
                    ),
                    shot_db=shot_db,
                    shot_stats=shot_stats,
                    tracking=tracking,
                    range_samples=range_samples,
                    session_started=session_started,
                    code_commit=code_commit,
                )
                platform_history.clear()
                prev_py = None
                target = None  # re-detect hoop after it repositions
                continue

        prev_py = py

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
