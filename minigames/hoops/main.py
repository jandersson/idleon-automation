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
from minigames.hoops.shot_log import open_db, log_shot, set_make, fetch_makes, fetch_clean_trajectories, fetch_vy_labeled, CLEAN_MAKE_TOLERANCE, MAX_BACK_DRIFT_PX
from common.predictor import fit_knn, fit_bivariate, fit_gp, fit_trajectory_knn, fit_trajectory_gp, fit_trajectory_rf, fit_make_prob
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

# Run find_game_over only every Nth main-loop tick — it's a full-frame
# multi-scale match (112ms profiled) for a persistent terminal screen.
# Worst-case added detection latency is (N-1) ticks (~a quarter second);
# the 5s rim-missing bail is the backstop either way.
GAME_OVER_CHECK_EVERY = 5

# Game-over near-miss capture: a banner that scores in [floor, 0.7) almost
# matched but fell under find_game_over's threshold — the suspected
# higher-score-variant failure. Save the first such frame per run so the
# failing banner can be inspected and the template re-picked / the threshold
# re-set from real data (we currently have no frame of a failing banner — the
# rim-missing diagnostics are all post-banner overworld views). The floor is
# above the ~0.59 whole-frame noise seen on the overworld, so overworld noise
# isn't logged as a candidate banner.
GAME_OVER_NEARMISS_FLOOR = 0.6

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


# 20-make oval phase (#59). After ~20 makes the shooting platform stops
# being a fixed-x vertical bobber and traces a clockwise oval, so its
# x-sample spread opens from ~0 to ~100px. make_prob (platform_y/vy, fixed
# x) can't satisfy its 1-D target there (fire-window -> 0%), so the bot
# stalls and false-bails at make ~20. These gate an oval "fire-to-collect"
# mode that keeps firing (logging the full 2-D platform state) instead of
# stalling, so the data for a 2-D aim model accrues.
OVAL_X_RANGE_PX = 40        # platform x-spread above this = oval phase
OVAL_FIRE_INTERVAL_MS = 1000  # min gap between collect-fires in the oval

# Throttle (s) for the hoop horizontal-motion sampler (#76). The poll loop
# is CV-bound at ~POLL_INTERVAL, so re-running find_rim every tick would
# roughly double per-tick CV cost; 100ms bounds the extra rim matches to
# ~10/s while still giving ~15 samples over the ~1.5s flight window — ample
# for a hoop_vx slope and the oscillation range.
HOOP_SAMPLE_INTERVAL_S = 0.1


def _platform_velocity_x(
    samples: list[tuple[float, int, int]],
    window_s: float = 1.5,
    min_span_s: float = 0.2,
) -> float | None:
    """Platform HORIZONTAL velocity (px/s, positive = rightward) at the
    newest sample — the oval-phase counterpart of _platform_velocity, same
    least-squares slope but over px instead of py. ~0 during the fixed-x
    vertical bob; non-zero once the oval starts (#59)."""
    if not samples:
        return None
    t_end = samples[-1][0]
    recent = [(t, px) for t, px, _ in samples if t_end - t <= window_s]
    if len(recent) < 2 or (recent[-1][0] - recent[0][0]) < min_span_s:
        return None
    n = len(recent)
    mean_t = sum(t for t, _ in recent) / n
    mean_x = sum(x for _, x in recent) / n
    denom = sum((t - mean_t) ** 2 for t, _ in recent)
    if denom == 0:
        return None
    return sum((t - mean_t) * (x - mean_x) for t, x in recent) / denom


def _platform_x_range(range_samples) -> int:
    """Horizontal spread (max-min) of the buffered platform x samples."""
    xs = [p[1] for p in range_samples]
    return (max(xs) - min(xs)) if xs else 0


def _hoop_motion(
    hoop_samples,
) -> tuple[float | None, int | None, int | None, int | None]:
    """(hoop_vx, hoop_xmin, hoop_xmax, hoop_x_at_fire) from the
    (timestamp, hoop_x, hoop_y) rim buffer at the newest sample (#76).

    The hoop oscillates horizontally, but the aim locks hoop_x once at
    target-set and the ball is airborne ~1.5s more, so a moving hoop is
    elsewhere by arrival. These per-shot diagnostics quantify that:
    hoop_vx is the horizontal velocity (px/s, + = rightward) — the
    generic least-squares px-slope, reused from _platform_velocity_x;
    xmin/xmax the observed oscillation range; hoop_x_at_fire the freshest
    detected x, whose delta from the logged (target-set) hoop_x is the
    lining-up staleness. Instrumentation only — the aim is unchanged."""
    vx = _platform_velocity_x(list(hoop_samples)) if hoop_samples else None
    xmin = xmax = None
    if len(hoop_samples) >= 5:
        xs = [p[1] for p in hoop_samples]
        xmin, xmax = int(min(xs)), int(max(xs))
    x_at_fire = int(hoop_samples[-1][1]) if hoop_samples else None
    return vx, xmin, xmax, x_at_fire


def _platform_recently_moving(
    range_samples, n: int = 15, min_motion_px: int = 8
) -> bool:
    """Whether the platform has actually moved over the last `n` samples
    (x OR y spread above `min_motion_px`, i.e. beyond detector jitter).

    Guards oval-fire against an UNDETECTED game-over where find_platform
    false-positives on a now-static screen: the 200-deep buffer still holds
    real oval samples (wide spread) but nothing is moving, so without this
    `oval_active` would stay True, fire-to-collect would keep refreshing
    last_fired_at_ms, and the no-shot runaway bail would never arm (#59
    review). A live oval always has recent motion, so this never blocks it.
    Returns True when there aren't `n` samples yet (don't claim 'stuck')."""
    recent = list(range_samples)[-n:]
    if len(recent) < n:
        return True
    xs = [p[1] for p in recent]
    ys = [p[2] for p in recent]
    return (max(xs) - min(xs)) > min_motion_px or (max(ys) - min(ys)) > min_motion_px


def _oval_active(range_samples, min_samples: int = 40) -> bool:
    """True once the platform is tracing the post-20-make clockwise oval:
    its x samples have a sustained spread exceeding OVAL_X_RANGE_PX (the
    spread is ~0 during the fixed-x bob) AND the platform is still moving.

    Uses the 10th-90th percentile spread, NOT raw max-min: a single
    detector x-glitch (e.g. the 136->374 jump in the 2026-05-24 session)
    would blow up max-min and falsely trip oval-fire mode during normal
    play, since one outlier lingers in the 200-deep buffer. The real oval
    moves the platform across the whole x-range continuously, so the
    percentile spread is large; a lone outlier sits past the 90th pct and
    barely moves it. The recent-motion gate (see _platform_recently_moving)
    drops oval mode the moment the platform stops, so a stale buffer on an
    undetected game-over can't hold off the runaway bail. Requires >=
    min_samples (#59)."""
    if len(range_samples) < min_samples:
        return False
    xs = sorted(p[1] for p in range_samples)
    lo = xs[int(len(xs) * 0.1)]
    hi = xs[int(len(xs) * 0.9)]
    if (hi - lo) <= OVAL_X_RANGE_PX:
        return False
    return _platform_recently_moving(range_samples)


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

# A miss whose ball still arrived SHORT of hoop_x by no more than this was a
# launch/rim outcome, not an aim error — shots arriving short in this band
# make at ~64% (front-rim luck), so the sweep must NOT step away from a
# target that produced one. The hold is SHORT-side only (#97): a miss that
# arrived PAST hoop_x by more than BACK_RIM_LO_PX clanked the back rim and
# bounced out — re-firing the same launch just grinds the back rim, so those
# advance the sweep instead (see _sweep_should_advance).
IN_BAND_RESIDUAL_PX = 60

# Back-rim notch geometry (#97), all in px of bounce-aware arrival residual
# (arrival_x - hoop_x; arrival_x per _shot_arrival_x). Calibrated from
# shots.db (n=840 clean shots): make-rate falls off the swish plateau
# (residual [-50,-15] make 0.78) once the ball arrives past center —
# residual [15,110] makes only ~0.40-0.53, a notch flanked by swish below
# and far-bank backboard banks (residual >120, make ~0.79) above.
# - BACK_RIM_LO_PX: positive-side onset; past this the sweep advances and the
#   recovery nudge may fire.
# - BACK_RIM_HI_PX: upper edge; above it is far-bank (productive banks) which
#   must NOT be nudged shorter.
BACK_RIM_LO_PX = 15
BACK_RIM_HI_PX = 110

# Directional shorter-step applied to the recovery shot's perturbation after a
# detected back-rim bounce-out, on the "nudge" A/B arm (#97). Negative lowers
# target_y → the bot waits for a lower platform crossing → flatter/shorter arc
# → arrival walks from the back-rim notch toward the swish band. Within
# PERTURBATION_MAX=32. Geometry-calibrated, not fitted — the A/B's tunable knob.
BACK_RIM_NUDGE_PX = -16


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


def _is_back_rim_bounceout(
    peak_x: int | None,
    landing_x: int | None,
    hoop_x: int | None,
) -> bool:
    """True when a miss clanked the BACK rim and bounced out: the ball was
    flung backward (peak_x - landing_x > MAX_BACK_DRIFT_PX) AND its rightmost
    reach landed BACK_RIM_LO_PX..BACK_RIM_HI_PX past hoop center (#97).

    Direction-safe by construction. It excludes:
    - backward bounces that peaked SHORT of center (peak_x - hoop_x < LO):
      those hit the FRONT rim and need MORE reach — nudging them shorter would
      push toward the airball zone;
    - far-bank shots reaching past HI (peak_x - hoop_x > 110): productive
      backboard banks that score and must not be steered shorter;
    - clean rim-drops with no backward bounce (peak ≈ landing).

    This is the trigger for the recovery shorter-nudge — only launch states
    that overshot into the back rim get steered shorter.
    """
    if peak_x is None or landing_x is None or hoop_x is None:
        return False
    if (peak_x - landing_x) <= MAX_BACK_DRIFT_PX:
        return False
    return BACK_RIM_LO_PX <= (peak_x - hoop_x) <= BACK_RIM_HI_PX


def _apply_back_rim_nudge(perturbation: int, arm: str | None) -> int:
    """Recovery-shot perturbation override (#97). On the 'nudge' A/B arm,
    force the perturbation to BACK_RIM_NUDGE_PX (a directional shorter step)
    regardless of the sweep value; on 'control' or None, leave it unchanged.
    Pure so the one-sided, bounded override is unit-testable in isolation."""
    if arm == "nudge":
        return BACK_RIM_NUDGE_PX
    return perturbation


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
    next entry. The hold band is ASYMMETRIC (#97):

    - Arrived SHORT of hoop_x by <= IN_BAND_RESIDUAL_PX (residual in
      [-60, 0]): HOLD — re-fire the same target. Short/center in-band
      misses are front-rim luck that converts on a re-fire.
    - Arrived just PAST hoop_x but within the notch onset (residual in
      (0, BACK_RIM_LO_PX]): HOLD — still essentially on-center.
    - Arrived PAST hoop_x by > BACK_RIM_LO_PX (back-rim zone): ADVANCE —
      the ball clanked the back rim and bounced out; re-firing the same
      launch just grinds the back rim instead of stepping toward the make.
    - Arrived SHORT by > IN_BAND_RESIDUAL_PX (big short miss): ADVANCE.

    `arrival_x` should come from _shot_arrival_x so structure clanks aren't
    misread as short misses. No trajectory data → advance (can't tell a
    mislaunch from a near-miss, and standing still forever on an unmeasured
    target risks a stuck loop).
    """
    if arrival_x is None or hoop_x is None:
        return True
    resid = arrival_x - hoop_x
    return resid > BACK_RIM_LO_PX or resid < -IN_BAND_RESIDUAL_PX


# Score>=10 hoop drift (measured 2026-06-10 from flight frames: the hoop
# sweeps horizontally at ~15-35 px/s, oscillating at constant y with
# occasional vertical steps). Aim read at target acquisition is stale by
# seconds at fire time — good aim at a ghost. When the platform closes
# on the target, refresh the hoop fix and carry the chosen offset to the
# fresh position. The refresh happens during the wait, never between the
# fire decision and the click (CLAUDE.md click-timing rule), and only at
# 10+ where drift exists (find_rim costs ~30-80ms per call).
DRIFT_SCORE_MIN = 10
DRIFT_REDETECT_APPROACH_PX = 60


def _should_refresh_hoop(session_score: int, py: int, target_y: int) -> bool:
    """True when the drifting-hoop re-aim should run this poll: score in
    drift territory and the platform closing on the target."""
    return (
        session_score >= DRIFT_SCORE_MIN
        and abs(py - target_y) <= DRIFT_REDETECT_APPROACH_PX
    )


def _score_implies_prev_make(
    prev_made,
    prompt_visible: bool,
    score_before_int: int | None,
    anchor: int,
) -> bool:
    """True when this shot's pre-read proves the PREVIOUS shot made: the
    anchor advanced by a single make's worth (+1 or +2) with no other
    explanation. The respawn signal can't cover score>=10 (the hoop
    drifts every detection, so respawn is indistinguishable from drift)
    — observed 2026-06-10 12:36 shot 12: score went 10->11, OCR dropped
    post-shot, row logged miss. Prompt-up pre-reads are noise and never
    count.
    """
    if prev_made or prompt_visible or score_before_int is None:
        return False
    return 1 <= score_before_int - anchor <= 2


def _crossing_candidates(
    samples: list[tuple[float, int, int]],
    bucket_px: int = 8,
    max_age_s: float = 40.0,
) -> list[tuple[int, float, str]]:
    """Candidate (target_y, expected_vy, direction) triples from the live
    platform buffer (#38).

    Each consecutive sample pair sweeps a y-band in one direction; the
    bob is periodic, so the platform will cross those ys again next
    cycle at a similar phase — meaning the trailing-window vy observed
    during the sweep is the expected vy of a future fire at that y.
    This is what lets the make-probability model choose a (y, vy, dir)
    state empirically instead of from bob physics. Bucketed by y band
    and direction; the newest observation of a bucket wins.
    """
    if len(samples) < 3:
        return []
    t_end = samples[-1][0]
    buckets: dict[tuple[int, str], tuple[int, float]] = {}
    for i in range(1, len(samples)):
        t1, _, y1 = samples[i]
        t0, _, y0 = samples[i - 1]
        if t_end - t1 > max_age_s or y1 == y0:
            continue
        direction = "up" if y1 < y0 else "down"
        mid_y = (y0 + y1) // 2
        vy = _platform_velocity(samples[: i + 1])
        if vy is None:
            continue
        buckets[(mid_y // bucket_px, direction)] = (mid_y, vy)
    return [(y, vy, d) for (_b, d), (y, vy) in buckets.items()]


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
#   "make_prob"       — GP classifier P(make | hoop pos, platform_y,
#                       platform_vy); ranks live (platform_y, vy, dir)
#                       candidates and fires the argmax. Falls back to
#                       the per-direction regressors until the labeled-
#                       vy data floor is met. See GitHub issue #38.
#
# The launcher's Bots tab exposes a Predictor dropdown that sets the
# HOOPS_PREDICTOR_KIND env var; this constant is the default when the
# env var is unset (e.g. running `uv run hoops` directly).
#
# Default promoted "gp" → "make_prob" 2026-06-10 (#38): 55/83 = 66.3%
# post-policy non-explore makes, Wilson 95% CI [0.556, 0.755] — lower
# bound clear of the #37 policy stack's entire CI [0.273, 0.433]
# (n=132). Per-session rates 57-77% with no decay. Measured by
# scripts/check_38_promotion.py.
PREDICTOR_KIND = os.environ.get("HOOPS_PREDICTOR_KIND", "make_prob")

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
    oval_active: bool = False,
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
    #
    # `oval_active` also overrides the trajectory check (#59): in the
    # 20-make oval phase the platform launches the ball from a shifted x,
    # so ball_x_at_rim is legitimately far from hoop_x even on makes — the
    # fixed-x trajectory check then vetoes real makes (session 03:01:28:
    # shots 19/20 scored 20->21->22 but were logged as misses). The score
    # OCR is reliable there, so a clean +1/+2 increment IS the make signal.
    changed = plausible_inc and (trajectory_plausible or prompt_disappeared or oval_active)

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
        if oval_active and not trajectory_plausible:
            label = f"MAKE (+{inferred_increment}) [oval — score increment accepted]"
        elif prompt_disappeared and not trajectory_plausible:
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

# Final catch-all session-bail: once the game has actually started (>=1 shot
# fired), if NO shot fires for this long the run is stuck and must end. The
# other watchdogs can all be defeated together when the game-over screen is up
# but its template variant doesn't match AND find_rim returns a phantom
# low-confidence "rim": the rim-missing 5s bail never fires (rim "present"),
# the platform-not-found path only clears the target, and the stale-target
# 60s watchdog never accumulates (the target keeps getting cleared and reset).
# That looped forever on 2026-06-15 (session_20260615_035159: platform-not-
# found / rim@0.48 repeating). This bail depends on nothing but wall-clock
# since the last shot, so it always exits.
NO_SHOT_BAIL_S = 30.0


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
    # A/B arm armed for the NEXT shot at this hoop after a detected back-rim
    # bounce-out (#97): "nudge" | "control" | None. Reset on make / new hoop
    # so a stale arm never leaks across hoops.
    pending_nudge_arm: str | None = None

    def reset_for_new_hoop(self, key: tuple[int, int] | None) -> None:
        self.key = key
        self.misses = 0
        self.holds = 0
        self.shots = 0
        self.pending_nudge_arm = None


@dataclass
class _Target:
    """Everything _select_target decided for the current hoop."""
    target_y: int
    offset: int
    perturbation: int
    predicted_offset: int
    target_source: str   # "predictor" | "sweep" | "explore" | "model"
    required_dir: str
    predictor_fit: bool  # whether a fitted predictor (vs cold start) aimed this
    # Back-rim recovery A/B arm (#97): "nudge" | "control" | None. Non-None
    # only on a shot fired right after a detected back-rim bounce-out at this
    # hoop. Logged to shots.back_rim_recovery for the nudge-vs-control gate.
    back_rim_recovery: str | None = None


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
    # Back-rim recovery (#97): on the "nudge" A/B arm, force a directional
    # shorter step after the prior shot back-rimmed at this hoop. The arm was
    # set in the miss branch (_record_shot_outcome); it flows through both the
    # predictor offset (below) and the model offset, since both add
    # `perturbation`. The make_prob argmax itself is NOT touched — only the
    # offset added around the chosen state shifts. Cleared in the explore
    # branch so free pre-game shots are never nudged or tagged.
    back_rim_arm = tracking.pending_nudge_arm
    perturbation = _apply_back_rim_nudge(perturbation, back_rim_arm)
    offset = base_offset + perturbation
    target_y = hoop_y + offset
    target_source = "sweep" if perturbation else "predictor"
    model_tag = ""
    # Make-probability override (#38): rank the live (platform_y, vy,
    # dir) candidates from the bob buffer and fire at the most makeable
    # state — direction included, so the model can out-vote the
    # hand-drawn direction policy where it has the data. Falls through
    # to the per-direction predictor when the model or buffer is thin.
    # The miss-sweep perturbation still applies around the model's pick.
    model = predictors.get("make_prob")
    if model is not None and len(range_samples) >= 40:
        cands = _crossing_candidates(list(range_samples))
        if cands:
            probs = model.score_candidates(
                hoop_y, hoop_x, [(y, vy) for y, vy, _d in cands],
            )
            best_i = max(range(len(cands)), key=lambda i: probs[i])
            best_y, best_vy, best_dir = cands[best_i]
            required_dir = best_dir
            predicted_offset_value = best_y - hoop_y
            offset = predicted_offset_value + perturbation
            target_y = hoop_y + offset
            target_source = "model"
            predictor = model  # marks predictor_fit for logging
            model_tag = f" [model p={probs[best_i]:.2f} @ y={best_y} vy~{best_vy:+.0f} dir={best_dir}]"
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
            back_rim_arm = None  # free shots are never nudged or A/B-tagged
            model_tag = ""  # exploration overrides the model's pick
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
    back_rim_tag = f" [back-rim recovery: {back_rim_arm}]" if back_rim_arm else ""
    print(f"Hoop rim at ({hoop_x},{hoop_y}) (conf={hoop_conf:.2f}){sigma_tag}, offset={offset}{tag}{override_tag}{cap_tag}{model_tag}{dir_tag}{back_rim_tag}, target launch y={target_y}")
    return _Target(
        target_y=target_y,
        offset=offset,
        perturbation=perturbation,
        predicted_offset=predicted_offset_value,
        target_source=target_source,
        required_dir=required_dir,
        predictor_fit=predictor is not None,
        back_rim_recovery=back_rim_arm,
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
    prev_last_shot: dict | None = None,
    hoop_samples=(),
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
    # Score-anchored retro-make: this shot's pre-read proves the PREVIOUS
    # shot scored when the anchor advanced by a make's worth — the 10+
    # counterpart of the respawn signal (which is disabled there, since
    # drift makes every detection look like a respawn).
    if prev_last_shot is not None and _score_implies_prev_make(
        prev_last_shot["made"], shot.prompt_visible_before,
        score_before_int, shot_stats.get("session_score", 0),
    ):
        clean_value, _ = _classify_clean_make(
            True, prev_last_shot["landing"], prev_last_shot["peak"],
            prev_last_shot["hoop_x"],
        )
        set_make(shot_db, prev_last_shot["id"], 1, clean_value, "score")
        shot_stats["makes"] += 1
        print(
            f"  [score-retro] pre-read {score_before_int} proves the previous shot made — "
            f"retro-marking shot id={prev_last_shot['id']} (clean={clean_value}) | "
            f"confident makes {shot_stats['makes']}/{shot_stats['attempts']}"
        )
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
    # Oval phase (#59): in the 20-make oval the ball launches from a shifted
    # x, so the fixed-x trajectory check would veto real makes — let
    # _log_shot_result trust the score increment there. Computed once and
    # reused for the logged oval_active column below.
    oval_active_value = _oval_active(range_samples)
    made, score_diff, score_increment = _log_shot_result(
        shot_stats, shot.score_before, score_after,
        score_before_int=score_before_int,
        score_after_int=score_after_int,
        ball_x_at_rim=trajectory["ball_x_at_rim_height"],
        hoop_x=shot.hoop_x,
        prompt_disappeared=prompt_disappeared,
        prompt_visible_before=shot.prompt_visible_before,
        oval_active=oval_active_value,
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
    # Horizontal velocity (#59): ~0 during the normal fixed-x bob, populated
    # once the platform traces the post-20-make oval. oval_active_value was
    # computed above (it also gates the make-detection trajectory bypass).
    platform_vx_value = _platform_velocity_x(list(range_samples))
    # Hoop horizontal motion at fire (#76): the aim's hoop_x is the
    # target-set snapshot; the hoop has kept oscillating since. hoop_vx +
    # the observed x-range + the freshest x quantify how stale the aim's
    # hoop_x is (hoop_x_at_fire - hoop_x), feeding a later lead-the-target fix.
    hoop_vx_value, hoop_xmin_value, hoop_xmax_value, hoop_x_at_fire_value = (
        _hoop_motion(list(hoop_samples))
    )
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
        platform_vx=platform_vx_value,
        oval_active=int(oval_active_value),
        hoop_vx=hoop_vx_value,
        hoop_xmin=hoop_xmin_value,
        hoop_xmax=hoop_xmax_value,
        hoop_x_at_fire=hoop_x_at_fire_value,
        prompt_up=int(bool(shot.prompt_visible_before)),
        target_source=shot.target.target_source,
        back_rim_recovery=shot.target.back_rim_recovery,
    )
    # Update perturbation tracking. Made → reset (next hoop will be in a
    # new position anyway). Miss → advance the sweep, unless the ball
    # arrived in-band (aim was right; re-fire the same target rather
    # than stepping away from it).
    if made:
        tracking.reset_for_new_hoop(None)
    else:
        peak = trajectory["ball_peak_x"]
        landing = trajectory["ball_landing_x"]
        arrival_x = _shot_arrival_x(
            trajectory["ball_x_at_rim_height"], peak, landing,
        )
        advance, tracking.holds = _sweep_step(arrival_x, shot.hoop_x, tracking.holds)
        if advance:
            tracking.misses += 1
        else:
            resid = arrival_x - shot.hoop_x
            bounce_tag = (
                " [bounce-aware: peak_x]"
                if arrival_x == peak and arrival_x != trajectory["ball_x_at_rim_height"]
                else ""
            )
            print(f"  [perturb] miss but ball arrived {resid:+d}px from hoop_x{bounce_tag} — re-firing same target (hold {tracking.holds}/{MAX_CONSECUTIVE_HOLDS})")
        # Arm the NEXT shot's back-rim recovery A/B (#97). When this miss
        # clanked the BACK rim and bounced out, the next shot at this hoop
        # alternates a directional shorter nudge ("nudge") vs normal sweep
        # ("control") by shot-index parity (~50/50, mirrors darts #48) so a
        # powered nudge-vs-control comparison accumulates before the nudge
        # becomes the default. Any non-bounce-out miss clears a stale arm.
        if _is_back_rim_bounceout(peak, landing, shot.hoop_x):
            arm = "nudge" if shot.shot_idx % 2 == 0 else "control"
            tracking.pending_nudge_arm = arm
            print(f"  [back-rim] bounce-out (peak_x={peak}, hoop_x={shot.hoop_x}, "
                  f"drift={peak - landing}px) — next shot armed '{arm}'")
        else:
            tracking.pending_nudge_arm = None
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
SHOT_DB_REL = "minigames/hoops/assets/shots.db"


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
            if PREDICTOR_KIND == "make_prob":
                # The classifier ranks live (platform_y, vy, dir)
                # candidates; the per-direction predictors above stay as
                # the cold-start / thin-buffer fallback.
                vy_rows = fetch_vy_labeled(shot_db)
                model = fit_make_prob(vy_rows)
                if model is None:
                    print(f"make_prob: data floor not met ({len(vy_rows)} labeled vy rows) — running on the per-direction fallback")
                else:
                    print(f"MAKE_PROB classifier fitted (n={model.n}, makes={sum(r[4] for r in vy_rows)}) — ranking live candidates")
                    predictors["make_prob"] = model
            _run_inner(session_started, shot_db, predictors, code_commit)
        finally:
            shot_db.close()
            _refresh_and_commit_snapshot()


def _fit_predictor(shot_db, direction: str):
    """Fit the PREDICTOR_KIND model on rows for one firing direction.
    Returns None (cold start) when too few samples exist. make_prob's
    per-direction fallback uses gp."""
    kind = "gp" if PREDICTOR_KIND == "make_prob" else PREDICTOR_KIND
    if kind in ("trajectory_knn", "trajectory_gp", "trajectory_rf"):
        # All trajectory predictors learn from every shot (make or miss)
        # and use the same input schema (4-tuple with arrival_x) and
        # envelope clamp from nearby makes.
        rows = fetch_clean_trajectories(shot_db, direction)
        makes = fetch_makes(shot_db, direction)
        if kind == "trajectory_knn":
            predictor = fit_trajectory_knn(rows, makes=makes, k=5)
        elif kind == "trajectory_gp":
            predictor = fit_trajectory_gp(rows, makes=makes)
        else:
            predictor = fit_trajectory_rf(rows, makes=makes)
    else:
        rows = fetch_makes(shot_db, direction)
        if kind == "knn":
            # k=3 (was 5): smaller K is more local. With k=5 the
            # predictor over-smoothed in regions with sparse training
            # data — a single nearby make at hoop=(593,390) with
            # offset=78 got diluted to offset=23 by 4 distant
            # neighbours. k=3 lets the local data dominate more.
            predictor = fit_knn(rows, k=3)
        elif kind == "bivariate":
            predictor = fit_bivariate(rows)
        elif kind == "gp":
            predictor = fit_gp(rows)
        else:
            raise ValueError(f"Unknown PREDICTOR_KIND {PREDICTOR_KIND!r}")
    if predictor is None:
        print(f"No predictor fit ({kind!r}, too few samples in dir={direction!r}); using cold-start offset={COLD_START_OFFSET}")
    else:
        sample_kind = "shots" if kind in ("trajectory_knn", "trajectory_gp", "trajectory_rf") else "makes"
        print(f"{kind.upper()} target predictor (dir={direction!r}, n={predictor.n} {sample_kind})")
    return predictor


def _refresh_and_commit_snapshot() -> None:
    """Regenerate shots_snapshot.json from the DB, then commit the DB +
    snapshot together and push (if in window). The DB is tracked so other
    machines get session data via `git pull`. Best-effort — failures are
    reported but don't propagate."""
    try:
        from scripts import dump_shots
        dump_shots.main()
    except Exception as e:
        print(f"  [snapshot] dump failed (non-fatal): {e}")
        return
    commit_file_if_changed(
        REPO_ROOT,
        [SHOT_DB_REL, SNAPSHOT_REL],
        "chore(hoops): refresh shots.db + snapshot (auto)",
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
    platform_missing_since: float | None = None  # no-platform bail (00:40 runaway)
    range_samples: deque[tuple[float, int, int]] = deque(maxlen=200)  # (ts, px, py) for range + period diagnostics
    last_range_log = time.time()
    # Hoop horizontal-motion buffer (#76): (ts, hoop_x, hoop_y), the rim
    # counterpart of range_samples. Sampled on a throttle (HOOP_SAMPLE_INTERVAL_S)
    # so hoop_vx + the oscillation range can be logged per shot — the aim
    # path that locks hoop_x at target-set is unchanged.
    hoop_samples: deque[tuple[float, int, int]] = deque(maxlen=200)
    last_hoop_sample = 0.0
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
    tick = 0  # for the every-Nth-tick game-over check
    go_nearmiss_saved = False  # one game-over near-miss frame saved per run
    oval_announced = False  # print the 20-make oval transition once (#59)

    while True:
        check_failsafe()
        try:
            left, top, width, height = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        frame = grab_region(left, top, width, height)

        # Stop cleanly when the trial ends. Checked every Nth tick only:
        # the full-frame match is the single largest per-tick cost
        # (112ms, profiled 2026-06-10). CORRECTION (2026-06-11, the
        # 00:40 runaway): the game-over BANNER is a ~3-4s timed window,
        # not a persistent terminal state — and the bot spends most of
        # it blind inside _record_shot_outcome (~6s of flight capture +
        # OCR per shot). The banner matches the template at conf ~0.92
        # (true-positive frames now on disk: monitor/shot_007_004117/);
        # the old "no true positive above 0.574" record was measured on
        # NON-banner frames. After the banner, the rim/platform remain
        # rendered (rim matched 0.98 on the banner screen), so the
        # rim-missing bail can't catch it either. Hence: every shot
        # outcome resets `tick` below so the FIRST post-shot iteration
        # always runs this check, inside the banner window.
        tick += 1
        if tick % GAME_OVER_CHECK_EVERY == 1:
            is_over, go_conf = find_game_over(frame)
            if is_over:
                print(f"Game over detected (conf={go_conf:.2f}). Final session: {shot_stats['makes']}/{shot_stats['attempts']} makes.")
                return
            # A near-miss (almost matched but under threshold) is the prime
            # suspect for an unrecognised game-over variant — capture it once.
            if (not go_nearmiss_saved
                    and GAME_OVER_NEARMISS_FLOOR <= go_conf < 0.7):
                go_nearmiss_saved = True
                diag_dir = _HERE / "assets" / "diagnostics"
                diag_dir.mkdir(parents=True, exist_ok=True)
                p = diag_dir / f"go_nearmiss_{go_conf:.2f}_{datetime.now():%Y%m%d_%H%M%S}.png"
                save_frame(p, frame)
                print(f"  [game-over near-miss] conf={go_conf:.2f} (<0.70) — saved "
                      f"{p.name}; re-pick the template (hoops-pick-game-over) if it's a banner.")

        # Catch-all stuck-bail: the game has started (>=1 shot) but nothing has
        # fired for NO_SHOT_BAIL_S — the run is wedged (e.g. an unmatched
        # game-over screen variant + a phantom low-conf rim defeating the other
        # bails; see NO_SHOT_BAIL_S). Save the frame so the unrecognised screen
        # can be inspected / re-picked, then exit.
        if (last_fired_at_ms is not None
                and time.time() * 1000.0 - last_fired_at_ms > NO_SHOT_BAIL_S * 1000.0):
            diag_dir = _HERE / "assets" / "diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            diag_path = diag_dir / f"noshot_{datetime.now():%Y%m%d_%H%M%S}.png"
            save_frame(diag_path, frame)
            print(f"No shot fired in {NO_SHOT_BAIL_S:.0f}s — bailing out (game over "
                  f"likely undetected; saved {diag_path.name} — if it's a new "
                  f"game-over variant, re-run hoops-pick-game-over). "
                  f"Final session: {shot_stats['makes']}/{shot_stats['attempts']} makes.")
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
            # No-platform bail (2026-06-11): this path used to sleep and
            # continue forever, silently — with a target set the rim
            # check above is skipped too, so a screen with neither
            # platform nor a way to detect game over kept the bot alive
            # ~65s until the stale-target watchdog (the 00:40 runaway).
            # Mirror the rim-missing rule: platform continuously gone
            # for >5s clears the target, handing control back to the
            # rim-seek path, whose own 5s bail ends the session.
            if platform_missing_since is None:
                platform_missing_since = time.time()
            elif time.time() - platform_missing_since > 5:
                print(f"Platform not found for >5s (best conf={platform_conf:.2f}) — "
                      f"clearing target and re-seeking the rim.")
                target = None
                platform_missing_since = None
            time.sleep(POLL_INTERVAL)
            continue
        platform_missing_since = None

        px, py = platform_pos
        platform_history.append(py)
        range_samples.append((time.time(), px, py))

        # Hoop horizontal-motion sampler (#76). The aim locks hoop_x once at
        # target-set (~find_rim above) and the ball flies ~1.5s more, so a
        # horizontally-oscillating hoop is stale by arrival. Sample the rim
        # on a throttle into its own buffer — purely for the hoop_vx /
        # oscillation-range diagnostics logged per shot; the aim is untouched.
        # Reuses `frame` already grabbed this tick (no extra capture).
        now_hs = time.time()
        if now_hs - last_hoop_sample >= HOOP_SAMPLE_INTERVAL_S:
            hs_pos, _hs_conf, _hs_scale = find_rim(frame)
            if hs_pos is not None:
                hoop_samples.append((now_hs, int(hs_pos[0]), int(hs_pos[1])))
            last_hoop_sample = now_hs

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

        # Drifting-hoop re-aim (score >= 10): refresh the hoop fix while
        # the platform closes on the target, carrying the chosen offset
        # to the fresh position (see DRIFT_SCORE_MIN rationale).
        if _should_refresh_hoop(
            shot_stats.get("session_score", 0), py, target.target_y,
        ):
            fresh_pos, fresh_conf, fresh_scale = find_rim(frame)
            if fresh_pos is not None and fresh_conf >= 0.9:
                fx, fy = fresh_pos
                if abs(fx - hoop_x) > 2 or abs(fy - hoop_y) > 2:
                    print(f"  [drift] hoop ({hoop_x},{hoop_y}) -> ({fx},{fy}) — re-aiming with offset {target.offset:+d}")
                    hoop_x, hoop_y = fx, fy
                    hoop_conf_last = fresh_conf
                    hoop_scale_last = fresh_scale
                    target.target_y = fy + target.offset

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
        # 20-make oval phase (#59): once the platform's x-spread opens up it's
        # tracing the clockwise oval, where the fixed-x make_prob target is
        # never satisfied (fire-window -> 0%) so the bot stalls and false-
        # bails. Fire-to-collect: fire on a cooldown to log oval shots
        # (platform_x/y/vx/vy + outcome) for the future 2-D aim, rather than
        # stalling. Entirely gated on oval_active, so the validated pre-oval
        # path is unchanged. (Aim here is still the bob target, so these
        # mostly miss — the value is the logged 2-D state, not the make.)
        oval_active = _oval_active(range_samples)
        if oval_active and not oval_announced:
            oval_announced = True
            print(f"  [oval] platform x-spread {_platform_x_range(range_samples)}px "
                  f"> {OVAL_X_RANGE_PX} — 20-make oval phase detected; firing to "
                  f"collect 2-D state (aim model TODO, #59)")
        oval_fire_due = oval_active and (
            last_fired_at_ms is None
            or time.time() * 1000.0 - last_fired_at_ms > OVAL_FIRE_INTERVAL_MS
        )
        if in_window or crossed or oval_fire_due:
            y_open_window += 1
            # Direction from the actual crossing if available, else from history.
            if crossed and prev_py is not None:
                direction = "up" if py < prev_py else "down"
            else:
                direction = _direction(platform_history)
            # When clamped, the platform only crosses the target at a bob extreme,
            # where direction is whatever-it-just-was → never matches. Bypass.
            # In the oval the y-direction gate is meaningless (2-D motion), so
            # bypass it too — the collect-fire must not be blocked by it.
            direction_ok = clamped or oval_active or target.required_dir == "any" or direction == target.required_dir
            if direction_ok:
                tag = " [crossed]" if crossed and not in_window else ""
                if clamped:
                    tag += " [clamped]"
                if oval_active:
                    tag += " [oval-collect]"
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
                # "Make a shot to start the game!" prompt state at fire
                # time — the in-game signal that this is the first shot of
                # a fresh game (prompt up before + gone after = make,
                # regardless of OCR or trajectory checks). Computed AFTER
                # the click, from the frame already in memory: the match
                # result is identical (the frame predates the click either
                # way), but running it pre-click sat ~100ms of full-frame
                # template matching between the fire decision and the
                # click, scattering launch y by platform_vy * latency —
                # the exact bias the click-timing rule exists to prevent.
                # It crept in 2026-05-09 (cc42529), five days after the
                # original latency fix, and shipped with every shot since.
                prompt_visible_before = find_game_prompt(frame)[0]
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
                    prev_last_shot=last_shot,
                    hoop_samples=hoop_samples,
                )
                platform_history.clear()
                # Clear the hoop buffer too: the hoop teleports to a new x on
                # a make, so carrying samples across shots would mix two hoops
                # and inflate hoop_xmin/xmax. Each shot's window observes only
                # the current hoop's oscillation (#76).
                hoop_samples.clear()
                last_hoop_sample = 0.0
                prev_py = None
                target = None  # re-detect hoop after it repositions
                # Force the game-over check on the very next iteration:
                # a life-losing shot is the only thing that triggers the
                # ~3-4s game-over banner, and _record_shot_outcome just
                # consumed most of that window. Without this, detection
                # is a 1-in-5 tick lottery on the banner's last second
                # (lost in the 2026-06-11 00:40 session — ran dead until
                # manually killed).
                tick = 0
                continue

        prev_py = py

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
