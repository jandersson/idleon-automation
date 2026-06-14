import time
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, check_failsafe, press_key
from common.monitor import make_shot_dir, save_frame, save_meta
from common.regions import get_region
from common.session_log import session_log
from common.window import get_bounds, WindowNotFoundError
from common.score_ocr import read_score as _read_score_tesseract
from common.score_template_ocr import make_score_reader
from common.dart_trajectory import analyse_throw_dir
from common.git_info import current_code_commit
from common.auto_commit import commit_file_if_changed
from minigames.darts.detector import find_release_pose, find_game_over, find_celebration, score_region, score_changed
from minigames.darts.wind import parse_wind
from minigames.darts.shot_log import open_db, log_throw, log_poll
from minigames.darts.arm_motion import (
    REF_MOTION_DT_S,
    centroid_vy_px_s,
    compute_arm_centroid,
)
from minigames.darts.stripe_model import (
    MIN_SAMPLES as STRIPE_MIN_SAMPLES,
    StripeEvModel,
    fetch_stripe_rows,
    fit_stripe_gp,
    model_vy_band,
)

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
THROW_DB_PATH = _HERE / "assets" / "darts.db"
REPO_ROOT = _HERE.parent.parent

WINDOW_TITLE = "Legends Of Idleon"
POLL_INTERVAL = 0.02

# Template-based score reader (replaces tesseract). Templates live in
# assets/digit_templates/ and are bootstrapped from past clean OCR reads
# via scripts/bootstrap_digit_templates.py. Falls back to tesseract when
# the template reader returns None (missing digit template → don't lose
# what tesseract could have caught even if imperfect). Closes the read-
# side gap from #27.
_template_reader = make_score_reader(_HERE / "assets" / "digit_templates")


def read_score(crop):
    """Read the darts score from a region crop. Tries template OCR first
    (deterministic, no false 195/-4 reads); falls back to tesseract when
    a digit isn't in the template library yet."""
    val = _template_reader(crop)
    if val is not None:
        return val
    return _read_score_tesseract(crop)

# Template-match confidence threshold for the release pose. The hand sweeps
# through other angles where the template matches weakly; threshold gates
# firing to the moments when it's in the captured release angle.
RELEASE_THRESHOLD = 0.7

# Mean Otsu-binarized pixel-diff threshold for "the score digits changed".
# Was 3.0 (the common.score_diff default); spot-check on 30 recent throws
# found two real hits at diff=2.25 (3→8) and diff=2.63 (5→6) being marked
# as misses. Real misses sit at exactly 0.0, so the floor for "something
# happened" is well below 1.0. Single-digit score region (40×17 px) means
# even a full digit change only flips a small fraction of pixels.
SCORE_CHANGE_THRESHOLD = 1.0

# Steam screenshot on Nine Dart Finish completion. When enabled, the bot
# presses F12 (Steam's default screenshot binding) after detecting 9
# consecutive hits — the requirement for the in-game Nine Dart Finish
# trophy. Limitation: counts any hit as a streak-keeper; if the trophy
# strictly requires *bullseyes* (and a non-bullseye hit resets the
# streak in-game), we'd false-fire on a 9-hit run that wasn't all
# bullseyes. Tighten to bullseye-only via score-magnitude or visual
# streak-counter detection if false fires happen.
STEAM_SCREENSHOT_ON_NINE_DART = False
NINE_DART_STREAK = 9

# Wait after throwing for: dart to land, score/animation to settle, new dart to
# load, and the player+platform to teleport to a new spawn position.
POST_THROW_COOLDOWN = 1.5

# Score region for make/miss diff. Calibrate via darts-pick-score-region (TODO)
# or set to None to skip score logging.
# Score and wind regions are loaded fresh from regions.json each call (using
# current window dims) so they survive the user resizing the game window.
# Pick via darts-pick-score-region / darts-pick-wind-region.
WIND_SAMPLES_DIR = Path(__file__).parent / "assets" / "wind_samples"
WIND_DEDUP_THRESHOLD = 5.0  # mean pixel diff above this = new wind state

# When enabled, every throw writes a per-throw subfolder under assets/monitor/
# with pre/post screenshots, the wind crop, and a metadata file. User can zip
# and share for offline review (since the bot can't watch the screen live).
MONITOR_MODE = True
MONITOR_DIR = Path(__file__).parent / "assets" / "monitor"
POST_LAND_DELAY = 0.6  # how long to wait after the cooldown before post-screenshot

# Capture flight frames during the post-throw cooldown so we can later
# extract dart-landing position relative to the bullseye and study how
# wind state correlates with landing offset. Heavyweight (~50KB/frame
# × ~30 frames per throw) but the data is what unlocks any future
# release-angle predictor. Mirrors hoops' MONITOR_FLIGHT pattern.
MONITOR_FLIGHT = True
FLIGHT_POLL = 0.05

# If no release pose has matched in this many seconds, assume the game-over
# screen has replaced the dartboard scene (the entire scene is replaced at
# game over, so the player avatar disappears and the template can't match).
# 10s was too tight — false-fired between normal throws after a 7-throw run
# while the game was still active. 25s gives plenty of headroom for slow
# cycles (wind change, score animations) while still terminating quickly
# when the dartboard is actually gone.
GAME_OVER_NO_POSE_SEC = 25.0

# Streak celebration handling ("...one hundred and EIGHTY!", appears at
# 4 bullseyes in a row): the celebration animation replaces the throwing
# pose, so no release match happens until it's dismissed — without
# detection the no-pose timeout bails mid-game (observed session
# 2026-06-10 11:52). Start checking for the banner once the pose has
# been missing this long, and click periodically to continue (clicks
# are button-presses in Idleon).
CELEBRATION_CHECK_AFTER_S = 6.0
CELEBRATION_CLICK_EVERY_S = 4.0

# Generic unknown-overlay handling: a second stuck-mid-game report
# (2026-06-10 12:01) was NOT the 180 banner — some other popup replaced
# the throwing pose and the bot bailed blind. Templating popups one at a
# time is whack-a-mole; instead, when the pose has been missing this
# long AND the frame is definitely not the game-over screen (conf well
# below threshold), probe-click the center a few times — Idleon popups
# dismiss on click. A diagnostic frame is saved on the first probe and
# at bail time so the next unknown overlay arrives with evidence.
UNKNOWN_OVERLAY_PROBE_AFTER_S = 10.0
MAX_OVERLAY_PROBE_CLICKS = 3
OVERLAY_PROBE_GO_CONF_MAX = 0.5  # only probe when clearly not game over
# Probe only when the player's swing is NOT visible in the arm-motion
# mask. Session 2026-06-10 12:08: a sub-threshold spawn looked like a
# stall, the probes threw two blind darts into live play and burned two
# lives (the diagnostic frames show hearts going 3 -> 1). Live swings
# show arm_pixel_count ~150-330 every poll; a scene-covering or
# scene-freezing popup collapses it. Visible-but-unmatched swings are
# the adaptive threshold's job, not the probes'.
OVERLAY_PROBE_MAX_ARM_MOTION = 80
DIAGNOSTICS_DIR = Path(__file__).parent / "assets" / "diagnostics"

# Per-spawn adaptive threshold (#26, the original complaint): the player
# teleports between spawns, and at some spawns the release template
# peaks BELOW the fixed threshold — the 2026-06-10 12:01 stall watched
# the swing cycle at conf 0.62-0.66 for 25s without firing. When no
# pose has matched for a while but a credible sub-threshold match has
# been seen, drop the effective threshold to just under that observed
# ceiling. Safe now that the dy-gate (not the threshold) is what
# prevents wrong-moment fires.
ADAPT_THRESHOLD_AFTER_S = 8.0
MIN_RELEASE_THRESHOLD = 0.55  # never adapt below this — noise floor headroom
ADAPT_MARGIN = 0.03           # fire at (observed ceiling - this)


def _effective_release_threshold(
    elapsed_no_pose_s: float,
    best_recent_conf: float,
    base: float,
) -> float:
    """The release threshold for the current poll. Normally `base`; after
    a long pose drought with a credible best-recent match, adapt down to
    just under that spawn's observed match ceiling."""
    if (
        elapsed_no_pose_s > ADAPT_THRESHOLD_AFTER_S
        and best_recent_conf >= MIN_RELEASE_THRESHOLD + ADAPT_MARGIN
    ):
        return max(MIN_RELEASE_THRESHOLD, min(base, best_recent_conf - ADAPT_MARGIN))
    return base

# Swing-pass fire gate (#26): skip the fire when the arm-centroid
# vertical velocity is positive — the up-swing pass signature.
# Validated against launch-angle ground truth across every instrumented
# fire (sessions 2026-05-24 .. 2026-06-10): vy > 0 fires hit ~0/15
# (launch angles +20..26°, the guaranteed up-swing miss), vy <= ~-28
# px/s fires hit ~21/24 with all exceptions explained by the #40
# high-streak zone shift. vy=None (first poll, or arm-motion dropout)
# fires rather than starves — the signal is an instrument, not a
# precondition.
#
# UNITS (since 2026-06-10): px/second, cadence-invariant. Historical
# arm_centroid_dy_at_fire was px/poll at the ~250ms compute-bound
# cadence — multiply by ~4 to compare. The change unblocks poll-loop
# speedups: per-poll units would silently rescale with every tick-cost
# optimization.
#
# Replaces the #25 arm-area apex gate (MAX_ARM_AREA_AT_FIRE=210): its
# N=16 calibration ("HIT 174..197, MISS 204..1665") did not survive more
# data — both classes span 174..220 in the fuller record — so it both
# missed most up-swings and risked skipping good releases.
VY_GATE_MAX = 0  # fire only when arm_centroid_vy <= this (or is None)


def _is_upswing_fire(arm_centroid_vy: int | None) -> bool:
    """True when the centroid-vy signal says this match is the up-swing
    pass (skip it); False fires — including when vy is unavailable."""
    return arm_centroid_vy is not None and arm_centroid_vy > VY_GATE_MAX


# vy-band aim (#41 step 1): WITHIN the release pass, where the click
# lands steers the arc — vy near 0-from-below = late-pass = flat arc =
# mid-board red/bullseye (see CLAUDE.md darts findings). Prefer fires
# inside the band; skip up to MAX_AIM_SKIPS passes waiting for one (a
# skipped pass costs one ~2-4s swing cycle), then take any valid pass
# rather than starve. Every EXPLORE_EVERY_N-th throw fires on the
# first valid pass regardless, so the vy→outcome surface keeps getting
# sampled outside the band — darts has no free shots, so exploration
# is budgeted rather than free (unlike hoops).
#
# [-32, -20] px/s is the exact px/s image of the original [-8, -5]
# px/poll band: the 90 recorded band fires span -38..-19 px/s with
# median -28 (per-fire dt from the polls table; median gap 253ms).
VY_AIM_LO = -32
VY_AIM_HI = -20
MAX_AIM_SKIPS = 2
EXPLORE_EVERY_N = 8


def _aim_fire_decision(
    arm_centroid_vy: int | None,
    aim_skips: int,
    explore_throw: bool,
    model_band: tuple[int, int] | None = None,
) -> tuple[bool, str | None]:
    """Decide whether a valid (non-up-swing) match should fire, and tag
    the intent. Returns (fire, aim_mode):
      'model'    — vy inside the wind-conditioned E[stripe] band
                   (#41 step 3; supplied by the caller when the GP is
                   fitted and the wind parse succeeded this poll)
      'band'     — vy inside the static calm-air EV band (model
                   unavailable)
      'explore'  — ε-exploration throw, fires on any valid pass
      'fallback' — band (or a vy reading) not seen within the skip
                   budget
      (False, None) — wait for the next pass.
    Up-swing rejection happens before this (see _is_upswing_fire).

    vy=None passes consume the skip budget instead of firing instantly
    (changed 2026-06-11): the original "instrument dropout must not
    starve" rule predates the budget and turned toxic at the faster
    poll cadence — post-sprint blind fires hit 3/10 vs ~75% for
    vy-known fires, and the 01:06 session threw three of them in 9s at
    a near-static pose right after a wind change. Starvation is still
    impossible: the budget fallback below fires after MAX_AIM_SKIPS
    passes regardless of whether vy ever came back.
    """
    if explore_throw and arm_centroid_vy is not None:
        return True, "explore"
    if arm_centroid_vy is not None:
        if model_band is not None:
            if model_band[0] <= arm_centroid_vy <= model_band[1]:
                return True, "model"
        elif VY_AIM_LO <= arm_centroid_vy <= VY_AIM_HI:
            return True, "band"
    if aim_skips >= MAX_AIM_SKIPS:
        return True, "fallback"
    return False, None

# Stripe color → score-increment mapping (confirmed 2026-05-24 from user
# session reporting 3 red/bullseye hits + the +1/+2/+3/+5 stripe values
# in STRATEGY.md). Lookup is increment → color so log_throw can derive
# stripe_color directly from score_increment without OCR-color detection.
STRIPE_COLOR_BY_INCREMENT: dict[int, str] = {
    5: "red",     # bullseye, center stripe
    3: "green",
    2: "yellow",
    1: "gray",
}

# Valid post-throw score increments. Tesseract loves misreading the +5
# bullseye as e.g. 195 or -4 (and occasionally NULL); constraining the
# increment to this set lets us reject obviously wrong OCR reads and
# vote among candidates that pass the sanity check.
VALID_SCORE_INCREMENTS = frozenset(STRIPE_COLOR_BY_INCREMENT.keys())


def _pick_best_score_after(
    score_before_int: int | None,
    post_readings: list,
) -> tuple:
    """Choose (score_after_int, score_increment) from a list of OCR
    candidates by requiring the increment to be in VALID_SCORE_INCREMENTS.
    Vote among valid candidates by frequency; fall back to the last
    non-None reading when nothing passes the sanity check.

    Returns (best_after, best_increment). Either can be None if no
    candidate is usable.
    """
    from collections import Counter
    if score_before_int is None:
        for c in reversed(post_readings):
            if c is not None:
                return c, None
        return None, None
    valid = [(c - score_before_int, c) for c in post_readings
             if c is not None and (c - score_before_int) in VALID_SCORE_INCREMENTS]
    if valid:
        incr_counts = Counter(t[0] for t in valid)
        best_incr = incr_counts.most_common(1)[0][0]
        best_after = next(a for i, a in valid if i == best_incr)
        return best_after, best_incr
    # No valid increment — fall back to last non-None reading (preserves
    # the pre-multi-pass behavior so we don't lose information when OCR
    # is completely broken for this throw).
    for c in reversed(post_readings):
        if c is not None:
            return c, c - score_before_int
    return None, None


def _crop_wind(frame_bgra) -> np.ndarray | None:
    bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    h_img, w_img = bgr.shape[:2]
    region = get_region(_HERE, "wind", w_img, h_img)
    if region is None:
        return None
    x0 = max(0, region["left"])
    y0 = max(0, region["top"])
    x1 = min(w_img, region["left"] + region["width"])
    y1 = min(h_img, region["top"] + region["height"])
    return bgr[y0:y1, x0:x1]


def _match_or_save_wind_sample(wind_crop: np.ndarray, seen: list) -> str | None:
    """Return the name of the saved wind sample this crop matches, saving
    it as a new sample first when the state hasn't been seen before.

    `seen` holds (name, image) pairs and is mutated in place. The name is
    what log_throw records as wind_sample — without it the throws table
    can't answer "what does wind state X do to landing_x", which is the
    input the wind-conditioned release predictor needs (the column was
    silently NULL on every throw before 2026-06-10).

    Returns None when there's no wind crop (wind region not picked).
    """
    if wind_crop is None or wind_crop.size == 0:
        return None
    for name, ref in seen:
        if ref.shape != wind_crop.shape:
            continue
        diff = float(cv2.absdiff(wind_crop, ref).astype(np.float32).mean())
        if diff < WIND_DEDUP_THRESHOLD:
            return name
    WIND_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    # Index suffix: the bare %H%M%S stamp collided (and silently
    # overwrote) when two new states arrived within one second.
    stamp = datetime.now().strftime("%H%M%S")
    name = f"sample_{stamp}_{len(seen):03d}.png"
    cv2.imwrite(str(WIND_SAMPLES_DIR / name), wind_crop)
    seen.append((name, wind_crop))
    return name


def _load_existing_wind_samples() -> list:
    if not WIND_SAMPLES_DIR.exists():
        return []
    samples = []
    for p in sorted(WIND_SAMPLES_DIR.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            samples.append((p.name, img))
    return samples


def _save_monitor_throw(
    throw_idx: int,
    pre_frame_bgra: np.ndarray,
    pose: tuple[int, int],
    conf: float,
    wind_crop: np.ndarray | None,
    post_frame_bgra: np.ndarray,
    score_before,
    score_after,
    score_diff: float | None,
    score_changed_flag: bool | None,
    sub: Path | None = None,
) -> Path:
    # When the caller has already allocated the throw folder (e.g. to
    # write flight frames during the cooldown), reuse it instead of
    # creating a new one — keeps everything for one throw in one place.
    if sub is None:
        sub = make_shot_dir(MONITOR_DIR, throw_idx, prefix="throw")
    save_frame(sub / "pre_throw.png", pre_frame_bgra)
    save_frame(sub / "post_throw.png", post_frame_bgra)
    if wind_crop is not None and wind_crop.size > 0:
        cv2.imwrite(str(sub / "wind.png"), wind_crop)
    save_meta(
        sub / "meta.txt",
        timestamp=datetime.now().isoformat(),
        release_pose=f"({pose[0]},{pose[1]})",
        release_conf=f"{conf:.3f}",
        score_diff=score_diff if score_diff is not None else "n/a",
        score_changed=score_changed_flag if score_changed_flag is not None else "n/a",
    )
    return sub


def _capture_score(left: int, top: int, width: int, height: int):
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


def _log_shot_result(stats: dict, before, after) -> None:
    if before is None or after is None:
        return
    changed, diff = score_changed(before, after, threshold=SCORE_CHANGE_THRESHOLD)
    stats["attempts"] += 1
    if changed:
        stats["makes"] += 1
        stats["streak"] = stats.get("streak", 0) + 1
        streak = stats["streak"]
        suffix = f" | streak {streak}" if streak >= 2 else ""
        print(f"  [score] HIT (diff={diff:.1f}) | session {stats['makes']}/{stats['attempts']}{suffix}")
        if STEAM_SCREENSHOT_ON_NINE_DART and streak == NINE_DART_STREAK:
            print(f"  [nine-dart] {NINE_DART_STREAK}-hit streak reached — pressing F12 for Steam screenshot")
            press_key("f12")
            # Don't reset — Steam dedupes its own screenshot key, but we
            # also don't want to spam if the streak continues past 9.
            # Reset the counter so the next screenshot only fires after
            # another full streak.
            stats["streak"] = 0
    else:
        if stats.get("streak", 0) > 0:
            print(f"  [streak] reset (was {stats['streak']})")
        stats["streak"] = 0
        print(f"  [score] miss (diff={diff:.1f}) | session {stats['makes']}/{stats['attempts']}")


def run():
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        session_started = datetime.now().isoformat(timespec="seconds")
        code_commit = current_code_commit(REPO_ROOT)
        if code_commit:
            print(f"Code commit: {code_commit}")
        throw_db = open_db(THROW_DB_PATH)
        # E[stripe] GP (#41 step 3): wind-conditioned dy band. Fitted
        # once per session — the training set only grows between
        # sessions, and a startup fit keeps the poll loop allocation-free.
        ev_rows = fetch_stripe_rows(throw_db)
        ev_model = fit_stripe_gp(ev_rows)
        if ev_model is None:
            print(f"stripe EV model: data floor not met ({len(ev_rows)} rows "
                  f"< {STRIPE_MIN_SAMPLES}) — static vy band [{VY_AIM_LO},{VY_AIM_HI}] px/s")
        else:
            print(f"stripe EV model: fitted on {len(ev_rows)} throws — "
                  f"wind-conditioned dy band live (#41 step 3)")
        try:
            _run_inner(session_started, throw_db, code_commit, ev_model)
        finally:
            throw_db.close()
            # The DB is tracked so other machines get session data via
            # `git pull`. Safe to commit here: the connection is closed.
            commit_file_if_changed(
                REPO_ROOT,
                "minigames/darts/assets/darts.db",
                "chore(darts): refresh darts.db (auto)",
            )


def _run_inner(
    session_started: str,
    throw_db,
    code_commit: str | None,
    ev_model: StripeEvModel | None = None,
):
    print(f"Darts bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    shot_stats: dict = {"makes": 0, "attempts": 0}
    throws_taken = 0  # increments every throw, independent of score detection
    best_recent_conf = 0.0  # for visibility into how close the matcher is getting between shots
    wind_seen = _load_existing_wind_samples()
    if wind_seen:
        print(f"Loaded {len(wind_seen)} existing wind samples; will only save new states.")
    last_pose_time = time.time()
    last_wind_crop: np.ndarray | None = None
    last_wind_parse: tuple | None = None  # (speed, rounded angle) last announced
    # Temporal diagnostics for the apex-vs-forward-release discriminator
    # (see STRATEGY.md "Game-physics assumptions"). match_streak_len counts
    # consecutive POLL_INTERVAL frames where pose was detected; prev_match_y
    # is the dart y from the immediately preceding matched frame in the
    # current streak. Both reset whenever pose drops below threshold.
    match_streak_len = 0
    prev_match_y: int | None = None
    # Wall-clock anchor for the polls table. Every poll's t_ms is
    # relative to this so we can correlate poll rows with throws rows
    # post-hoc without a perfectly synced clock.
    session_t0 = time.time()
    # Frame history for arm-motion centroid computation. We diff the
    # current frame against one ~REF_MOTION_DT_S old — NOT simply the
    # previous poll — so the per-diff motion magnitude (and MIN_AREA's
    # dropout behavior) stays calibrated no matter how fast the poll
    # loop runs. Each entry is (wall_time, bgr).
    motion_history: list[tuple[float, np.ndarray]] = []
    prev_arm_centroid_y: int | None = None  # last VALID centroid, for the vy discriminator
    prev_arm_centroid_t: float | None = None  # wall time of that centroid
    last_celebration_click = 0.0  # rate-limits dismiss-clicks during the streak banner
    overlay_probe_clicks = 0  # unknown-overlay probes since the last real pose match
    aim_skips = 0  # out-of-band passes skipped since the last fire (#41 dy-band aim)

    while True:
        check_failsafe()
        try:
            left, top, width, height = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        frame = grab_region(left, top, width, height)
        cur_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Fast template check for the game-over screen first. Returns
        # (False, 0.0) if the template hasn't been captured yet — falls
        # through to the no-pose timeout heuristic below.
        is_over, go_conf = find_game_over(frame)
        if is_over:
            print(f"Game over detected (conf={go_conf:.2f}). Final session: "
                  f"{shot_stats['makes']}/{shot_stats['attempts']} hits.")
            return

        # Wind-change watch. The game flips the wind panel (plus a "The
        # wind has changed" toast just below it — outside the wind
        # region, so no occlusion) during the PREVIOUS throw's cooldown;
        # see monitor/throw_012_233144 flight_008->014. The per-poll
        # model band and the at-fire DB snapshot both read the live
        # panel, so aim and attribution were already fresh — but the
        # change used to be printed only as post-click bookkeeping of
        # the NEXT throw, which read in the log as one-throw-late
        # detection (user report, 2026-06-10 23:30 session). Announce it
        # the poll it appears instead. The pixel diff gates the ~ms
        # parse; comparing the parsed (speed, angle) suppresses repeat
        # announcements from panel animation frames.
        wind_now = _crop_wind(frame)
        if wind_now is not None and wind_now.size:
            if last_wind_crop is None or wind_now.shape != last_wind_crop.shape:
                last_wind_crop = wind_now
            else:
                wind_diff = float(cv2.absdiff(wind_now, last_wind_crop).astype(np.float32).mean())
                if wind_diff >= WIND_DEDUP_THRESHOLD:
                    last_wind_crop = wind_now
                    w_speed, w_angle, w_x, w_y = parse_wind(wind_now)
                    parse_key = (w_speed, None if w_angle is None else round(w_angle))
                    if parse_key != last_wind_parse:
                        last_wind_parse = parse_key
                        desc = (f"{w_speed} mph @ {w_angle:+.0f}° -> components ({w_x:+.1f}, {w_y:+.1f})"
                                if w_speed else "calm")
                        print(f"  [wind] changed (diff={wind_diff:.1f}): {desc}")

        # threshold=0 so we get a position+conf for every poll, including
        # sub-threshold ones — required for the polls table. The fire
        # decision still gates on RELEASE_THRESHOLD below.
        pose, conf = find_release_pose(frame, threshold=0.0)
        match_x = int(pose[0]) if pose else 0
        match_y = int(pose[1]) if pose else 0
        t_ms = int((time.time() - session_t0) * 1000)
        now = time.time()
        # Reference frame for the motion diff: the history entry whose
        # age is CLOSEST to REF_MOTION_DT_S (None during the first
        # ~250ms). Was "newest entry >= REF_MOTION_DT_S old", which at
        # the ~250ms cadence meant exactly 250ms but at the post-sprint
        # ~119ms cadence silently widened the window to ~357ms — the
        # arm-motion dropout rate tripled (1-3% -> 6-9% of polls)
        # starting exactly at the sprint sessions. Closest-to keeps the
        # diff spacing that MIN_AREA was calibrated against at any
        # cadence.
        ref_bgr = None
        best_age_err = None
        for ft, fb in motion_history:
            age = now - ft
            if age < 0.6 * REF_MOTION_DT_S:
                break  # too fresh for a calibrated diff; later entries only fresher
            err = abs(age - REF_MOTION_DT_S)
            if best_age_err is None or err < best_age_err:
                best_age_err = err
                ref_bgr = fb
        motion_history.append((now, cur_bgr))
        while motion_history and now - motion_history[0][0] > 2 * REF_MOTION_DT_S:
            motion_history.pop(0)
        arm_centroid_y, arm_pixel_count = compute_arm_centroid(cur_bgr, ref_bgr)
        # Swing-pass discriminator (#26): arm-centroid vertical velocity
        # (px/s) vs the last valid centroid. Validated post-hoc
        # 2026-06-10 against launch-angle ground truth over 23 fires:
        # vy < 0 at fire → 9/10 hits, vy > 0 → 10/11 misses. (The first
        # candidate — re-matching the dart template in the previous
        # frame — returned None on every live fire: the dart ROTATES
        # between polls at the ~250ms cadence, so the release-angle
        # template can't match the tilted prev-frame dart.) Dropout
        # bridging (#42) is inherent in the time denominator: dt is the
        # real elapsed time since the last valid centroid, capped by
        # MAX_VY_DT_S inside centroid_vy_px_s.
        arm_centroid_vy = centroid_vy_px_s(
            arm_centroid_y,
            prev_arm_centroid_y,
            (now - prev_arm_centroid_t) if prev_arm_centroid_t is not None else 0.0,
        )
        if arm_centroid_y is not None:
            prev_arm_centroid_y = arm_centroid_y
            prev_arm_centroid_t = now

        effective_threshold = _effective_release_threshold(
            time.time() - last_pose_time, best_recent_conf, RELEASE_THRESHOLD,
        )
        if conf < effective_threshold:
            log_poll(
                throw_db, session_started, t_ms, conf, match_x, match_y, threw=0,
                arm_centroid_y=arm_centroid_y, arm_pixel_count=arm_pixel_count,
            )
            best_recent_conf = max(best_recent_conf, conf)
            # Streak ends whenever pose drops below threshold — reset both
            # diagnostic state vars so the next match starts a fresh streak.
            match_streak_len = 0
            prev_match_y = None
            # Streak celebration check: the banner replaces the throwing
            # pose, so a long no-pose stretch may be a celebration, not
            # game over. Keep the timeout from bailing while it's up and
            # click periodically to continue.
            elapsed_no_pose = time.time() - last_pose_time
            if elapsed_no_pose > CELEBRATION_CHECK_AFTER_S:
                celebrating, celeb_conf = find_celebration(frame)
                if celebrating:
                    last_pose_time = time.time()
                    if time.time() - last_celebration_click > CELEBRATION_CLICK_EVERY_S:
                        print(f"  [celebration] streak banner up (conf={celeb_conf:.2f}) — clicking to continue")
                        click(left + width // 2, top + height // 2)
                        last_celebration_click = time.time()
                elif (
                    elapsed_no_pose > UNKNOWN_OVERLAY_PROBE_AFTER_S
                    and go_conf < OVERLAY_PROBE_GO_CONF_MAX
                    and overlay_probe_clicks < MAX_OVERLAY_PROBE_CLICKS
                    and time.time() - last_celebration_click > CELEBRATION_CLICK_EVERY_S
                    and (arm_pixel_count is None or arm_pixel_count < OVERLAY_PROBE_MAX_ARM_MOTION)
                ):
                    # Unknown overlay: not the celebration, clearly not the
                    # game-over screen, and the pose is long gone. Save
                    # evidence, then probe-click — Idleon popups dismiss
                    # on click. The pose reappearing resets the budget.
                    if overlay_probe_clicks == 0:
                        DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
                        diag = DIAGNOSTICS_DIR / f"nopose_{datetime.now():%Y%m%d_%H%M%S}.png"
                        save_frame(diag, frame)
                        print(f"  [overlay] saved diagnostic frame to {diag}")
                    overlay_probe_clicks += 1
                    print(f"  [overlay] unknown screen state (no pose {elapsed_no_pose:.0f}s, "
                          f"go_conf={go_conf:.2f}) — probe click {overlay_probe_clicks}/{MAX_OVERLAY_PROBE_CLICKS}")
                    click(left + width // 2, top + height // 2)
                    last_celebration_click = time.time()
            # Game-over signal: the entire dartboard scene is replaced when
            # the trial ends, so the player avatar disappears and the release
            # template can't match. If we haven't seen the player in a while,
            # bail.
            if time.time() - last_pose_time > GAME_OVER_NO_POSE_SEC:
                throw_db.commit()  # flush any unflushed polls before exit
                DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
                diag = DIAGNOSTICS_DIR / f"bail_{datetime.now():%Y%m%d_%H%M%S}.png"
                save_frame(diag, frame)
                print(f"No release pose detected for {GAME_OVER_NO_POSE_SEC:.0f}s — "
                      f"assuming game over (bail frame: {diag}). Final session: "
                      f"{shot_stats['makes']}/{shot_stats['attempts']} makes.")
                return
            time.sleep(POLL_INTERVAL)
            continue

        # Swing-pass skip-gate (#26). Bot's about to fire but centroid-vy
        # says this match is the up-swing pass — a guaranteed +20° launch
        # into the bottom of the screen. Logs the candidate as threw=0 so
        # the polls table reflects the skip, then waits for the release
        # pass (~250ms later).
        if _is_upswing_fire(arm_centroid_vy):
            log_poll(
                throw_db, session_started, t_ms, conf, match_x, match_y, threw=0,
                arm_centroid_y=arm_centroid_y, arm_pixel_count=arm_pixel_count,
            )
            last_pose_time = time.time()  # don't fire, but keep game-over heuristic happy
            overlay_probe_clicks = 0
            print(f"  [skip] vy-gate: centroid_vy={arm_centroid_vy:+d} px/s > "
                  f"{VY_GATE_MAX} — up-swing pass, waiting for release")
            time.sleep(POLL_INTERVAL)
            continue

        # vy-band aim (#41): valid release pass, but is it the arc we
        # want? Band fires hit mid-board red/bullseye; out-of-band passes
        # get skipped within a budget; ε-exploration keeps sampling the
        # whole surface. With the E[stripe] GP fitted, the band is
        # wind-conditioned (step 3): parse the wind panel from the
        # current frame (a ~ms template-OCR on a small crop — acceptable
        # pre-click latency) and scan the model for the near-argmax vys.
        # Unparseable wind or no model falls back to the static band.
        explore_throw = (throws_taken + 1) % EXPLORE_EVERY_N == 0
        model_band: tuple[int, int] | None = None
        model_best_vy: int | None = None
        model_best_ev: float | None = None
        if (
            ev_model is not None
            and arm_centroid_vy is not None
            and not explore_throw
            # Outside the training data's pose_y support the band is GP
            # mean-reversion junk — seen live when the (146, 38) phantom
            # match slipped past the adapted conf gate. Static band
            # handles those passes.
            and ev_model.supports_pose(float(pose[1]))
        ):
            _, _, wx_now, wy_now = parse_wind(_crop_wind(frame))
            if wx_now is not None:
                model_band, model_best_vy, model_best_ev = model_vy_band(
                    ev_model, wx_now, wy_now, float(pose[1]),
                )
        fire, aim_mode = _aim_fire_decision(
            arm_centroid_vy, aim_skips, explore_throw, model_band,
        )
        if not fire:
            log_poll(
                throw_db, session_started, t_ms, conf, match_x, match_y, threw=0,
                arm_centroid_y=arm_centroid_y, arm_pixel_count=arm_pixel_count,
            )
            last_pose_time = time.time()
            overlay_probe_clicks = 0
            aim_skips += 1
            band_desc = (
                f"model band [{model_band[0]},{model_band[1]}] px/s "
                f"(best vy={model_best_vy}, ev={model_best_ev:.2f})"
                if model_band is not None
                else f"band [{VY_AIM_LO},{VY_AIM_HI}] px/s"
            )
            vy_desc = f"{arm_centroid_vy:+d}" if arm_centroid_vy is not None else "?"
            print(f"  [aim] vy={vy_desc} px/s outside {band_desc} — "
                  f"waiting for a better pass (skip {aim_skips}/{MAX_AIM_SKIPS})")
            time.sleep(POLL_INTERVAL)
            continue
        aim_skips = 0

        log_poll(
            throw_db, session_started, t_ms, conf, match_x, match_y, threw=1,
            arm_centroid_y=arm_centroid_y, arm_pixel_count=arm_pixel_count,
        )
        last_pose_time = time.time()
        overlay_probe_clicks = 0
        px, py = pose
        match_streak_len += 1
        prev_match_y_for_log = prev_match_y  # capture for log_throw below
        prev_match_y = int(py)
        # Lead-up gate (commit da45a41) reverted 2026-05-24: it
        # consistently fired the up-swing/apex matches (all 7 throws at
        # +20-23° launch angle), exactly the wrong half. The hypothesis
        # was built on N=2 captured streaks at one player position; the
        # live bot at a different spawn showed the OPPOSITE pattern.
        # Insufficient data to commit to a directional rule.
        cvy_tag = (f", centroid_vy={arm_centroid_vy:+d} px/s [{aim_mode}]"
                   if arm_centroid_vy is not None else f", centroid_vy=? [{aim_mode}]")
        print(f"Release pose at ({px},{py}), conf={conf:.2f}{cvy_tag} "
              f"(recent best while waiting={best_recent_conf:.2f}, streak={match_streak_len}"
              + (f", adapted thr={effective_threshold:.2f}" if effective_threshold < RELEASE_THRESHOLD else "")
              + ") — throwing")
        # Fire immediately. Bookkeeping (score capture, wind crop +
        # diff + sample save) runs after the click — every ms between
        # pose detection and click landing drifts the arm a few degrees
        # off the captured release angle. Same principle as hoops; see
        # CLAUDE.md "Click timing".
        fired_at = datetime.now().isoformat(timespec="seconds")
        click(left + width // 2, top + height // 2)
        # Snapshot wind state from the pre-click frame (still valid —
        # `frame` is the BGRA buffer captured before the pose match).
        wind_crop = _crop_wind(frame)
        # Capture the score region post-click: the in-game score only
        # updates after the dart lands, so the region still shows the
        # pre-throw value here.
        score_before = _capture_score(left, top, width, height)
        # (Wind-change announcements happen in the per-poll watch at the
        # top of the loop, the poll the panel flips — not here. This
        # block only attributes the fire-frame wind to the throw row.)
        n_wind_before = len(wind_seen)
        wind_sample_name = _match_or_save_wind_sample(wind_crop, wind_seen)
        wind_speed, wind_angle, wind_x, wind_y = parse_wind(wind_crop)
        if wind_speed:
            print(f"  [wind] {wind_speed} mph @ {wind_angle:+.0f}° -> components ({wind_x:+.1f}, {wind_y:+.1f})")
        if len(wind_seen) > n_wind_before:
            print(f"  [wind] new wind state saved (total samples: {len(wind_seen)})")
        # Per-throw monitor folder is allocated up-front so flight frames
        # have a destination. throws_taken hasn't been incremented yet —
        # do it now so the folder index matches the meta written below.
        throws_taken += 1
        flight_dir: Path | None = None
        if MONITOR_MODE:
            flight_dir = make_shot_dir(MONITOR_DIR, throws_taken, prefix="throw")
        # Sample flight frames during the cooldown instead of just sleeping.
        # Frames go to the throw folder so a later trajectory module can
        # extract dart-landing-x without a second live session.
        # Also accumulate score-region crops from those same frames for
        # multi-pass OCR after the cooldown — costs nothing extra in
        # capture latency since we've already grabbed the full frame.
        score_region_meta = get_region(_HERE, "score", width, height)
        score_crops_during_flight: list = []
        if MONITOR_FLIGHT and flight_dir is not None:
            flight_deadline = time.time() + POST_THROW_COOLDOWN + POST_LAND_DELAY
            flight_idx = 0
            while time.time() < flight_deadline:
                check_failsafe()
                flight_idx += 1
                f = grab_region(left, top, width, height)
                bgr = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
                cv2.imwrite(str(flight_dir / f"flight_{flight_idx:03d}.png"), bgr)
                if score_region_meta is not None:
                    score_crops_during_flight.append(score_region(
                        f,
                        score_region_meta["left"], score_region_meta["top"],
                        score_region_meta["width"], score_region_meta["height"],
                    ))
                time.sleep(FLIGHT_POLL)
        else:
            time.sleep(POST_THROW_COOLDOWN)
            time.sleep(POST_LAND_DELAY)
        post_frame = grab_region(left, top, width, height)
        # Read the post-throw score from THIS frame (the one saved as
        # post_throw.png) rather than a fresh grab. A separate grab can
        # catch the score one render-frame before the leading digit lands,
        # dropping it (13 -> 3) and faking a negative increment — the #40
        # "busts" were all this artifact, while the saved post_throw.png
        # always read correctly (verified across 545 throws).
        score_after = (
            score_region(
                post_frame,
                score_region_meta["left"], score_region_meta["top"],
                score_region_meta["width"], score_region_meta["height"],
            )
            if score_region_meta is not None else None
        )
        # Compute the diff once so we can log AND save it to meta.
        diff_val = None
        diff_changed = None
        if score_before is not None and score_after is not None:
            diff_changed, diff_val = score_changed(score_before, score_after, threshold=SCORE_CHANGE_THRESHOLD)
        _log_shot_result(shot_stats, score_before, score_after)
        # OCR the score crops — gives us the actual increment (+1/+2/+3/+5)
        # rather than just hit/miss. None when tesseract isn't installed
        # or the digit read fails.
        score_before_int = read_score(score_before) if score_before is not None else None
        # Multi-pass: combine the last N flight crops (score should be
        # post-update by then) with the final score_after. _pick_best_*
        # filters readings to those producing an increment in {1,2,3,5}
        # and votes — directly addresses tesseract misreading +5 bullseyes
        # as 195/-4/null. Falls back to the last reading if no candidate
        # passes the sanity check, preserving the pre-multi-pass behavior.
        post_score_candidates = list(score_crops_during_flight[-5:]) + [score_after]
        post_readings = [
            read_score(crop) if crop is not None else None
            for crop in post_score_candidates
        ]
        score_after_int, score_increment = _pick_best_score_after(
            score_before_int, post_readings
        )
        # Trajectory analysis on the captured flight frames — angle,
        # apex, landing. Empty dict if no flight frames captured this
        # throw or the dart wasn't detected in any frame.
        trajectory: dict = {
            "launch_angle_deg": None,
            "apex_y": None,
            "landing_x": None,
            "frames_seen": 0,
        }
        if MONITOR_FLIGHT and flight_dir is not None:
            try:
                trajectory = analyse_throw_dir(flight_dir)
            except Exception as e:
                print(f"  [trajectory] analysis failed (non-fatal): {e}")
        # bullseye = score_increment of exactly 5 (per Throwy Darts'
        # +1/+2/+3/+5 stripe scoring). Used by future Nine-Dart-Finish
        # streak tightening.
        bullseye = (
            1 if score_increment == 5 else (0 if score_increment is not None else None)
        )
        stripe_color = STRIPE_COLOR_BY_INCREMENT.get(score_increment) if score_increment else None
        log_throw(
            throw_db,
            session_started=session_started,
            throw_idx=throws_taken,
            fired_at=fired_at,
            release_pose_x=int(px),
            release_pose_y=int(py),
            release_conf=float(conf),
            arm_centroid_vy_at_fire=arm_centroid_vy,
            aim_mode=aim_mode,
            wind_sample=wind_sample_name,
            wind_speed=wind_speed,
            wind_angle_deg=wind_angle,
            wind_x=wind_x,
            wind_y=wind_y,
            launch_angle_deg=trajectory["launch_angle_deg"],
            apex_y=trajectory["apex_y"],
            landing_x=trajectory["landing_x"],
            frames_seen=trajectory["frames_seen"],
            score_before_int=score_before_int,
            score_after_int=score_after_int,
            score_increment=score_increment,
            hit=int(bool(diff_changed)) if diff_changed is not None else None,
            bullseye=bullseye,
            streak=shot_stats.get("streak", 0),
            throw_dir=str(flight_dir) if flight_dir is not None else None,
            window_w=int(width),
            window_h=int(height),
            code_commit=code_commit,
            source="bot",
            match_streak_len_before_fire=int(match_streak_len),
            prev_match_y=prev_match_y_for_log,
            stripe_color=stripe_color,
        )
        if MONITOR_MODE:
            sub = _save_monitor_throw(
                throw_idx=throws_taken,
                pre_frame_bgra=frame,
                pose=(px, py),
                conf=conf,
                wind_crop=wind_crop,
                post_frame_bgra=post_frame,
                score_before=score_before,
                score_after=score_after,
                score_diff=diff_val,
                score_changed_flag=diff_changed,
                sub=flight_dir,
            )
            print(f"  [monitor] saved {sub.name}")
        best_recent_conf = 0.0
        # Reset streak state after the cooldown — the next match is the
        # start of a fresh swing cycle, not a continuation of this one.
        match_streak_len = 0
        prev_match_y = None


if __name__ == "__main__":
    run()
