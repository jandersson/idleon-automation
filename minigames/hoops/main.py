import time
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay, check_failsafe
from common.monitor import make_shot_dir, save_frame, save_meta
from common.regions import get_region
from common.session_log import session_log
from common.shot_log import open_db, log_shot
from common.window import get_bounds, WindowNotFoundError
from minigames.hoops.detector import find_rim, find_platform, find_ball, find_game_over, score_region, score_changed

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
SHOT_DB_PATH = _HERE / "assets" / "shots.db"

WINDOW_TITLE = "Legends Of Idleon"
POLL_INTERVAL = 0.005  # Tight loop: each find_platform call already takes
                       # ~15-30ms (multi-scale matching), so this sleep is
                       # mostly irrelevant — we're CPU-bound on the matcher.
                       # Lowered from 0.02 anyway to make sure sleep isn't
                       # the bottleneck on cycles where matching is fast.

# Position-dependent vertical offset added to hoop rim Y to get the ideal
# platform-launch Y. Rolls rim-vs-platform geometry AND ball-travel lead into
# one number per hoop position. Linearly interpolated between anchor points;
# extrapolation is clamped to the nearest anchor's value.
#
# Aim point: per community strategy, target the BOTTOM of the rim opening,
# not the center -- nothing-but-net hits score 2 points instead of 1. The
# matched hoop template center is roughly the backboard middle, so the
# actual rim bottom is well below it; that informs the sign of offset
# corrections (positive = launch when platform is lower = ball arcs lower).
#
# To tune: watch where shots miss for a given hoop_y and add/move an anchor:
#   - Ball clears top of backboard → offset is too small → bump it up
#   - Ball hits front of rim       → offset is too large → bump it down
#   - Ball hits back of rim        → offset is too small → bump it up
# Two strategies for landing shots. Set SHOT_STRATEGY to switch.
#
# - "direct"   : tune launch timing so the ball arc passes through the rim
#                naturally. Mid-flight rescue still active as a backup but is
#                NOT the primary make mechanism. Offsets calibrated for shots
#                that go in by themselves.
# - "overshoot": deliberately aim past the rim; rescue drops the ball straight
#                down through the rim opening. Depends on the click-on-the-ball
#                drop trick actually working — to date unconfirmed in this
#                game version.
#
# Default to "direct" since (a) it's empirically known-good (6/7 makes in an
# earlier session) and (b) overshoot was failing with no evidence the drop
# trick activates as the wiki claims.
SHOT_STRATEGY = "direct"

OFFSET_ANCHORS_DIRECT: list[tuple[int, int]] = [
    # 960x572 window, REQUIRED_DIRECTION="up". Calibrated from the first
    # dir=up session (shots.db ids 5-10):
    # - hoop_y=371, offset=80 → big overshoot (ball over the backboard) ×2
    # - hoop_y=448, offset=33 → MAKE
    # - hoop_y=464, offset=31 → 2/3 MAKES; the miss fired late (py=494,
    #   close to target=495); makes fired earlier in upstroke (py=481, 489).
    #   Effective ideal target for hoop_y=464 is ~485 → offset ≈ 21.
    # Legacy (400, 80) and (416, 80) anchors removed — they were tuned for
    # dir=down at higher hoops and produced wrong slope under dir=up.
    (371, 25),   # was 50 — still overshot (back rim) at hoop_y=359 in
                 # the next session, with ball landing ~30-40px past
                 # hoop_x at rim height. Halve the offset.
    (450, 28),   # was 33. The hoop_y=448 makes happen when fired_py is
                 # below ~475; with offset=33 the platform fires anywhere
                 # 463..477 and only the lower end makes. Smaller offset
                 # shifts the trigger earlier in the upstroke (fired_py
                 # closer to 462) so makes are consistent.
    (464, 22),   # CONFIRMED make territory (fired at py=481-489).
    (474, 23),   # CONFIRMED make from session 17:15 (fired_py=495).
    (700, 50),   # untested in dir=up regime; legacy from dir=down.
    (835, 14),   # untested in dir=up; legacy.
    (900, 11),   # untested in dir=up; legacy.
]

OFFSET_ANCHORS_OVERSHOOT: list[tuple[int, int]] = [
    (700, 55),   # high hoops
    (835, 22),   # upper-mid
    (900, 18),   # mid-range
]

OFFSET_ANCHORS = OFFSET_ANCHORS_DIRECT if SHOT_STRATEGY == "direct" else OFFSET_ANCHORS_OVERSHOOT


def _compute_offset(hoop_y: int) -> int:
    pts = sorted(OFFSET_ANCHORS)
    if hoop_y <= pts[0][0]:
        return pts[0][1]
    if hoop_y >= pts[-1][0]:
        return pts[-1][1]
    for (y1, o1), (y2, o2) in zip(pts, pts[1:]):
        if y1 <= hoop_y <= y2:
            t = (hoop_y - y1) / (y2 - y1)
            return int(round(o1 + t * (o2 - o1)))
    return pts[-1][1]

# Accepted window around target_y (pixels) when deciding to fire.
Y_TOLERANCE = 2

# Required direction of platform motion to fire. "up", "down", or "any".
# Back to "up" after dir=down sweep on hoop_y=448: offsets 60->10 all hit
# the front of the rim (5px short, plateaued). Hypothesis: launch inherits
# platform velocity, so dir=down imparts downward bias → flat arc. dir=up
# adds upward bias → higher arc → more time aloft → more horizontal range.
# Earlier "dir=up overshot" note was at hoop_y=337 (much higher hoop, easier
# to overshoot); at lower hoops it should land closer to right.
REQUIRED_DIRECTION = "up"

# Wait after clicking so the ball can travel, land, and the score animation
# completes. Was 2.0 — but observed ball arrival at the rim is ~2.9s and the
# score updates after that. With 2.0s the post_shot snapshot consistently
# captured the OLD score, so every make was logged as miss (the next shot's
# pre_shot would show the updated score, but we don't compare across shots).
POST_SHOT_COOLDOWN = 4.0

# At score >=10 the platform also moves horizontally. We sample the platform's X
# during the early stationary phase, lock in the median as our anchor, and from
# then on require px to be within X_TOLERANCE of the anchor before firing.
HOME_X_SAMPLES = 10
X_TOLERANCE = 9999  # effectively disabled — re-enable with small value (e.g. 4) for score 10+

# Mid-flight rescue: after the launch click, watch for the ball. When it crosses
# over the hoop's X (still above the rim), click on it — the wiki trick that
# makes the ball drop straight down. Saves shots that would otherwise overshoot.
#
# Disabled by default: even with detection working (60-70/90 frames seen),
# the rescue session went 0/N. Pure-trajectory makes happen ~20-30% of the
# time at known-good offsets, and the rescue's mid-flight clicks seem to
# interfere more than they help. Flip back to True to A/B test.
RESCUE_ENABLED = False
RESCUE_WINDOW = 3.5  # seconds to track the ball after launch. Was 1.5 — but
                     # observed ball arrival at the rim is ~2.9s after click,
                     # so the rescue's "fire if ball is over hoop" check was
                     # being gated out by the deadline expiring before the
                     # ball got there.
# Strategy-dependent rescue tolerance:
# - direct:    wider, since rescue is a backup safety net, not the primary
#              make mechanism. We don't want it to interfere with shots that
#              would already make on their own.
# - overshoot: tight, since drop placement determines the make.
BALL_X_TOLERANCE = 6 if SHOT_STRATEGY == "overshoot" else 18
RESCUE_POLL = 0.01  # tight loop — ball moves fast

# Monitor mode: per-shot subfolder under assets/monitor/ with pre/post-shot
# screenshots, all frames captured during the rescue window (so we can see the
# full ball flight for offline review and ball-template extraction), and a
# meta.txt with shot details. Heavyweight (~200KB per shot) but invaluable
# for tuning offline. Toggle off in production.
MONITOR_MODE = True
MONITOR_DIR = _HERE / "assets" / "monitor"

# Capture flight frames during the post-shot cooldown, independent of
# RESCUE_ENABLED. Used to record actual ball trajectory for offline offset
# tuning when rescue is off.
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


def _log_shot_result(stats: dict, before, after) -> tuple[bool | None, float | None]:
    """Print the score diff line and update stats. Returns (changed, diff) or
    (None, None) if either snapshot was missing — caller can persist either way."""
    if before is None or after is None:
        return None, None
    changed, diff = score_changed(before, after)
    stats["attempts"] += 1
    if changed:
        stats["makes"] += 1
        print(f"  [score] MAKE (diff={diff:.1f}) | session {stats['makes']}/{stats['attempts']}")
    else:
        print(f"  [score] miss (diff={diff:.1f}) | session {stats['makes']}/{stats['attempts']}")
    return changed, diff


def _try_rescue(left: int, top: int, width: int, height: int,
                hoop_x: int, hoop_y: int, platform_x: int,
                monitor_dir: Path | None = None,
                landing_timeout: float = 1.5) -> bool:
    """After a launch click, track the ball; click on it when it's over the hoop,
    then continue tracking until the ball has visibly landed.

    "Landed" = motion-masked ball detection misses for 3 consecutive frames
    after at least one detection. Returns True if a rescue click was fired.

    landing_timeout caps total time spent waiting for landing after the
    rescue window expires (the ball can take longer than RESCUE_WINDOW for
    high arcs; this gives us a chance to confirm landing without spinning
    forever if the ball was never seen).
    """
    rescue_deadline = time.time() + RESCUE_WINDOW
    landing_hard_deadline = time.time() + RESCUE_WINDOW + landing_timeout
    sx0 = platform_x + 120
    sx1 = max(platform_x, hoop_x) + 40
    sy0 = 0
    sy1 = hoop_y + 5

    detected = 0
    iters = 0
    last_ball: tuple[int, int] | None = None
    closest_dx: int | None = None
    prev_frame = None
    consecutive_unseen = 0
    rescue_fired = False
    landed_at: float | None = None
    rescue_start = time.time()

    while time.time() < landing_hard_deadline:
        iters += 1
        frame = grab_region(left, top, width, height)
        ball = find_ball(frame, sx0, sy0, sx1, sy1, prev_frame=prev_frame)
        if monitor_dir is not None:
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(str(monitor_dir / f"flight_{iters:03d}.png"), bgr)
        prev_frame = frame

        if ball is not None:
            detected += 1
            last_ball = ball
            consecutive_unseen = 0
            bx, by = ball
            dx = abs(bx - hoop_x)
            if closest_dx is None or dx < closest_dx:
                closest_dx = dx
            # Only fire rescue while still inside the rescue window.
            if not rescue_fired and time.time() < rescue_deadline:
                if dx <= BALL_X_TOLERANCE and by < hoop_y:
                    print(f"  [rescue] ball at ({bx},{by}), over hoop_x={hoop_x} — dropping")
                    click(left + bx, top + by)
                    rescue_fired = True
        else:
            if detected > 0:
                consecutive_unseen += 1
                if consecutive_unseen >= 3 and landed_at is None:
                    landed_at = time.time() - rescue_start
                    # Ball has settled — exit early.
                    break
        time.sleep(RESCUE_POLL)

    if detected == 0:
        print(f"  [rescue] no ball detected in {iters} frames — HSV bounds may be off")
    elif not rescue_fired:
        print(f"  [rescue] ball seen {detected}/{iters} frames; closest to hoop_x={hoop_x} was {closest_dx}px (last at {last_ball})")
    if landed_at is not None:
        print(f"  [land] ball landed after {landed_at:.2f}s")
    elif detected > 0:
        print(f"  [land] ball detection didn't terminate within {RESCUE_WINDOW + landing_timeout:.1f}s; assuming landed")
    return rescue_fired


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


def run():
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        session_started = datetime.now().isoformat(timespec="seconds")
        shot_db = open_db(SHOT_DB_PATH)
        try:
            _run_inner(session_started, shot_db)
        finally:
            shot_db.close()


def _run_inner(session_started: str, shot_db):
    print(f"Hoops bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    platform_history: deque[int] = deque(maxlen=5)
    target_y: int | None = None
    hoop_x: int | None = None
    hoop_y: int | None = None
    hoop_conf_last: float = 0.0  # carries to shot_log row
    hoop_missing_since: float | None = None  # for exit-when-stuck
    x_samples: list[int] = []
    home_x: int | None = None
    range_samples: deque[tuple[int, int]] = deque(maxlen=200)  # (px, py) for range diagnostics
    last_range_log = time.time()
    prev_py: int | None = None
    shot_stats: dict = {"makes": 0, "attempts": 0}
    # Hoops only respawn on MAKES (per user observation). If we fire a clamped
    # shot (target unreachable) at a hoop position and the hoop is still there
    # next iteration, we'd waste another life on a guaranteed miss. Track the
    # last clamped position; if we'd clamp at the same place again, exit.
    last_unreachable_hoop: tuple[int, int] | None = None

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

        if target_y is None:
            hoop_pos, hoop_conf = find_rim(frame)
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
                if elapsed > 20:
                    print("Rim invisible for >20s — bailing out.")
                    return
                time.sleep(0.2)
                continue
            hoop_missing_since = None
            hoop_x, hoop_y = hoop_pos
            hoop_conf_last = hoop_conf
            offset = _compute_offset(hoop_y)
            target_y = hoop_y + offset
            print(f"Hoop rim at ({hoop_x},{hoop_y}) (conf={hoop_conf:.2f}), offset={offset}, target launch y={target_y}")

        platform_pos, platform_conf = find_platform(frame)
        if platform_pos is None:
            time.sleep(POLL_INTERVAL)
            continue

        px, py = platform_pos
        platform_history.append(py)
        range_samples.append((px, py))

        if time.time() - last_range_log > 3.0 and range_samples:
            xs = [p[0] for p in range_samples]
            ys = [p[1] for p in range_samples]
            print(f"  [diag] platform last 200 samples: x={min(xs)}..{max(xs)}, y={min(ys)}..{max(ys)}, target_y={target_y}")
            last_range_log = time.time()

        if home_x is None:
            x_samples.append(px)
            if len(x_samples) >= HOME_X_SAMPLES:
                spread = max(x_samples) - min(x_samples)
                home_x = sorted(x_samples)[len(x_samples) // 2]
                if spread > X_TOLERANCE * 3:
                    print(f"Platform X varied by {spread}px during sampling — bot likely started at score >=10. Anchor home_x={home_x} may be inaccurate.")
                else:
                    print(f"Locked platform home_x={home_x} (spread {spread}px over {HOME_X_SAMPLES} samples)")

        # Clamp target_y to platform's observed reachable range. If the hoop
        # repositions outside the platform's bob, we still want to fire — even
        # a miss forces the hoop to reposition, breaking the deadlock.
        effective_target_y = target_y
        clamped = False
        if len(range_samples) >= 40:
            ys = [p[1] for p in range_samples]
            ymin, ymax = min(ys), max(ys)
            if target_y > ymax:
                effective_target_y = ymax
                clamped = True
            elif target_y < ymin:
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
        if in_window or crossed:
            x_ok = home_x is None or abs(px - home_x) <= X_TOLERANCE
            # Direction from the actual crossing if available, else from history.
            if crossed and prev_py is not None:
                direction = "up" if py < prev_py else "down"
            else:
                direction = _direction(platform_history)
            # When clamped, the platform only crosses the target at a bob extreme,
            # where direction is whatever-it-just-was → never matches. Bypass.
            direction_ok = clamped or REQUIRED_DIRECTION == "any" or direction == REQUIRED_DIRECTION
            if x_ok and direction_ok:
                # If the hoop is outside the platform's bob range, every shot is a
                # guaranteed miss — and a miss costs a life (the prompt-dismiss
                # click is the first scored shot, not a freebie). Idle here and
                # let the user notice + recalibrate the offset.
                if clamped:
                    if last_unreachable_hoop != (hoop_x, hoop_y):
                        print(f"Hoop at {(hoop_x, hoop_y)} is outside platform's reach "
                              f"(target_y={target_y}). Idling — likely needs offset recalibration.")
                        last_unreachable_hoop = (hoop_x, hoop_y)
                    prev_py = py
                    time.sleep(POLL_INTERVAL)
                    continue
                tag = " [crossed]" if crossed and not in_window else ""
                print(f"Platform at ({px},{py}) (target_y={target_y}, dir={direction}) — shooting{tag}")
                # Per-shot monitor folder: we'll save pre/post-shot screenshots
                # plus all flight frames captured during _try_rescue.
                shot_idx = shot_stats["attempts"] + 1
                shot_dir = _make_monitor_dir(shot_idx) if MONITOR_MODE else None
                if shot_dir is not None:
                    save_frame(shot_dir / "pre_shot.png", frame)
                # Snapshot score and lives regions before launch so we can diff later.
                score_before = _capture_score_region(left, top, width, height)
                lives_before = _capture_lives_region(left, top, width, height)
                random_delay(10, 40)
                fired_at = datetime.now().isoformat(timespec="seconds")
                click(left + width // 2, top + height // 2)
                # Try to rescue an overshoot by clicking the ball mid-flight.
                if RESCUE_ENABLED:
                    _try_rescue(left, top, width, height, hoop_x, hoop_y, px, monitor_dir=shot_dir)
                    time.sleep(POST_SHOT_COOLDOWN)
                elif MONITOR_FLIGHT and shot_dir is not None:
                    flight_deadline = time.time() + POST_SHOT_COOLDOWN
                    flight_idx = 0
                    while time.time() < flight_deadline:
                        check_failsafe()
                        flight_idx += 1
                        f = grab_region(left, top, width, height)
                        bgr = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
                        cv2.imwrite(str(shot_dir / f"flight_{flight_idx:03d}.png"), bgr)
                        time.sleep(FLIGHT_POLL)
                else:
                    time.sleep(POST_SHOT_COOLDOWN)
                score_after = _capture_score_region(left, top, width, height)
                lives_after = _capture_lives_region(left, top, width, height)
                made, score_diff = _log_shot_result(shot_stats, score_before, score_after)
                log_shot(
                    shot_db,
                    session_started=session_started,
                    shot_idx=shot_idx,
                    fired_at=fired_at,
                    hoop_x=hoop_x,
                    hoop_y=hoop_y,
                    hoop_conf=float(hoop_conf_last),
                    platform_x=int(px),
                    platform_y=int(py),
                    offset=int(offset),
                    target_y=int(target_y),
                    eff_target_y=int(effective_target_y),
                    clamped=int(bool(clamped)),
                    direction=direction,
                    required_direction=REQUIRED_DIRECTION,
                    score_diff=float(score_diff) if score_diff is not None else None,
                    made=int(bool(made)) if made is not None else None,
                    shot_dir=str(shot_dir) if shot_dir is not None else None,
                )
                # Lives counter check: when the digit visibly changes between
                # pre and post, log it. Doesn't reliably indicate "click didn't
                # register" when unchanged (the lives counter doesn't always
                # tick per shot — game-side semantics vary, and our diff
                # threshold + region pick make the signal noisy).
                if lives_before is not None and lives_after is not None:
                    lives_visible = float(lives_before.std()) >= 5.0 or float(lives_after.std()) >= 5.0
                    if lives_visible:
                        lives_changed_flag, lives_diff = score_changed(lives_before, lives_after)
                        if lives_changed_flag:
                            print(f"  [lives] counter ticked down (diff={lives_diff:.1f})")
                if shot_dir is not None:
                    post_frame = grab_region(left, top, width, height)
                    save_frame(shot_dir / "post_shot.png", post_frame)
                    save_meta(
                        shot_dir / "meta.txt",
                        hoop=f"({hoop_x},{hoop_y})",
                        platform=f"({px},{py})",
                        offset=offset,
                        target_y=target_y,
                        eff_target_y=effective_target_y,
                        clamped=clamped,
                        direction=direction,
                    )
                platform_history.clear()
                prev_py = None
                target_y = None  # re-detect hoop after it repositions
                continue

        prev_py = py

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
