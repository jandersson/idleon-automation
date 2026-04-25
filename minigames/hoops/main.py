import time
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay
from common.window import get_bounds, WindowNotFoundError
from minigames.hoops.detector import find_hoop, find_platform, find_ball, score_region, score_changed

WINDOW_TITLE = "Idleon"
POLL_INTERVAL = 0.02

# Position-dependent vertical offset added to hoop rim Y to get the ideal
# platform-launch Y. Rolls rim-vs-platform geometry AND ball-travel lead into
# one number per hoop position. Linearly interpolated between anchor points;
# extrapolation is clamped to the nearest anchor's value.
#
# To tune: watch where shots miss for a given hoop_y and add/move an anchor:
#   - Ball clears top of backboard → offset is too small → bump it up
#   - Ball hits front of rim       → offset is too large → bump it down
#   - Ball hits back of rim        → offset is too small → bump it up
OFFSET_ANCHORS: list[tuple[int, int]] = [
    (700, 35),   # high hoops — initial guess, tune from observation
    (900, 8),    # mid-range — 14 still overshot; reducing further. Make zone
                 # likely between 0 (back-of-rim misses) and 18 (was 6/7 once).
]


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
REQUIRED_DIRECTION = "up"

# Wait after clicking so the ball can travel and the hoop can reposition.
POST_SHOT_COOLDOWN = 2.0

# At score >=10 the platform also moves horizontally. We sample the platform's X
# during the early stationary phase, lock in the median as our anchor, and from
# then on require px to be within X_TOLERANCE of the anchor before firing.
HOME_X_SAMPLES = 10
X_TOLERANCE = 9999  # effectively disabled — re-enable with small value (e.g. 4) for score 10+

# Mid-flight rescue: after the launch click, watch for the ball. When it crosses
# over the hoop's X (still above the rim), click on it — the wiki trick that
# makes the ball drop straight down. Saves shots that would otherwise overshoot.
RESCUE_WINDOW = 1.5  # seconds to track the ball after launch
BALL_X_TOLERANCE = 20  # how close ball X must be to hoop X to trigger drop
RESCUE_POLL = 0.01  # tight loop — ball moves fast

# Window-relative crop for the "Score: N" readout in the upper-left of the
# minigame UI. Tune via hoops-score-calibrate; defaults are an initial guess
# based on a 1280x1392 window. Disable score detection by setting to None.
SCORE_REGION_REL: dict | None = {"left": 6, "top": 384, "width": 78, "height": 18}


def _capture_score_region(left: int, top: int, width: int, height: int):
    if SCORE_REGION_REL is None:
        return None
    frame = grab_region(left, top, width, height)
    return score_region(
        frame,
        SCORE_REGION_REL["left"],
        SCORE_REGION_REL["top"],
        SCORE_REGION_REL["width"],
        SCORE_REGION_REL["height"],
    )


def _log_shot_result(stats: dict, before, after) -> None:
    if before is None or after is None:
        return
    changed, diff = score_changed(before, after)
    stats["attempts"] += 1
    if changed:
        stats["makes"] += 1
        print(f"  [score] MAKE (diff={diff:.1f}) | session {stats['makes']}/{stats['attempts']}")
    else:
        print(f"  [score] miss (diff={diff:.1f}) | session {stats['makes']}/{stats['attempts']}")


def _try_rescue(left: int, top: int, width: int, height: int,
                hoop_x: int, hoop_y: int, platform_x: int) -> bool:
    """After a launch click, track the ball; click on it when it's over the hoop.

    Returns True if a rescue click was fired. Logs a one-line diagnostic when
    it doesn't fire so we can see whether the ball is being detected at all and,
    if so, why it never crossed the drop window.
    """
    deadline = time.time() + RESCUE_WINDOW
    # Search airspace: between platform x and hoop x, above the hoop rim.
    sx0 = min(platform_x, hoop_x) - 20
    sx1 = max(platform_x, hoop_x) + 40
    sy0 = 0
    sy1 = hoop_y + 5  # don't let the rim itself confuse the mask

    detected = 0
    iters = 0
    last_ball: tuple[int, int] | None = None
    closest_dx: int | None = None
    while time.time() < deadline:
        iters += 1
        frame = grab_region(left, top, width, height)
        ball = find_ball(frame, sx0, sy0, sx1, sy1)
        if ball is not None:
            detected += 1
            last_ball = ball
            bx, by = ball
            dx = abs(bx - hoop_x)
            if closest_dx is None or dx < closest_dx:
                closest_dx = dx
            if dx <= BALL_X_TOLERANCE and by < hoop_y:
                print(f"  [rescue] ball at ({bx},{by}), over hoop_x={hoop_x} — dropping")
                click(left + bx, top + by)
                return True
        time.sleep(RESCUE_POLL)

    if detected == 0:
        print(f"  [rescue] no ball detected in {iters} frames — HSV bounds may be off")
    else:
        print(f"  [rescue] ball seen {detected}/{iters} frames; closest to hoop_x={hoop_x} was {closest_dx}px (last at {last_ball})")
    return False


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
    print(f"Hoops bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    platform_history: deque[int] = deque(maxlen=5)
    target_y: int | None = None
    hoop_x: int | None = None
    hoop_y: int | None = None
    x_samples: list[int] = []
    home_x: int | None = None
    range_samples: deque[tuple[int, int]] = deque(maxlen=200)  # (px, py) for range diagnostics
    last_range_log = time.time()
    prev_py: int | None = None
    shot_stats: dict = {"makes": 0, "attempts": 0}

    while True:
        try:
            left, top, width, height = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        frame = grab_region(left, top, width, height)

        if target_y is None:
            hoop_pos, hoop_conf = find_hoop(frame)
            if hoop_pos is None:
                print(f"Hoop not found (best confidence={hoop_conf:.2f})")
                time.sleep(0.2)
                continue
            hoop_x, hoop_y = hoop_pos
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
        if len(range_samples) >= 100:
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
        in_window = abs(py - effective_target_y) <= Y_TOLERANCE
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
                tag = " [clamped — likely miss]" if clamped else (" [crossed]" if crossed and not in_window else "")
                print(f"Platform at ({px},{py}) (target_y={target_y}, eff={effective_target_y}, dir={direction}) — shooting{tag}")
                # Snapshot score region before launch so we can diff later.
                score_before = _capture_score_region(left, top, width, height)
                random_delay(10, 40)
                click(left + width // 2, top + height // 2)
                # Try to rescue an overshoot by clicking the ball mid-flight.
                _try_rescue(left, top, width, height, hoop_x, hoop_y, px)
                time.sleep(POST_SHOT_COOLDOWN)
                score_after = _capture_score_region(left, top, width, height)
                _log_shot_result(shot_stats, score_before, score_after)
                platform_history.clear()
                prev_py = None
                target_y = None  # re-detect hoop after it repositions
                continue

        prev_py = py

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
