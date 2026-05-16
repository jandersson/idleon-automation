"""Mining minigame bot.

Loop: detect cart + next obstacle, click to jump when an approaching pit
is within the trigger-distance window. First-pass policy is jump-only —
no slam — the goal of this version is just "don't fall into the first
pit." Tune JUMP_TRIGGER_MIN/MAX and JUMP_COOLDOWN_S empirically.

Run flow:
    1. User opens the mining minigame so the "Play Game" prompt is visible
    2. uv run mining
    3. Bot clicks Play Game (requires start_button region — pick via
       mining-pick-start-button beforehand)
    4. Bot enters the detection loop. When find_next_terrain reports a pit
       with distance in [JUMP_TRIGGER_MIN, JUMP_TRIGGER_MAX], it clicks
       once at the cart's position to jump.
    5. Loop continues until the minigame ends (plank goes off-screen) or
       the user slam-corners the mouse.
"""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import check_failsafe, click as bot_click
from common.regions import get_region
from common.session_log import session_log
from common.window import get_bounds, WindowNotFoundError
from minigames.mining.detector import find_cart, find_next_terrain

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"

WINDOW_TITLE = "Legends Of Idleon"
POLL_INTERVAL = 0.03

# Click policy. Trigger when a pit is detected at distance in this range.
# Picked from the trace data: scroll speed is ~80-93 px/s leftward, so
# a 40-90 px window is roughly 0.5-1.0s of warning — enough time for the
# click + jump-physics to clear the pit.
JUMP_TRIGGER_MIN = 40
JUMP_TRIGGER_MAX = 90

# After a jump click, ignore further triggers for this long so we don't
# spam clicks while the same pit is still in the trigger window.
JUMP_COOLDOWN_S = 0.6

# How long the bot keeps running after losing sight of the plank before
# giving up — covers brief transitions in/out of the minigame UI.
PLANK_LOST_TIMEOUT_S = 8.0


def run():
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        _run_inner()


def _run_inner():
    print(f"Mining bot starting — tracking window {WINDOW_TITLE!r}.")
    print("Move mouse to any screen corner to abort.")
    time.sleep(2)

    try:
        win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
    except WindowNotFoundError as e:
        print(e)
        return

    # Click Play Game button to start the attempt.
    if not _click_start_button(win_left, win_top, win_w, win_h):
        return

    last_click_time = 0.0
    last_plank_seen = time.time()
    jumps = 0

    while True:
        check_failsafe()
        try:
            win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError:
            time.sleep(0.5)
            continue

        frame_bgra = grab_region(win_left, win_top, win_w, win_h)
        frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        cart = find_cart(frame)
        if cart is None:
            # Cart not visible — could be airborne in a pose we haven't
            # templated, or the minigame ended. Either way, skip this tick.
            if time.time() - last_plank_seen > PLANK_LOST_TIMEOUT_S:
                print(f"Plank lost for >{PLANK_LOST_TIMEOUT_S}s — exiting. Jumps: {jumps}")
                return
            time.sleep(POLL_INTERVAL)
            continue
        last_plank_seen = time.time()

        terrain = find_next_terrain(frame, cart)
        now = time.time()
        if (terrain is not None
                and terrain["kind"] == "pit"
                and JUMP_TRIGGER_MIN <= terrain["distance_px"] <= JUMP_TRIGGER_MAX
                and now - last_click_time >= JUMP_COOLDOWN_S):
            screen_x = win_left + cart[0]
            screen_y = win_top + cart[1]
            print(f"JUMP t={now:.2f}  pit_dist={terrain['distance_px']}  "
                  f"cart=({cart[0]},{cart[1]})")
            bot_click(screen_x, screen_y)
            last_click_time = now
            jumps += 1

        time.sleep(POLL_INTERVAL)


def _click_start_button(win_left, win_top, win_w, win_h) -> bool:
    start_btn = get_region(_HERE, "start_button", win_w, win_h)
    if start_btn is None:
        print("No 'start_button' region in regions.json. "
              "Run mining-pick-start-button first.")
        return False
    cx = win_left + start_btn["left"] + start_btn["width"] // 2
    cy = win_top + start_btn["top"] + start_btn["height"] // 2
    print(f"Clicking Play Game at screen ({cx}, {cy})")
    bot_click(cx, cy)
    # Brief pause so the minigame UI is fully up before we start detecting.
    time.sleep(0.5)
    return True


if __name__ == "__main__":
    run()
