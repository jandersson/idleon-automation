import time
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay, check_failsafe
from common.regions import get_region
from common.session_log import session_log
from common.shot_log import current_code_commit
from common.window import get_bounds, WindowNotFoundError
from minigames.chopping.chop_log import open_db, log_chop, set_outcome
from minigames.chopping.detector import analyze_bar, nearest_red_distance, find_game_over

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
CHOPPING_DB = _HERE / "assets" / "chopping.db"

WINDOW_TITLE = "Legends Of Idleon"

# Regions are loaded from assets/regions.json each iteration so they survive
# window resizes. Pick via chopping-pick-bar-region / chopping-pick-button-region.
# - bar: the colored zone strip with the sliding pointer
# - button: the CHOP button (bot clicks its center)

POLL_INTERVAL = 0.01
COOLDOWN_AFTER_CLICK = 0.45

# If the same pointer x is reported in the same zone this many clicks in a row,
# assume the minigame is over (a stationary post-game UI element looks like a
# leaf to the detector) and exit cleanly instead of spamming clicks.
STAGNATION_LIMIT = 5

# How often to template-match for the game-over screen, in seconds. The poll
# loop runs ~100 Hz so a per-iteration grab+match would dominate CPU; once
# per second is plenty since the game-over banner stays put for many seconds.
GAME_OVER_CHECK_INTERVAL = 1.0

# Skip the click if the leaf's left edge is within this many pixels of a red
# column. Click latency (~50ms pre-click delay + OS jitter) lets the leaf drift
# a few px between detection and click — landing in red ends the minigame.
RED_SAFETY_MARGIN_PX = 8


def run():
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        _run_inner()


def _run_inner():
    print(f"Chopping bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    conn = open_db(CHOPPING_DB)
    session_started = datetime.now().isoformat(timespec="seconds")
    code_commit = current_code_commit(_HERE.parent.parent)
    chop_idx = 0
    # The last logged chop sits with outcome=NULL until the next iteration
    # tells us whether the bot survived (another successful chop) or the
    # run ended (game_over detected). pending = (row_id, click_time).
    pending: tuple[int, float] | None = None
    print(f"Chopping DB: {CHOPPING_DB} (session={session_started})")

    last_click: tuple[int, str] | None = None
    stagnation_count = 0
    last_game_over_check = 0.0

    while True:
        check_failsafe()
        try:
            win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        if time.time() - last_game_over_check > GAME_OVER_CHECK_INTERVAL:
            last_game_over_check = time.time()
            full_frame = grab_region(win_left, win_top, win_w, win_h)
            is_over, conf = find_game_over(full_frame)
            if is_over:
                if pending is not None:
                    row_id, click_time = pending
                    set_outcome(conn, row_id, "game_over",
                                int((time.time() - click_time) * 1000))
                    pending = None
                print(f"Game over detected (conf={conf:.2f}). Stopping.")
                conn.close()
                return

        bar_region = get_region(_HERE, "bar", win_w, win_h)
        leaf_region = get_region(_HERE, "leaf", win_w, win_h)
        button_region = get_region(_HERE, "button", win_w, win_h)
        if bar_region is None or button_region is None:
            print("Missing region(s) in regions.json. Run chopping-pick-bar-region and chopping-pick-button-region first.")
            time.sleep(2)
            continue

        bar_frame = grab_region(
            win_left + bar_region["left"],
            win_top + bar_region["top"],
            bar_region["width"],
            bar_region["height"],
        )
        # Leaf region is optional: if not picked, fall back to detecting the
        # leaf in the bar region itself (works when they visually overlap).
        leaf_frame = None
        if leaf_region is not None:
            leaf_frame = grab_region(
                win_left + leaf_region["left"],
                win_top + leaf_region["top"],
                leaf_region["width"],
                leaf_region["height"],
            )
        pointer_x, zone = analyze_bar(bar_frame, leaf_frame=leaf_frame)

        if pointer_x is not None and zone in ("green", "gold"):
            red_dist = nearest_red_distance(bar_frame, pointer_x)
            if red_dist is not None and red_dist < RED_SAFETY_MARGIN_PX:
                print(f"Pointer at x={pointer_x} in {zone} but only {red_dist}px from red — skipping (unsafe)")
                time.sleep(POLL_INTERVAL)
                continue
            if last_click == (pointer_x, zone):
                stagnation_count += 1
                if stagnation_count >= STAGNATION_LIMIT:
                    print(f"Pointer stuck at x={pointer_x} in {zone} for {stagnation_count} clicks — minigame likely over, stopping.")
                    conn.close()
                    return
            else:
                stagnation_count = 0
            print(f"Pointer at x={pointer_x} in {zone} zone — chopping")
            random_delay(20, 60)
            button_cx = button_region["left"] + button_region["width"] // 2
            button_cy = button_region["top"] + button_region["height"] // 2
            click_x = win_left + button_cx
            click_y = win_top + button_cy
            click(click_x, click_y)
            click_time = time.time()
            last_click = (pointer_x, zone)

            # The previous chop survived — we made it here and clicked again.
            if pending is not None:
                prev_row_id, prev_click_time = pending
                set_outcome(conn, prev_row_id, "survived",
                            int((click_time - prev_click_time) * 1000))

            chop_idx += 1
            row_id = log_chop(
                conn,
                session_started=session_started,
                chop_idx=chop_idx,
                clicked_at=datetime.now().isoformat(timespec="milliseconds"),
                pointer_x=pointer_x,
                zone=zone,
                nearest_red_distance=red_dist,
                red_safety_margin=RED_SAFETY_MARGIN_PX,
                bar_left=bar_region["left"],
                bar_top=bar_region["top"],
                bar_width=bar_region["width"],
                bar_height=bar_region["height"],
                button_click_x=button_cx,
                button_click_y=button_cy,
                window_w=win_w,
                window_h=win_h,
                code_commit=code_commit,
                source="bot",
            )
            pending = (row_id, click_time)
            time.sleep(COOLDOWN_AFTER_CLICK)
            continue

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
