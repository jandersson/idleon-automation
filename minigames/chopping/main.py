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
from minigames.chopping.chop_log import open_db, log_chop, log_poll, set_outcome
from minigames.chopping.detector import (
    analyze_bar,
    bar_pixel_count,
    nearest_red_distance,
)

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
CHOPPING_DB = _HERE / "assets" / "chopping.db"

WINDOW_TITLE = "Legends Of Idleon"

# Regions are loaded from assets/regions.json each iteration so they survive
# window resizes. Pick via chopping-pick-bar-region / chopping-pick-button-region.

POLL_INTERVAL = 0.01

# Time to wait after a click before sampling again. The leaf needs a frame
# or two to register the chop visually, but 0.45 was way over — limited the
# bot to 6 chops in ~9.6s (session 1 on 2026-05-24). 0.15 lets the leaf
# scroll a healthy distance before we re-sample without re-clicking the
# same window.
COOLDOWN_AFTER_CLICK = 0.15

# If the same pointer x is reported in the same zone this many clicks in a row,
# assume the minigame is over (a stationary post-game UI element looks like a
# leaf to the detector) and exit cleanly instead of spamming clicks.
STAGNATION_LIMIT = 5

# Round-end detector. While the round runs, the bar has hundreds of colored
# pixels (green + red + occasional gold). When the round ends the bar UI
# disappears entirely and the count collapses toward zero. If we see fewer
# than BAR_DEAD_PIXEL_THRESHOLD colored pixels for BAR_DEAD_GRACE_S
# consecutive seconds, the round is over.
#
# Replaces the previous game_over.png template match (the "JUST IDLE" world-
# map button) which fired inconsistently — caught session 1 but missed
# session 2 on 2026-05-24.
BAR_DEAD_PIXEL_THRESHOLD = 30
BAR_DEAD_GRACE_S = 1.5

# Skip the click if the leaf's left edge is within this many pixels of a red
# column. Click latency (~50ms pre-click delay + OS jitter) lets the leaf drift
# a few px between detection and click — landing in red ends the minigame.
RED_SAFETY_MARGIN_PX = 8

# Per-poll DB write rate cap. ~100Hz polling × many fields would flood the
# DB; 10Hz is plenty to see what happens between chops.
POLL_LOG_INTERVAL = 0.1


def run():
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        _run_inner()


def _run_inner():
    print(f"Chopping bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    conn = open_db(CHOPPING_DB)
    session_started = datetime.now().isoformat(timespec="seconds")
    session_start_t = time.time()
    code_commit = current_code_commit(_HERE.parent.parent)
    chop_idx = 0
    # The last logged chop sits with outcome=NULL until the next iteration
    # tells us whether the bot survived (another successful chop) or the
    # round ended (bar disappeared). pending = (row_id, click_time).
    pending: tuple[int, float] | None = None
    print(f"Chopping DB: {CHOPPING_DB} (session={session_started})")

    last_click: tuple[int, str] | None = None
    stagnation_count = 0
    last_poll_log = 0.0
    bar_dead_since: float | None = None  # wall-clock t when bar first looked dead

    while True:
        check_failsafe()
        try:
            win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

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
        leaf_frame = None
        if leaf_region is not None:
            leaf_frame = grab_region(
                win_left + leaf_region["left"],
                win_top + leaf_region["top"],
                leaf_region["width"],
                leaf_region["height"],
            )
        pointer_x, zone = analyze_bar(bar_frame, leaf_frame=leaf_frame)
        bar_px = bar_pixel_count(bar_frame)

        # Round-end check via bar disappearance — replaces game_over template.
        now = time.time()
        if bar_px < BAR_DEAD_PIXEL_THRESHOLD:
            if bar_dead_since is None:
                bar_dead_since = now
            elif now - bar_dead_since >= BAR_DEAD_GRACE_S:
                if pending is not None:
                    row_id, click_time = pending
                    set_outcome(conn, row_id, "round_ended",
                                int((now - click_time) * 1000))
                    pending = None
                print(f"Bar gone for {now - bar_dead_since:.1f}s (bar_px={bar_px}) — round ended, stopping.")
                conn.commit()
                conn.close()
                return
        else:
            bar_dead_since = None

        red_dist = None
        if pointer_x is not None:
            red_dist = nearest_red_distance(bar_frame, pointer_x)

        if pointer_x is not None and zone in ("green", "gold"):
            if red_dist is not None and red_dist < RED_SAFETY_MARGIN_PX:
                if now - last_poll_log >= POLL_LOG_INTERVAL:
                    log_poll(conn, session_started,
                             int((now - session_start_t) * 1000),
                             pointer_x, zone, red_dist, bar_px, 0)
                    last_poll_log = now
                time.sleep(POLL_INTERVAL)
                continue
            if last_click == (pointer_x, zone):
                stagnation_count += 1
                if stagnation_count >= STAGNATION_LIMIT:
                    print(f"Pointer stuck at x={pointer_x} in {zone} for {stagnation_count} clicks — minigame likely over, stopping.")
                    conn.commit()
                    conn.close()
                    return
            else:
                stagnation_count = 0

            # CLAUDE.md fire-first rule: click immediately after deciding —
            # no random_delay / disk writes between sample and click, or
            # the leaf moves before we click.
            button_cx = button_region["left"] + button_region["width"] // 2
            button_cy = button_region["top"] + button_region["height"] // 2
            click(win_left + button_cx, win_top + button_cy)
            click_time = time.time()
            last_click = (pointer_x, zone)
            print(f"Pointer at x={pointer_x} in {zone} (red_d={red_dist}) — chop #{chop_idx + 1}")

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
            log_poll(conn, session_started,
                     int((click_time - session_start_t) * 1000),
                     pointer_x, zone, red_dist, bar_px, 1)
            last_poll_log = click_time
            pending = (row_id, click_time)

            # Random delay goes AFTER the click (no fire-time latency);
            # cooldown lets the bar reset visually before we re-sample.
            random_delay(20, 60)
            time.sleep(COOLDOWN_AFTER_CLICK)
            continue

        if now - last_poll_log >= POLL_LOG_INTERVAL:
            log_poll(conn, session_started,
                     int((now - session_start_t) * 1000),
                     pointer_x, zone, red_dist, bar_px, 0)
            last_poll_log = now

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
