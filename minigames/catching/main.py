import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay, check_failsafe
from common.regions import get_region
from common.session_log import session_log
from common.window import get_bounds, WindowNotFoundError
from common.git_info import current_code_commit
from common.auto_commit import commit_file_if_changed
from minigames.catching.detector import find_fly, find_next_gap, find_play_button
from minigames.catching.catch_log import open_db, log_flap, log_run

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
CATCH_DB_PATH = _HERE / "assets" / "catching.db"
REPO_ROOT = _HERE.parent.parent

WINDOW_TITLE = "Legends Of Idleon"
POLL_INTERVAL = 0.02

# Click vertical strategy: if the fly's projected Y after the next click delay
# exceeds GAP_LOWER_MARGIN px past the gap's bottom, fire a click to gain
# altitude. Tune empirically once we see real flight physics.
GAP_LOWER_MARGIN = 8

# Cooldown between consecutive clicks (avoids click-spamming if our model
# overestimates how fast the fly is dropping).
MIN_CLICK_INTERVAL = 0.05

# Auto-start (#47): when no fly is on screen the bot is at the entry prompt
# between attempts, so it clicks the "PLAY GAME" button to start the next
# play and keeps going until the daily plays run out. START_WAIT_S gives
# the minigame time to load (the fly to appear) after the click.
START_WAIT_S = 2.5
# Stop after this many PLAY-GAME clicks in a row that don't produce a fly —
# the plays are exhausted (the button still renders but no game starts), or
# the bot is stuck. Resets whenever a play actually starts (a fly appears).
MAX_START_ATTEMPTS = 3

# Game-over bail (#47): with no fly AND no PLAY GAME button for this long,
# the player has left the catching screen (or the prompt vanished) — exit
# cleanly so the run summary is logged. Longer than a normal end-of-attempt
# transition so auto-start isn't cut off between plays.
NO_FLY_BAIL_S = 12.0

# Where to click in the play area (Flappy Bird usually accepts clicks
# anywhere in the play region). Center of the 'play' region by default.


def run():
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        session_started = datetime.now().isoformat(timespec="seconds")
        code_commit = current_code_commit(REPO_ROOT)
        if code_commit:
            print(f"Code commit: {code_commit}")
        if not (_HERE / "assets" / "fly.png").exists():
            print("WARNING: assets/fly.png not found — the fly detector can't "
                  "match, so zero flaps will be logged. Capture + extract the "
                  "template first (catching-capture during a game, then "
                  "catching-extract-fly).")
        db = open_db(CATCH_DB_PATH)
        started_at = time.time()
        # Mutable so the finally below sees the running count even though
        # _run_inner's loop only exits by exception (failsafe / Ctrl-C).
        stats = {"n_flaps": 0, "end_reason": "process_exit"}
        try:
            _run_inner(session_started, db, code_commit, stats)
        except KeyboardInterrupt:
            stats["end_reason"] = "keyboard_interrupt"
            raise
        except Exception as e:  # FailSafeException (corner-slam) and any crash
            stats["end_reason"] = type(e).__name__
            raise
        finally:
            log_run(
                db,
                session_started=session_started,
                attempt_idx=1,
                ended_at=datetime.now().isoformat(timespec="seconds"),
                n_flaps=stats["n_flaps"],
                duration_s=round(time.time() - started_at, 1),
                end_reason=stats["end_reason"],
                code_commit=code_commit,
            )
            db.close()
            # Tracked DB — other machines get the data via `git pull`.
            commit_file_if_changed(
                REPO_ROOT,
                "minigames/catching/assets/catching.db",
                "chore(catching): refresh catching.db (auto)",
            )


def _run_inner(session_started, db, code_commit, stats):
    print(f"Catching bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    last_click_time = 0.0
    prev_fly: tuple[int, float] | None = None  # (fly_y, wall_clock) of last detected frame, for velocity
    last_fly_time = 0.0       # wall-clock of the last fly detection (game-over bail)
    first_fly_seen = False    # don't bail before the minigame has started
    start_attempts = 0        # consecutive PLAY-GAME clicks with no game starting
    while True:
        check_failsafe()
        try:
            win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        play_region = get_region(_HERE, "play", win_w, win_h)
        if play_region is None:
            print("No 'play' region in regions.json. Run catching-pick-play-region first.")
            time.sleep(2)
            continue

        frame = grab_region(
            win_left + play_region["left"],
            win_top + play_region["top"],
            play_region["width"],
            play_region["height"],
        )
        fly_pos = find_fly(frame)
        if fly_pos is None:
            # No fly = not in an active game. Click PLAY GAME to start the
            # next play. The prompt is anchored to the player, so search the
            # FULL window (not the play-region crop) for the button.
            full = grab_region(win_left, win_top, win_w, win_h)
            btn = find_play_button(full)
            if btn is not None:
                if start_attempts >= MAX_START_ATTEMPTS:
                    stats["end_reason"] = "plays_exhausted"
                    print(f"Clicked PLAY GAME {start_attempts}x with no game "
                          f"starting — plays likely exhausted. "
                          f"Flaps total: {stats['n_flaps']}.")
                    return
                bx, by = btn
                start_attempts += 1
                print(f"PLAY GAME at ({bx},{by}) — starting play "
                      f"(attempt {start_attempts}/{MAX_START_ATTEMPTS})")
                click(win_left + bx, win_top + by)
                last_fly_time = time.time()  # don't let the bail fire mid-start
                time.sleep(START_WAIT_S)
                continue
            # No fly AND no button: a long transition, or off the screen.
            if first_fly_seen and time.time() - last_fly_time > NO_FLY_BAIL_S:
                stats["end_reason"] = "game_over"
                print(f"No fly or PLAY GAME button for {NO_FLY_BAIL_S:.0f}s — "
                      f"done. Flaps this run: {stats['n_flaps']}.")
                return
            time.sleep(POLL_INTERVAL)
            continue
        fly_x, fly_y = fly_pos
        first_fly_seen = True
        last_fly_time = time.time()
        start_attempts = 0    # a fly appeared → the last start succeeded

        # Fly vertical velocity at this frame, px/SECOND (+ = descending).
        # Wall-clock dt keeps it cadence-invariant (the darts vy lesson); a
        # stale prior frame from a detection dropout yields NULL rather than
        # a bogus slow reading. Updated every detected frame for continuity.
        now = time.time()
        fly_vy = None
        if prev_fly is not None:
            dt = now - prev_fly[1]
            if 0 < dt <= 0.2:
                fly_vy = int(round((fly_y - prev_fly[0]) / dt))
        prev_fly = (fly_y, now)

        gap = find_next_gap(frame, fly_pos)
        if gap is None:
            time.sleep(POLL_INTERVAL)
            continue
        gap_top, gap_bottom, gap_left_x, gap_right_x = gap

        # Click if the fly is dropping toward / past the gap's bottom edge.
        if fly_y > gap_bottom - GAP_LOWER_MARGIN:
            if now - last_click_time >= MIN_CLICK_INTERVAL:
                # Fire immediately on the decision. The fly is still falling,
                # so any pre-click latency (the old random_delay that sat
                # here) biases the sampled fly_y away from where the flap
                # lands. Stamp the fire time and click; bookkeeping runs
                # after — same rule as hoops/darts, see CLAUDE.md "Click
                # timing".
                clicked_at = datetime.now().isoformat(timespec="milliseconds")
                cx = play_region["left"] + play_region["width"] // 2
                cy = play_region["top"] + play_region["height"] // 2
                click(win_left + cx, win_top + cy)
                last_click_time = time.time()
                stats["n_flaps"] += 1
                print(f"fly y={fly_y} vy={fly_vy} gap=[{gap_top}..{gap_bottom}] — flap #{stats['n_flaps']}")
                # Keep this post-click work light: heavy work here delays the
                # next fly sample (the hoops cc42529 latency regression). One
                # indexed insert is fine; if it ever isn't, move it after
                # random_delay.
                log_flap(
                    db,
                    session_started=session_started,
                    attempt_idx=1,
                    flap_idx=stats["n_flaps"],
                    clicked_at=clicked_at,
                    fly_x=fly_x,
                    fly_y=fly_y,
                    fly_vy=fly_vy,
                    gap_top=gap_top,
                    gap_bottom=gap_bottom,
                    gap_left_x=gap_left_x,
                    gap_right_x=gap_right_x,
                    gap_center=(gap_top + gap_bottom) // 2,
                    gap_height=gap_bottom - gap_top,
                    fly_offset_below_gap=fly_y - gap_bottom,
                    gap_lower_margin=GAP_LOWER_MARGIN,
                    window_w=win_w,
                    window_h=win_h,
                    code_commit=code_commit,
                    source="bot",
                )
                random_delay(5, 20)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
