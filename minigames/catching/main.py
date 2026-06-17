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

# Flap policy. Aim the fly at the next hoop's CENTRE when one is visible,
# else hover at this fraction of the play-region height so it never
# free-falls between hoops (the old policy only flapped near a detected
# hoop's bottom and did nothing without one — the fly sank and died).
# From the observe frames the fly/hoop sit around mid-height.
DEFAULT_HOVER_FRAC = 0.5
# Flap once the fly has fallen this many px BELOW the target (screen y grows
# downward). A small deadband so it oscillates around the target.
FLAP_MARGIN = 6

# Min seconds between flaps — caps the flap rate. First live run (60 flaps)
# at 0.12 left the fly stuck low: fly_y median 145 vs a ~91 target (gravity
# beat 8/s flapping), so it survived but flew below the hoop band and didn't
# score. Lowered to lift it into the band. Still PROVISIONAL — retune from
# the DB (fly_y/vy per flap) if it now over-flaps high or still sinks.
MIN_CLICK_INTERVAL = 0.08

# Auto-start (#47): the bot clicks "PLAY GAME" to start ONE game, plays it,
# then STOPS when the fly dies — it does NOT auto-replay the next daily play
# (auto-replay-all is a possible future opt-in). After a click, poll for the
# fly for up to this long WITHOUT re-clicking or re-matching the button — the
# fly needs flapping the instant it appears, so we must NOT sleep the load.
START_CLICK_COOLDOWN_S = 2.0
# Give up starting after this many PLAY-GAME clicks that don't produce a fly
# (the click missed, or no plays left). Only governs the initial start.
MAX_START_ATTEMPTS = 3

# Game over: once a game has started, the fly being gone this long means the
# attempt ended (it died / the prompt is back), so exit cleanly. Long enough
# to ride out brief mid-play fly-detection dropouts, short enough to stop
# promptly without re-clicking PLAY GAME.
FLY_GONE_GAMEOVER_S = 2.0

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
    last_start_click = 0.0    # wall-clock of the last PLAY GAME click
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
            # A game has already started and the fly is gone -> the attempt is
            # over (it died / the prompt is back). STOP — do NOT re-click PLAY
            # GAME to start another play. The grace rides out brief mid-play
            # detection dropouts.
            if first_fly_seen:
                if time.time() - last_fly_time > FLY_GONE_GAMEOVER_S:
                    stats["end_reason"] = "game_over"
                    print(f"Fly gone {FLY_GONE_GAMEOVER_S:.0f}s — attempt over. "
                          f"Flaps: {stats['n_flaps']}.")
                    return
                time.sleep(POLL_INTERVAL)
                continue
            # No game started yet. If we just clicked PLAY GAME, poll fast for
            # the fly without re-clicking (don't sleep through the load).
            if time.time() - last_start_click < START_CLICK_COOLDOWN_S:
                time.sleep(POLL_INTERVAL)
                continue
            # At the entry prompt — click PLAY GAME to start the game. The
            # prompt is anchored to the player, so search the FULL window.
            full = grab_region(win_left, win_top, win_w, win_h)
            btn = find_play_button(full)
            if btn is None:
                time.sleep(POLL_INTERVAL)
                continue
            if start_attempts >= MAX_START_ATTEMPTS:
                stats["end_reason"] = "no_start"
                print(f"Clicked PLAY GAME {start_attempts}x with no game "
                      f"starting — giving up. Flaps: {stats['n_flaps']}.")
                return
            bx, by = btn
            start_attempts += 1
            last_start_click = time.time()
            print(f"PLAY GAME at ({bx},{by}) — starting game "
                  f"(attempt {start_attempts}/{MAX_START_ATTEMPTS})")
            click(win_left + bx, win_top + by)
            continue
        fly_x, fly_y = fly_pos
        first_fly_seen = True
        last_fly_time = time.time()

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
        # Aim at the next hoop's CENTRE when one is visible; otherwise hover
        # at a default height so the fly doesn't free-fall between hoops.
        if gap is not None:
            gap_top, gap_bottom, gap_left_x, gap_right_x = gap
            target_y = (gap_top + gap_bottom) // 2
        else:
            gap_top = gap_bottom = gap_left_x = gap_right_x = None
            target_y = int(play_region["height"] * DEFAULT_HOVER_FRAC)

        # Flap when the fly has fallen below the target (screen y grows
        # downward), rate-limited so it oscillates rather than ceiling-slams.
        if fly_y > target_y + FLAP_MARGIN and now - last_click_time >= MIN_CLICK_INTERVAL:
            # Fire immediately on the decision (the fly is still falling) —
            # the hoops/darts click-timing rule; bookkeeping runs after.
            clicked_at = datetime.now().isoformat(timespec="milliseconds")
            cx = play_region["left"] + play_region["width"] // 2
            cy = play_region["top"] + play_region["height"] // 2
            click(win_left + cx, win_top + cy)
            last_click_time = time.time()
            stats["n_flaps"] += 1
            where = f"gap=[{gap_top}..{gap_bottom}]" if gap is not None else "hover"
            print(f"fly y={fly_y} vy={fly_vy} target={target_y} {where} — flap #{stats['n_flaps']}")
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
                gap_center=target_y if gap is not None else None,
                gap_height=(gap_bottom - gap_top) if gap is not None else None,
                fly_offset_below_gap=(fly_y - gap_bottom) if gap is not None else None,
                gap_lower_margin=FLAP_MARGIN,
                window_w=win_w,
                window_h=win_h,
                code_commit=code_commit,
                source="bot",
            )
            random_delay(5, 20)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
