import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay, check_failsafe
from common.regions import get_region
from common.score_diff import score_region
from common.session_log import session_log
from common.window import get_bounds, WindowNotFoundError
from common.git_info import current_code_commit
from common.auto_commit import commit_file_if_changed
from minigames.catching.detector import find_fly, find_next_gap, find_play_button
from minigames.catching.catch_log import open_db, log_flap, log_run
from minigames.catching.score import make_pts_reader

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
CATCH_DB_PATH = _HERE / "assets" / "catching.db"
DIGIT_TEMPLATES_DIR = _HERE / "assets" / "digit_templates"
REPO_ROOT = _HERE.parent.parent

WINDOW_TITLE = "Legends Of Idleon"
POLL_INTERVAL = 0.02

# Read the "N PTS" score this often (seconds) — NOT every poll. OCR (a small
# grab + template match) is cheap but the flap loop must stay fast, and the
# score only ticks on a hoop pass-through, so ~2 Hz captures every change.
# Sampled AFTER the flap block so it never sits between a flap decision and
# the click (the CLAUDE.md click-timing rule).
SCORE_SAMPLE_INTERVAL = 0.5
# Built once: a read_pts(score_crop) -> int | None bound to the digit
# template library. Returns None for uncaptured digits, so max-tracking
# never records a false-high (grow the library with catching-capture-digits).
_read_pts = make_pts_reader(DIGIT_TEMPLATES_DIR)

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

# Start confirmation: catching is entered from busy world maps whose dark
# enemy/scenery sprites land in the sky-band play crop and false-trigger
# find_fly as a STATIC blob at a fixed y (observed in a desert region:
# 42 flaps all at fly_y=104, vy=0, PLAY GAME never clicked). A real fly
# bobs/free-falls, so only treat a game as started once the detected fly's y
# has spanned at least this many px over the last few detections — a static
# false-positive never does. Until then the PLAY GAME button, not the fly,
# drives the loop.
START_MOTION_RANGE_PX = 10
START_MOTION_FRAMES = 8

# Where to click in the play area (Flappy Bird usually accepts clicks
# anywhere in the play region). Center of the 'play' region by default.


def fly_started_moving(recent_ys, min_range=START_MOTION_RANGE_PX) -> bool:
    """True once the detected fly's y has spanned at least `min_range` px over
    the buffered detections — i.e. a real bobbing/free-falling fly, not a
    static scenery blob mistaken for the fly at the entry prompt. The gate
    that stops a false fly from being read as a started game (see the desert-
    region failure in START_MOTION_RANGE_PX's note)."""
    return len(recent_ys) >= 2 and max(recent_ys) - min(recent_ys) >= min_range


def _read_score(win_left, win_top, win_w, win_h):
    """Grab the score region and OCR the "N PTS" number, or None.

    Separate small grab (the score HUD sits just below the play-region crop),
    so it doesn't disturb the flap loop's frame. Returns None when the region
    isn't picked, nothing's readable, or the digit isn't in the template
    library yet."""
    region = get_region(_HERE, "score", win_w, win_h)
    if region is None:
        return None
    frame = grab_region(
        win_left + region["left"],
        win_top + region["top"],
        region["width"],
        region["height"],
    )
    crop = score_region(frame, 0, 0, region["width"], region["height"])
    return _read_pts(crop)


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
        # final_score is the max PTS read during the run (None if unread).
        stats = {"n_flaps": 0, "end_reason": "process_exit", "final_score": None}
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
                final_score=stats["final_score"],
                min_click_interval=MIN_CLICK_INTERVAL,
                flap_margin=FLAP_MARGIN,
                hover_frac=DEFAULT_HOVER_FRAC,
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
    game_running = False      # confirmed a REAL game (prompt gone + fly moving)
    start_attempts = 0        # consecutive PLAY-GAME clicks with no game starting
    last_start_click = 0.0    # wall-clock of the last PLAY GAME click
    last_score_sample = 0.0   # wall-clock of the last throttled score read
    last_score = None         # most recent PTS read, logged per flap
    recent_fly_y: deque[int] = deque(maxlen=START_MOTION_FRAMES)  # for the start-motion gate
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

        if not game_running:
            # --- Startup: the PLAY GAME prompt is authoritative ----------
            # Don't trust find_fly here. On busy world maps a dark enemy /
            # scenery sprite in the sky-band crop false-triggers it as a
            # static blob, which used to make the bot "play" a phantom and
            # never click PLAY GAME. Drive off the button instead, and only
            # hand control to the fly once it's confirmed to be MOVING.
            full = grab_region(win_left, win_top, win_w, win_h)
            btn = find_play_button(full)
            if btn is not None:
                # At the entry prompt. If we just clicked, wait it out rather
                # than spamming plays; else click PLAY GAME (attempt-capped so
                # we give up if no game ever starts). Prompt is player-
                # anchored, so the match searched the FULL window.
                if time.time() - last_start_click < START_CLICK_COOLDOWN_S:
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
                recent_fly_y.clear()
                print(f"PLAY GAME at ({bx},{by}) — starting game "
                      f"(attempt {start_attempts}/{MAX_START_ATTEMPTS})")
                click(win_left + bx, win_top + by)
                continue
            # No prompt on screen: the minigame is loading or has started.
            # Confirm only once a fly is present AND its y has moved enough —
            # a real fly bobs/free-falls; a leftover static blob (or a one-
            # frame button-match dropout at the prompt) does not.
            if fly_pos is not None:
                recent_fly_y.append(fly_pos[1])
            if fly_pos is None or not fly_started_moving(recent_fly_y):
                time.sleep(POLL_INTERVAL)
                continue
            game_running = True
            last_fly_time = time.time()
            print(f"Game started — fly moving at {fly_pos}. Playing.")
            # fall through to flap this frame

        # --- In a confirmed game --------------------------------------
        if fly_pos is None:
            # The fly is gone -> the attempt is over (it died / the prompt is
            # back). STOP — do NOT re-click PLAY GAME. The grace rides out
            # brief mid-play detection dropouts.
            if time.time() - last_fly_time > FLY_GONE_GAMEOVER_S:
                stats["end_reason"] = "game_over"
                print(f"Fly gone {FLY_GONE_GAMEOVER_S:.0f}s — attempt over. "
                      f"Flaps: {stats['n_flaps']}.")
                return
            time.sleep(POLL_INTERVAL)
            continue
        fly_x, fly_y = fly_pos
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
                score=last_score,
                code_commit=code_commit,
                source="bot",
            )
            random_delay(5, 20)

        # Throttled score sample — AFTER any flap above, so OCR never sits
        # between a flap decision and its click. Tracks the running max as
        # the run's final_score (the score only climbs, so max == score at
        # death; max guards against a misread-low on the final frame).
        if now - last_score_sample >= SCORE_SAMPLE_INTERVAL:
            last_score_sample = now
            pts = _read_score(win_left, win_top, win_w, win_h)
            if pts is not None:
                last_score = pts
                if stats["final_score"] is None or pts > stats["final_score"]:
                    stats["final_score"] = pts

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
