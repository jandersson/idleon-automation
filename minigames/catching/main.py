import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay, check_failsafe
from common.monitor import save_frame
from common.regions import get_region
from common.score_diff import score_region
from common.session_log import session_log
from common.window import get_bounds, WindowNotFoundError
from common.git_info import current_code_commit
from common.auto_commit import commit_file_if_changed
from minigames.catching.detector import find_avatar, find_next_gap, find_play_button
from minigames.catching.catch_log import open_db, log_flap, log_run
from minigames.catching.score import make_pts_reader
from minigames.catching.controller import (
    load_dynamics, should_flap_now, hover_target_y, floor_rescue_due,
)

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
CATCH_DB_PATH = _HERE / "assets" / "catching.db"
DIGIT_TEMPLATES_DIR = _HERE / "assets" / "digit_templates"
# Dense per-poll trajectory traces (--trace / CATCHING_TRACE). Gitignored —
# they're regenerable bulk like captures/. Each run writes one CSV; the fit in
# controller.py reads them to estimate the avatar's vertical dynamics (#60).
TRACES_DIR = _HERE / "assets" / "traces"
# Fitted vertical dynamics for the predictive flap timer (controller.py, #60).
# Committed (it's the model, not regenerable bulk); written by fitting a
# --trace CSV. When present and --model/CATCHING_USE_MODEL is set, the timed
# flap is decided by the model instead of the hand-tuned constants below.
DYNAMICS_PATH = _HERE / "assets" / "dynamics.json"
TRACE_COLUMNS = [
    "t", "fly_x", "fly_y", "fly_vy",
    "gap_top", "gap_bottom", "gap_left_x", "gap_right_x", "gap_center",
    "target_y", "fired", "where",
]
REPO_ROOT = _HERE.parent.parent

WINDOW_TITLE = "Legends Of Idleon"
POLL_INTERVAL = 0.02

# Read the "N PTS" score this often (seconds) — NOT every poll. OCR (a small
# grab + template match) is cheap but the flap loop must stay fast, and the
# score only ticks on a hoop pass-through, so ~2 Hz captures every change.
# Sampled AFTER the flap block so it never sits between a flap decision and
# the click (the CLAUDE.md click-timing rule).
SCORE_SAMPLE_INTERVAL = 0.5

# --save-frames debug capture cadence (seconds). Throttled and placed after
# the flap so it never delays a click. Writes full-window + play-region
# snapshots to a gitignored dir for offline detector diagnosis.
FRAME_SAVE_INTERVAL = 0.3
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
# Trigger the flap this many px BELOW the hoop centre. The avatar's bob is a
# fixed ~47px (the game's flap impulse) and the ring opening is ~53px, so the
# bob CAN fit inside — but it was riding ~7px high and clipping the TOP edge
# (measured bob 50-97 vs opening 54-107). Triggering below centre shifts the
# whole bob DOWN so it centres on the opening; the avatar then stays inside
# the ring through the pass and threads at ANY arrival phase, which is what
# makes threading reliable without true arrival-timing. ~half the bob minus
# the rate-limit sag. Visual-tune: clips top -> raise; clips bottom -> lower.
FLAP_CENTER_DROP_PX = 12

# Phase-timing (#60). The avatar's bob is ~55px peak-to-peak (measured from
# the saved frames) — bigger than the ring's passable hole — so it can't hover
# THROUGH a ring; it has to COAST through, timed so its slow descent lines up
# with the crossing. When a hoop's leading edge gets within LEAD_DIST_PX of the
# avatar AND the avatar is low in its bob (so a flap lifts it), fire ONE timed
# flap; then suppress hover flaps for COAST_S while it descends through the
# hole, flapping only to stay off the bottom ring edge. Provisional — these
# set the bob phase vs the crossing, so they're the knobs to visual-tune.
LEAD_DIST_PX = 45
COAST_S = 0.5
COAST_RESCUE_PX = 6
# Only fire the timed flap when the avatar is at least this far BELOW the hoop
# centre (low in its bob). Firing at the centre over-lifts the avatar above
# the hole (run 12 crashed firing at fly_y == centre); the timed flaps that
# threaded fired from ~+6..+17 below centre. So the lift launches a consistent
# slow descent through the hole.
TIMED_LOW_OFFSET_PX = 6

# Min seconds between flaps — caps the flap rate, which sets the avatar's max
# reachable height (it flaps to rise, gravity pulls between flaps). History:
# 0.12 left it stuck low; 0.08 lifted it but the first ring-threading run
# (#60) showed it PEAKING at y~92 while ring centres sit at y~80 — i.e. at
# 0.08 it can't climb high enough and clips the ring's lower edge. Lowered to
# 0.05 so it can reach the ring centres. Still PROVISIONAL — visual-tune
# (threads vs slams the top); a per-shot fly_y/ring-centre fit needs the
# score-on-banner read (#60) for an objective signal.
MIN_CLICK_INTERVAL = 0.05

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

# Game over (motionless sprite): a live avatar bobs/free-falls continuously,
# so if the tracked sprite holds |vy| below this for STATIC_GAMEOVER_S it's a
# static blob left after the attempt ended (the minigame closed and find_fly
# latched onto scenery) — end the run instead of flapping a phantom. Observed
# desert failure: 94 flaps frozen at fly_y=104 vy=0, no game-over, run only
# stopped by the corner-slam.
FREEZE_VY_THRESHOLD = 8     # px/s; the blob sits at exactly 0, a live avatar well above
STATIC_GAMEOVER_S = 2.0

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

# Startup free-fall arrest (run 15, 2026-06-17): the avatar spawns and
# free-falls ~50px before fly_started_moving confirms the game (it needs
# motion = the avatar falling), landing ~5px from the floor — a coin-flip
# whether the first in-game flap catches it (run 14 recovered from y=140; run
# 15 hit ~145 and died). Once PLAY GAME has been clicked, flap a detected fly
# that's fallen below this fraction of the play height to keep it up DURING
# confirmation. The flaps also add the y-motion the confirmation needs, so the
# game confirms sooner. Gated on start_attempts>0 so a world-map scenery blob
# (no prompt ever clicked) can't trigger phantom flapping.
STARTUP_FLAP_FRAC = 0.5

# --- Region anchoring to the PLAY GAME prompt (issue #60) -----------------
# The catching minigame is a player-anchored overlay BAND (banner + avatar +
# hoops) drawn over the local biome, not a fixed-position screen. The band
# sits at a fixed offset from the PLAY GAME prompt (both are drawn above the
# player), so deriving the play crop from the detected prompt centre tracks
# the band across world regions — a fixed top-of-window rectangle misses it
# when the player stands lower. Offsets measured from the desert run (prompt
# at (670,226), avatar at (666,170)). The avatar is at the prompt's x, so
# it's searched in a NARROW central strip: a wide dark-blob search latches
# onto desert scenery instead of the (differently-coloured) avatar.
ANCHOR_PLAY_HALF_W = 260       # play crop half-width about the prompt x
ANCHOR_PLAY_TOP_DY = -135      # play crop top, relative to the prompt y
ANCHOR_PLAY_HEIGHT = 185
ANCHOR_AVATAR_DX = -4          # avatar x offset from the prompt x
ANCHOR_AVATAR_HALF_W = 35      # avatar search half-width
ANCHOR_AVATAR_BAND_H = 150     # avatar search height from the play-crop top
                               # (tall enough that a falling avatar isn't lost
                               # before the bot flaps it; excludes the banner)
# Score readout ("N PTS") position relative to the prompt — the banner is
# player-anchored too, so the score crop follows it across regions. Offset
# measured from the desert run (prompt (670,226), "1" digit at ~(610,268)).
ANCHOR_SCORE_DX = -62          # score crop left, relative to the prompt x
ANCHOR_SCORE_DY = 40           # score crop top, relative to the prompt y
ANCHOR_SCORE_W = 50            # wide enough for ~3 digits before "PTS"
ANCHOR_SCORE_H = 16

# Where to click in the play area (Flappy Bird usually accepts clicks
# anywhere in the play region). Center of the 'play' region by default.


def _anchored_play_region(bx, by, win_w, win_h):
    """Play crop derived from the PLAY GAME prompt centre (bx, by), clamped to
    the window. The minigame band is player-anchored at a fixed offset from
    the prompt, so this follows the band across world regions (issue #60)."""
    left = max(0, bx - ANCHOR_PLAY_HALF_W)
    top = max(0, by + ANCHOR_PLAY_TOP_DY)
    right = min(win_w, bx + ANCHOR_PLAY_HALF_W)
    bottom = min(win_h, top + ANCHOR_PLAY_HEIGHT)
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


def _find_avatar(frame, play_region, anchor):
    """Locate the avatar (fly / horseshoe / whatever the biome uses) in the
    play frame. With a prompt anchor, search only a NARROW central x-strip at
    the player's fixed x (= prompt x) — a whole-frame densest-dark-blob search
    latches onto desert scenery, not the avatar (#60). Returns frame-relative
    (x, y) or None. Without an anchor, falls back to the whole-frame find."""
    if anchor is None:
        return find_avatar(frame)
    cx = anchor[0] + ANCHOR_AVATAR_DX - play_region["left"]  # avatar x in the frame
    x0 = max(0, cx - ANCHOR_AVATAR_HALF_W)
    x1 = min(frame.shape[1], cx + ANCHOR_AVATAR_HALF_W)
    y1 = min(frame.shape[0], ANCHOR_AVATAR_BAND_H)
    if x1 <= x0 or y1 <= 0:
        return find_avatar(frame)
    found = find_avatar(frame[0:y1, x0:x1])
    if found is None:
        return None
    return (found[0] + x0, found[1])


def _env_flag(name: str) -> bool:
    """Truthy env var? Lets the launcher GUI (which passes per-bot options as
    env vars, not CLI args) enable flags that the terminal sets via argparse."""
    return os.environ.get(name, "").strip().lower() in ("1", "on", "true", "yes")


def timed_flap_due(fly_x, fly_y, gap_left_x, gap_center_y,
                   lead_dist_px=LEAD_DIST_PX, low_offset_px=TIMED_LOW_OFFSET_PX) -> bool:
    """Fire the phase-timed flap: the hoop's leading edge is within
    lead_dist_px of the avatar AND the avatar is well below the hoop centre
    (low_offset_px down, i.e. low in its bob), so the flap launches a
    consistent slow descent through the hole as the hoop crosses. False with no
    hoop, or while the avatar is high (a flap there over-lifts past the ring)."""
    if gap_left_x is None or gap_center_y is None:
        return False
    dist = gap_left_x - fly_x
    return 0 < dist <= lead_dist_px and fly_y >= gap_center_y + low_offset_px


def startup_flap_due(fly_y, play_height, play_started,
                     frac=STARTUP_FLAP_FRAC) -> bool:
    """During the startup window (after PLAY GAME is clicked, before the game
    is confirmed), flap a detected fly that has fallen below `frac` of the play
    height — arresting the spawn free-fall before it reaches the floor.
    `play_started` is True once PLAY GAME has been clicked, so a world-map
    scenery blob (no prompt) never triggers it."""
    return bool(play_started) and fly_y > play_height * frac


def fly_started_moving(recent_ys, min_range=START_MOTION_RANGE_PX) -> bool:
    """True once the detected fly's y has spanned at least `min_range` px over
    the buffered detections — i.e. a real bobbing/free-falling fly, not a
    static scenery blob mistaken for the fly at the entry prompt. The gate
    that stops a false fly from being read as a started game (see the desert-
    region failure in START_MOTION_RANGE_PX's note)."""
    return len(recent_ys) >= 2 and max(recent_ys) - min(recent_ys) >= min_range


def _read_score(anchor, win_left, win_top, win_w, win_h):
    """Grab the "N PTS" readout (anchored to the prompt, like the play crop)
    and OCR it, or None. Separate small grab so it doesn't disturb the flap
    loop's frame. Returns None with no anchor, nothing readable, or a digit
    not in the template library yet."""
    if anchor is None:
        return None
    bx, by = anchor
    left = max(0, bx + ANCHOR_SCORE_DX)
    top = max(0, by + ANCHOR_SCORE_DY)
    w = min(win_w - left, ANCHOR_SCORE_W)
    h = min(win_h - top, ANCHOR_SCORE_H)
    if w <= 0 or h <= 0:
        return None
    frame = grab_region(win_left + left, win_top + top, w, h)
    crop = score_region(frame, 0, 0, w, h)
    return _read_pts(crop)


def run():
    parser = argparse.ArgumentParser(description="Catching (Flappy) minigame bot")
    parser.add_argument(
        "--save-frames", action="store_true",
        help="Save full-window + play-region snapshots to assets/captures/"
             "botrun_<stamp>/ (gitignored) for offline detector diagnosis — "
             "what the fly/ring detectors actually saw. Throttled and placed "
             "after the flap, so it never delays a click. The launcher's "
             "'Save frames' toggle sets CATCHING_SAVE_FRAMES instead.",
    )
    parser.add_argument(
        "--trace", action="store_true",
        help="Write a dense per-poll trajectory CSV to assets/traces/ "
             "(gitignored) — every poll's (t, fly_y, fly_vy, gap, fired, "
             "where) for offline fitting of the avatar's vertical dynamics "
             "(controller.py, #60). Cheap (no image encode); pairs well with "
             "--save-frames. The launcher's toggle sets CATCHING_TRACE.",
    )
    parser.add_argument(
        "--model", action="store_true",
        help="Use the fitted predictive flap timer (controller.py, #60) for "
             "the phase-timed launch instead of the hand-tuned constants — "
             "needs assets/dynamics.json (fit a --trace run first). Falls back "
             "to the hand-tuned timing if the file is missing. The launcher's "
             "toggle sets CATCHING_USE_MODEL.",
    )
    args = parser.parse_args()
    # The GUI can't pass --save-frames (it launches with no CLI args), so also
    # honour the CATCHING_SAVE_FRAMES env var its 'Save frames' toggle sets.
    save_frames = args.save_frames or _env_flag("CATCHING_SAVE_FRAMES")
    trace = args.trace or _env_flag("CATCHING_TRACE")
    use_model = args.model or _env_flag("CATCHING_USE_MODEL")
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
        stats = {"n_flaps": 0, "end_reason": "process_exit", "final_score": None,
                 "_trace_file": None}
        try:
            _run_inner(session_started, db, code_commit, stats,
                       save_frames=save_frames, trace=trace, use_model=use_model)
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
            if stats.get("_trace_file") is not None:
                stats["_trace_file"].close()  # flushes the buffered trace
            db.close()
            # Tracked DB — other machines get the data via `git pull`.
            commit_file_if_changed(
                REPO_ROOT,
                "minigames/catching/assets/catching.db",
                "chore(catching): refresh catching.db (auto)",
            )


def _run_inner(session_started, db, code_commit, stats, save_frames=False,
               trace=False, use_model=False):
    print(f"Catching bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    # Predictive flap timer (#60): only active when --model is set AND a fitted
    # dynamics file exists, else the hand-tuned timed_flap_due drives the launch
    # (so a missing/half-baked model can't regress the working ~2-hoop play).
    dyn = load_dynamics(DYNAMICS_PATH) if use_model else None
    if use_model and dyn is None:
        print(f"--model set but no fitted dynamics at {DYNAMICS_PATH.name} — "
              "using the hand-tuned phase timing. Fit a --trace run first.")
    elif dyn is not None:
        print(f"Predictive flap timer ON: g={dyn.gravity:.0f} "
              f"flap_vy={dyn.flap_vy:.0f} approach={dyn.approach_speed:.0f}")

    frames_dir = None
    last_frame_save = 0.0
    frame_idx = 0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_frames:
        frames_dir = _HERE / "assets" / "captures" / f"botrun_{stamp}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving frames to {frames_dir} (--save-frames)")

    # Dense trajectory trace (--trace): one CSV row per in-game poll, written
    # at the end of the loop where the detector state + flap decision are all
    # known. Buffered file writes; flushed by the finally on exit.
    trace_writer = trace_file = None
    trace_t0 = time.time()
    if trace:
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        trace_path = TRACES_DIR / f"trace_{stamp}.csv"
        trace_file = open(trace_path, "w", newline="")
        trace_writer = csv.writer(trace_file)
        trace_writer.writerow(TRACE_COLUMNS)
        # run()'s finally closes this on EVERY exit path (return / Ctrl-C /
        # failsafe), flushing the buffer — so no per-poll flush is needed.
        stats["_trace_file"] = trace_file
        print(f"Tracing trajectory to {trace_path} (--trace)")

    last_click_time = 0.0
    prev_fly: tuple[int, float] | None = None  # (fly_y, wall_clock) of last detected frame, for velocity
    last_fly_time = 0.0       # wall-clock of the last fly detection (game-over bail)
    last_motion_time = 0.0    # wall-clock the sprite last moved (motionless game-over)
    game_running = False      # confirmed a REAL game (prompt gone + fly moving)
    anchor = None             # PLAY GAME prompt centre; anchors the play crop (#60)
    start_attempts = 0        # consecutive PLAY-GAME clicks with no game starting
    last_start_click = 0.0    # wall-clock of the last PLAY GAME click
    last_score_sample = 0.0   # wall-clock of the last throttled score read
    last_score = None         # most recent PTS read, logged per flap
    last_gap_center = None     # last detected hoop centre; held over detection gaps
    coast_until = 0.0          # suppress hover flaps until here (timed coast through a hole)
    recent_fly_y: deque[int] = deque(maxlen=START_MOTION_FRAMES)  # for the start-motion gate
    while True:
        check_failsafe()
        try:
            win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        # Once a game has been started, anchor the play crop to the PLAY GAME
        # prompt position (the band moves with the player across regions, #60).
        # Before the first click, fall back to the picked/static play region.
        if anchor is not None:
            play_region = _anchored_play_region(anchor[0], anchor[1], win_w, win_h)
        else:
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
        fly_pos = _find_avatar(frame, play_region, anchor)

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
                anchor = (bx, by)   # band is at a fixed offset from here (#60)
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
                # Arrest the spawn free-fall while confirmation accrues, so the
                # avatar doesn't sink to the floor before control is handed off.
                if (startup_flap_due(fly_pos[1], play_region["height"],
                                     start_attempts > 0)
                        and time.time() - last_click_time >= MIN_CLICK_INTERVAL):
                    cx = play_region["left"] + play_region["width"] // 2
                    cy = play_region["top"] + play_region["height"] // 2
                    click(win_left + cx, win_top + cy)
                    last_click_time = time.time()
            if fly_pos is None or not fly_started_moving(recent_fly_y):
                time.sleep(POLL_INTERVAL)
                continue
            game_running = True
            last_fly_time = time.time()
            last_motion_time = time.time()
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

        # Motionless-sprite game over: a live avatar never holds still, so a
        # sustained ~0 velocity means the attempt ended and find_fly is stuck
        # on a static blob. (fly_vy None = a detection-gap frame; don't let it
        # count as either motion or stillness.)
        if fly_vy is not None and abs(fly_vy) > FREEZE_VY_THRESHOLD:
            last_motion_time = now
        elif fly_vy is not None and now - last_motion_time > STATIC_GAMEOVER_S:
            stats["end_reason"] = "game_over"
            print(f"Sprite motionless {STATIC_GAMEOVER_S:.0f}s — attempt over. "
                  f"Flaps: {stats['n_flaps']}.")
            return

        gap = find_next_gap(frame, fly_pos)
        # Aim at the next hoop's CENTRE when one is visible. The hoop is only
        # detected on ~60% of frames, so on the gaps HOLD the last known hoop
        # centre rather than snapping to a fixed mid-height (which sits below
        # the hoop band and pulls the avatar off it); fall back to the default
        # only until the first hoop is seen.
        if gap is not None:
            gap_top, gap_bottom, gap_left_x, gap_right_x = gap
            detected_center = (gap_top + gap_bottom) // 2
            last_gap_center = detected_center
        else:
            gap_top = gap_bottom = gap_left_x = gap_right_x = None
            detected_center = None
        # Trigger the flap FLAP_CENTER_DROP_PX below the hoop centre so the bob
        # centres on the opening, holding the last centre over detection gaps;
        # default only until the first hoop is seen.
        aim = detected_center if detected_center is not None else last_gap_center
        if aim is not None:
            # Model path hovers the bob's APEX on the hole centre (so the
            # avatar is phase-ready for apex-threading and never sinks); the
            # hand-tuned path drops below centre as before.
            target_y = hover_target_y(aim, dyn) if dyn is not None \
                else aim + FLAP_CENTER_DROP_PX
        else:
            target_y = int(play_region["height"] * DEFAULT_HOVER_FRAC)

        # Flap policy (#60). The bob is bigger than the hole, so coast THROUGH a
        # ring rather than hover: when a hoop's leading edge nears and the avatar
        # is low in its bob, fire ONE timed flap, then suppress hover flaps for
        # COAST_S so the slow descent carries it through the hole (flapping only
        # to stay off the bottom edge). Between rings, hover the centred bob.
        # The launch-flap decision: the fitted model when active, else the
        # hand-tuned phase-timing heuristic. Both gate on the coast window so a
        # launch isn't re-fired mid-coast.
        if dyn is not None:
            launch_due = should_flap_now(
                fly_x, fly_y, gap_left_x, gap_right_x, gap_top, gap_bottom, dyn)
            timed_tag = "model"
        else:
            launch_due = timed_flap_due(fly_x, fly_y, gap_left_x, detected_center)
            timed_tag = "timed"
        is_timed = launch_due and now >= coast_until
        # The phase-timed LAUNCH wins first — it fires in the same low-in-the-
        # bob zone as the floor rescue, and if floor preempts it (run 16: floor
        # 22 vs model 2) the launch is starved and never sets the coast, so the
        # avatar over-lifts into the next crossing and clips the top. The launch
        # also flaps UP, so it's a floor-safe choice; floor backstops the polls
        # where no launch is due (still beats coast, so a held coast / detection
        # gap can't sink the avatar — the 2026-06-17 sink death).
        if is_timed:
            do_flap, where = True, timed_tag
        elif floor_rescue_due(fly_y, fly_vy, play_region["height"]):
            do_flap, where = True, "floor"
        elif now < coast_until:
            floor = (gap_bottom - COAST_RESCUE_PX) if gap_bottom is not None \
                else (play_region["height"] - 25)
            do_flap, where = fly_y > floor, "coast"
        else:
            do_flap = fly_y > target_y + FLAP_MARGIN
            where = f"gap=[{gap_top}..{gap_bottom}]" if gap is not None else "hover"
        fired = False  # whether a click actually fired this poll (vs blocked)
        if do_flap and now - last_click_time >= MIN_CLICK_INTERVAL:
            # Fire immediately on the decision (the fly is still falling) —
            # the hoops/darts click-timing rule; bookkeeping runs after.
            clicked_at = datetime.now().isoformat(timespec="milliseconds")
            cx = play_region["left"] + play_region["width"] // 2
            cy = play_region["top"] + play_region["height"] // 2
            click(win_left + cx, win_top + cy)
            last_click_time = time.time()
            # Start the coast ONLY when the timed flap actually fired — setting
            # it on a rate-limited (blocked) timed flap coasts with no lift and
            # the avatar sinks below the hole.
            if is_timed:
                coast_until = last_click_time + COAST_S
            fired = True
            stats["n_flaps"] += 1
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
                gap_center=detected_center,
                gap_height=(gap_bottom - gap_top) if gap is not None else None,
                fly_offset_below_gap=(fly_y - gap_bottom) if gap is not None else None,
                gap_lower_margin=FLAP_MARGIN,
                window_w=win_w,
                window_h=win_h,
                score=last_score,
                where=where,
                code_commit=code_commit,
                source="bot",
            )
            random_delay(5, 20)

        # Dense trajectory trace (--trace): one row per in-game poll, here in
        # the bookkeeping section so it never sits between a flap decision and
        # its click. `where` is the branch that WOULD fire; `fired` flags
        # whether a click actually went out (vs rate-limited).
        if trace_writer is not None:
            trace_writer.writerow([
                round(now - trace_t0, 4), fly_x, fly_y, fly_vy,
                gap_top, gap_bottom, gap_left_x, gap_right_x, detected_center,
                target_y, int(fired), where,
            ])

        # Throttled score sample — AFTER any flap above, so OCR never sits
        # between a flap decision and its click. Tracks the running max as
        # the run's final_score (the score only climbs, so max == score at
        # death; max guards against a misread-low on the final frame).
        if now - last_score_sample >= SCORE_SAMPLE_INTERVAL:
            last_score_sample = now
            pts = _read_score(anchor, win_left, win_top, win_w, win_h)
            if pts is not None:
                last_score = pts
                if stats["final_score"] is None or pts > stats["final_score"]:
                    stats["final_score"] = pts

        # Optional debug capture (--save-frames): throttled snapshots AFTER the
        # flap so it never delays a click. play_ is the exact detector input
        # (find_fly / find_next_gap); full_ is the whole window for scene
        # context (where the real rings are, the HUD). Both gitignored.
        if frames_dir is not None and now - last_frame_save >= FRAME_SAVE_INTERVAL:
            last_frame_save = now
            save_frame(frames_dir / f"play_{frame_idx:04d}.png", frame)
            save_frame(frames_dir / f"full_{frame_idx:04d}.png",
                       grab_region(win_left, win_top, win_w, win_h))
            frame_idx += 1

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
