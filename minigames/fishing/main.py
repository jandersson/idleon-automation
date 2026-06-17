"""Fishing minigame bot — main loop and config.

The fishing minigame is a hold-to-charge distance game (see
docs/fishing_minigame.md): hold LMB to fill a charge bar, release to land the
lure on a fish (Green/Eel/Squid/Whale) while avoiding mines. The bot casts
CLOSED-LOOP — it polls the charge bar while holding and releases at a target
fill (common.input.charge_and_release) — so the charge LEVEL is the control
variable; its learned model is charge_level <-> cast distance (cast_model),
fitted at startup from logged casts and refined by exploration (darts pattern).

STATUS: auto-starts (clicks the PLAY GAME prompt, like catching/mining — the
cast bar is the 'minigame active' signal), plays the picked cast-bar region
with calibrated colour detection, and aims off the charge bar. The charge fill
is a clean, in-crop, reproducible distance signal (validated offline on the
run-7 frames: landed_x ~ 5.0*charge); the closed-loop release also detects the
rod-not-ready state (fill stays 0 = previous lure still reeling) and skips that
cast instead of burning it (#58). Until MIN_SAMPLES charged casts exist the bot
explores random target charge levels and logs (charge_level, landed_dist).
Open: warm-fish HSV verification + grey-mine detection (#63).
"""
import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.monitor import save_frame
from common.input import charge_and_release, click, random_delay, check_failsafe
from common.regions import get_region
from common.session_log import session_log
from common.window import get_bounds, WindowNotFoundError
from common.git_info import current_code_commit
from common.auto_commit import commit_file_if_changed
from minigames.fishing.detector import (
    find_fish, find_mines, find_lure, find_game_over, kind_at, find_cast_bar,
    find_play_button, find_charge_fill,
)
from minigames.fishing.fish_log import (
    open_db, log_cast, set_outcome, log_run, fetch_cast_samples,
)
from minigames.fishing.cast_model import (
    FISH_VALUE, MIN_SAMPLES, fit_cast_model, charge_for_distance,
    distance_for_charge, choose_target,
)

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
FISH_DB_PATH = _HERE / "assets" / "fishing.db"
REPO_ROOT = _HERE.parent.parent

WINDOW_TITLE = "Legends Of Idleon"
POLL_INTERVAL = 0.05

# Exploration charge range (fill px): random target charge levels within this
# span sample the charge->distance curve until the cast model is fitted.
# The charge thermometer ramps 0->~56 over the up-sweep then turns DOWN (it's a
# release-timing oscillation, #58). Keep explore targets safely below the peak
# so they're always reached on the up-sweep — a target above the peak would
# never trigger the release and the cast would fall to the max-hold backstop.
EXPLORE_CHARGE_MIN = 10
EXPLORE_CHARGE_MAX = 50
# Fire an exploration cast every Nth cast even after the model is fitted, so
# the charge->distance surface keeps getting sampled (darts EXPLORE_EVERY_N).
EXPLORE_EVERY_N = 10

# Closed-loop charge-and-release timing (common.input.charge_and_release):
# poll the charge bar this often while holding; give the rod this long to
# start charging before declaring it not-ready (previous lure still reeling);
# never hold longer than this (the bar saturates ~750ms, so this bounds a
# stuck hold and the wait on an over-cap target).
CHARGE_POLL_S = 0.025
# With the thermometer read correctly (find_charge_fill), the fill rises from 0
# within ~2-3 polls of a real hold, so the not-ready grace can be short; a
# genuinely-reeling rod leaves it at 0 and aborts here. (The old 0.8s was a
# band-aid for the wrong reading — x<10 read 0 forever, runs 8-10, #58.)
CHARGE_READY_GRACE_S = 0.4
# Backstop for a target the up-sweep never reaches: release near the up-sweep
# PEAK (~780ms) rather than holding into the down-sweep, which would cast at an
# unpredictable (low) fill. So cap just past the peak.
CHARGE_MAX_HOLD_S = 0.95
# After a not-ready abort, wait this long before retrying (the rod recovers
# over ~a cast cycle; the aborts are cheap — they poll for ready without
# burning a cast, unlike the old fixed-cooldown open-loop hold).
NOT_READY_RETRY_S = 0.3

# Landing measurement: the lure REELS IN after landing (the bobber's x
# decreases poll-to-poll until it vanishes — confirmed in the --save-frames
# trajectories, e.g. 161->126->62->15->gone). So the landing is the FARTHEST x
# the bobber reaches (max x), NOT the last/settled read (that's the retraction
# endpoint — the bug behind the same hold "landing" at 256 and 27). Poll,
# track the max x, and early-exit once the bobber retracts past it
# (x < max - LANDING_RETRACT_PX) or vanishes after being seen — keeping the
# grab count (screen-capture lag) low. CAST_SETTLE_S caps the wait (the lure
# arcs high / off-crop before landing on long casts).
CAST_SETTLE_S = 1.6
LANDING_POLL_S = 0.1
LANDING_RETRACT_PX = 25   # x dropped this far below the max = reeling in (landed)
# Wait between casts (the next charge can't start until the lure resets).
CAST_COOLDOWN_S = 0.6
# Auto-start (like catching/mining): click the PLAY GAME prompt to begin ONE
# minigame, play it, then STOP on game-over (no auto-replay). After a click,
# wait this long before re-clicking (the minigame is loading).
START_CLICK_COOLDOWN_S = 2.0
# Give up after this many PLAY GAME clicks that don't produce a cast bar.
MAX_START_ATTEMPTS = 3
# Once playing, the cast bar being gone this long means the attempt ended
# (rides out brief detection dropouts). The cast bar is the 'active' signal —
# it only exists during play, unlike a fish/colour count that the entry scene
# would false-trigger. Game-over template detection (game_over.png) is primary
# when that asset exists.
BAR_GONE_GAMEOVER_S = 3.0


def _measure_landing(win_left, win_top, play, settle_s, poll_s,
                     frames_dir=None, cast_idx=0):
    """Poll find_lure after a cast and return (landed (x, y), frame): the
    bobber's FARTHEST x = the landing (it reels in after landing, so max-x is
    the landing, not the settled/retracted read). Exits early once the bobber
    retracts past the max or vanishes after being seen, keeping the grab count
    low. (None, None) if never seen (cast landed off the play region).

    The cast power is the charge bar (returned by charge_and_release at
    release), not measured here — this poll only locates the bobber to classify
    the hit and supply landed_dist as the model's training target.

    When frames_dir is set (--save-frames), every poll frame is saved with the
    find_lure x (or NA) in the name — for debugging which moment the landing
    read latches onto (#58)."""
    best, best_frame = None, None   # (x, y) at the FARTHEST x = the landing
    seen = False
    deadline = time.time() + settle_s
    j = 0
    while time.time() < deadline:
        post = grab_region(
            win_left + play["left"], win_top + play["top"],
            play["width"], play["height"],
        )
        lure = find_lure(post)
        if frames_dir is not None:
            lx = lure[0] if lure is not None else "NA"
            save_frame(frames_dir / f"cast{cast_idx:02d}_p{j:02d}_x{lx}.png", post)
        j += 1
        if lure is not None:
            seen = True
            if best is None or lure[0] > best[0]:
                best, best_frame = lure, post          # new farthest = landing
            elif lure[0] < best[0] - LANDING_RETRACT_PX:
                break              # reeling in past the landing -> done
        elif seen:
            break                  # bobber vanished after landing -> done
        time.sleep(poll_s)
    return best, best_frame


def _cast_origin(play_region: dict, bar: tuple[int, int, int, int] | None) -> tuple[int, int]:
    """Lure cast origin (play-region-relative) — distance is measured from
    here, so it anchors the cast model. The lure is cast RIGHTWARD along the
    cast bar, so the origin is the bar's LEFT edge at its mid-height (the track
    start near the player). Falls back to the play-region centre when the bar
    isn't detected."""
    if bar is not None:
        x, y, w, h = bar
        return (x, y + h // 2)
    return (play_region["width"] // 2, int(play_region["height"] * 0.66))


def run():
    parser = argparse.ArgumentParser(description="Fishing minigame bot")
    parser.add_argument(
        "--save-frames", action="store_true",
        help="Save each cast's landing-poll frames to assets/captures/"
             "botrun_<stamp>/ (gitignored), named with the find_lure x — to "
             "debug landing detection (the launcher's toggle sets "
             "FISHING_SAVE_FRAMES).")
    args = parser.parse_args()
    save_frames = args.save_frames or os.environ.get(
        "FISHING_SAVE_FRAMES", "").strip().lower() in ("1", "on", "true", "yes")
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        session_started = datetime.now().isoformat(timespec="seconds")
        code_commit = current_code_commit(REPO_ROOT)
        if code_commit:
            print(f"Code commit: {code_commit}")
        db = open_db(FISH_DB_PATH)
        # Fit the hold<->distance cast model from past measured casts (darts
        # startup-fit pattern). None until MIN_SAMPLES landings exist, which
        # needs assets/lure.png for landing measurement — until then the bot
        # explores (random holds) and logs them.
        samples = fetch_cast_samples(db)
        model = fit_cast_model(samples)
        if model is None:
            print(f"cast model: not fitted ({len(samples)} charged casts "
                  f"< {MIN_SAMPLES}) — exploring charge levels "
                  f"[{EXPLORE_CHARGE_MIN},{EXPLORE_CHARGE_MAX}]")
        else:
            lo, hi = model.reach_px()
            print(f"cast model: fitted on {model.n} casts — "
                  f"charge {model.charge_min}-{model.charge_max} reaches "
                  f"{lo:.0f}-{hi:.0f} px")
        started_at = time.time()
        stats = {"n_casts": 0, "n_not_ready": 0, "points_total": 0,
                 "max_streak": 0, "end_reason": "process_exit"}
        try:
            _run_inner(session_started, db, code_commit, model, stats,
                       save_frames=save_frames)
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
                n_casts=stats["n_casts"],
                n_not_ready=stats["n_not_ready"],
                points_total=stats["points_total"] or None,
                max_streak=stats["max_streak"],
                duration_s=round(time.time() - started_at, 1),
                end_reason=stats["end_reason"],
                code_commit=code_commit,
            )
            db.close()
            commit_file_if_changed(
                REPO_ROOT,
                "minigames/fishing/assets/fishing.db",
                "chore(fishing): refresh fishing.db (auto)",
            )


def _run_inner(session_started, db, code_commit, model, stats, save_frames=False):
    print(f"Fishing bot starting — tracking window {WINDOW_TITLE!r}. "
          f"Move mouse to a corner to abort.")
    time.sleep(2)

    frames_dir = None
    if save_frames:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frames_dir = _HERE / "assets" / "captures" / f"botrun_{stamp}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving cast frames to {frames_dir} (--save-frames)")

    game_running = False        # confirmed active: the cast bar is present
    start_attempts = 0          # consecutive PLAY GAME clicks with no game started
    last_start_click = 0.0
    last_active_time = time.time()   # wall-clock the cast bar was last seen
    streak = 0
    charge_attempt = 0   # every charge_and_release call (ready or not) — frame tag
    while True:
        check_failsafe()
        try:
            win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        play = get_region(_HERE, "play", win_w, win_h)
        if play is None:
            print("No 'play' region in regions.json. Run fishing-pick-play-region first.")
            time.sleep(2)
            continue

        frame = grab_region(
            win_left + play["left"], win_top + play["top"],
            play["width"], play["height"],
        )

        # The cast bar is the 'minigame active' signal — it exists only during
        # play (a fish/colour count would false-trigger on the entry scene).
        bar = find_cast_bar(frame)

        if bar is None:
            if not game_running:
                # --- Startup: click the PLAY GAME prompt (like catching/mining).
                # The prompt sits above the player, not in the cast-bar region,
                # so match it on the FULL window.
                full = grab_region(win_left, win_top, win_w, win_h)
                btn = find_play_button(full)
                if btn is not None:
                    if time.time() - last_start_click < START_CLICK_COOLDOWN_S:
                        time.sleep(POLL_INTERVAL)
                        continue
                    if start_attempts >= MAX_START_ATTEMPTS:
                        stats["end_reason"] = "no_start"
                        print(f"Clicked PLAY GAME {start_attempts}x with no minigame "
                              f"starting — giving up.")
                        return
                    bx, by = btn
                    start_attempts += 1
                    last_start_click = time.time()
                    print(f"PLAY GAME at ({bx},{by}) — starting minigame "
                          f"(attempt {start_attempts}/{MAX_START_ATTEMPTS}).")
                    click(win_left + bx, win_top + by)
                    continue
                # No prompt and no bar yet — the minigame is loading; wait.
                time.sleep(POLL_INTERVAL)
                continue
            # Was playing and the bar vanished -> the attempt ended. STOP (one
            # game per run, like catching — no auto-replay).
            if time.time() - last_active_time > BAR_GONE_GAMEOVER_S:
                stats["end_reason"] = "game_over"
                print(f"Cast bar gone {BAR_GONE_GAMEOVER_S:.0f}s — minigame over. "
                      f"Casts={stats['n_casts']} points={stats['points_total']}.")
                return
            time.sleep(POLL_INTERVAL)
            continue

        # --- Cast bar present: the minigame is active ---
        if not game_running:
            game_running = True
            print(f"Minigame active (cast bar at {bar}) — casting.")
        last_active_time = time.time()

        is_over, go_conf = find_game_over(frame)
        if is_over:
            stats["end_reason"] = "game_over"
            print(f"Game over detected (conf={go_conf:.2f}). "
                  f"Casts={stats['n_casts']} points={stats['points_total']}.")
            return

        # Confine detection to the cast bar — over the raw frame the world
        # scenery (tan dock, shore plants) floods the colour masks (#63).
        fish = find_fish(frame, bar=bar)
        mines = find_mines(frame, bar=bar)
        if not fish:
            # Bar present but no fish this poll (detection gap / between spawns).
            time.sleep(POLL_INTERVAL)
            continue

        origin_x, origin_y = _cast_origin(play, bar)
        explore = model is None or (stats["n_casts"] + 1) % EXPLORE_EVERY_N == 0

        if explore:
            target_charge = random.randint(EXPLORE_CHARGE_MIN, EXPLORE_CHARGE_MAX)
            target, aim_mode = None, "explore"
        else:
            target = choose_target(fish, model, origin_x)
            if target is None:
                target_charge = random.randint(EXPLORE_CHARGE_MIN, EXPLORE_CHARGE_MAX)
                aim_mode = "fallback"
            else:
                target_charge = charge_for_distance(model, target["target_dist"])
                aim_mode = "model"
        predicted = distance_for_charge(model, target_charge) if model else None

        # Closed-loop cast: hold while polling the charge bar, release at
        # target_charge. The cast distance is set by the charge level we release
        # at (not where the click lands), so there's no sampled-then-stale
        # latency to fight as in the click-timing games — but no disk writes go
        # between the decision and the hold. The read_charge closure grabs the
        # play crop and reads the left-edge fill each poll.
        fired_at = datetime.now().isoformat(timespec="milliseconds")
        cx = win_left + play["left"] + play["width"] // 2
        cy = win_top + play["top"] + play["height"] // 2

        # The LIVE charge meter is a vertical red thermometer LEFT of the cast
        # bar (find_charge_fill), read from the FULL window anchored to the cast
        # bar in full-window coords (the play crop's x<10 misses it entirely —
        # the thermometer renders left of the crop, #58). The closure tracks the
        # PEAK fill (and dumps each poll's full frame with --save-frames).
        charge_attempt += 1
        charge_dbg = {"peak": 0, "polls": 0, "attempt": charge_attempt}
        cast_bar_full = (play["left"] + bar[0], play["top"] + bar[1], bar[2], bar[3])

        def _read_charge():
            full = grab_region(win_left, win_top, win_w, win_h)
            c = find_charge_fill(full, cast_bar_full)
            charge_dbg["peak"] = max(charge_dbg["peak"], c)
            if frames_dir is not None:
                save_frame(frames_dir / f"chargefull{charge_dbg['attempt']:03d}_"
                           f"p{charge_dbg['polls']:02d}_c{c}.png", full)
            charge_dbg["polls"] += 1
            return c

        charge, ready = charge_and_release(
            cx, cy, target_charge, _read_charge,
            poll_s=CHARGE_POLL_S, ready_grace_s=CHARGE_READY_GRACE_S,
            max_hold_s=CHARGE_MAX_HOLD_S)

        if not ready:
            # Rod not ready: the fill never crossed the floor within the grace
            # (previous lure still reeling, OR the bar didn't start charging in
            # time). Skip the cast — don't log/count it — and retry after a beat.
            # The peak/polls report makes the not-charging vs not-ready split
            # visible without needing the saved frames.
            stats["n_not_ready"] += 1
            print(f"rod not ready: peak charge {charge_dbg['peak']} over "
                  f"{charge_dbg['polls']} polls / {CHARGE_READY_GRACE_S}s — waiting")
            time.sleep(NOT_READY_RETRY_S)
            continue
        stats["n_casts"] += 1

        tag = (f"{target['kind']}@{target['target_dist']}px" if target else "explore")
        print(f"cast #{stats['n_casts']} charge={charge} (aim {target_charge}) "
              f"[{aim_mode}] -> {tag} "
              f"(fish={len(fish)} mines={len(mines)} streak={streak})")

        row_id = log_cast(
            db,
            session_started=session_started,
            attempt_idx=1,
            cast_idx=stats["n_casts"],
            fired_at=fired_at,
            hold_ms=None,           # closed-loop: charge is the control, not a fixed hold
            aim_mode=aim_mode,
            cast_origin_x=origin_x,
            cast_origin_y=origin_y,
            target_kind=(target["kind"] if target else "explore"),
            target_x=(target["x"] if target else None),
            target_dist_px=(target["target_dist"] if target else None),
            target_charge_level=target_charge,
            predicted_dist_px=(int(predicted) if predicted is not None else None),
            streak_before=streak,
            n_fish=len(fish),
            n_mines=len(mines),
            window_w=win_w,
            window_h=win_h,
            code_commit=code_commit,
            source="bot",
        )

        # Locate the bobber landing (max-x poll) to classify the hit and supply
        # landed_dist (the charge->distance model's training target). The cast
        # power is `charge` from the release above — the robust, in-crop signal.
        lure, post = _measure_landing(win_left, win_top, play,
                                      CAST_SETTLE_S, LANDING_POLL_S,
                                      frames_dir, stats["n_casts"])
        landed_x = landed_dist = landed_kind = points = made = None
        if lure is not None:
            landed_x, landed_y = lure
            landed_kind = kind_at(post, landed_x, landed_y) or "miss"
            points = FISH_VALUE.get(landed_kind, 0)
            made = 1 if points > 0 else 0
            landed_dist = abs(landed_x - origin_x)
            stats["points_total"] += points
            # Streak: any fish landing extends it; a Whale catch resets to 1
            # (wiki); a miss breaks it.
            streak = (1 if landed_kind == "whale" else streak + 1) if made else 0
            stats["max_streak"] = max(stats["max_streak"], streak)
        set_outcome(
            db, row_id,
            landed_x=landed_x,
            landed_dist_px=landed_dist,
            landed_kind=landed_kind,
            points=points,
            made=made,
            charge_level=charge or None,
        )

        random_delay(int(CAST_COOLDOWN_S * 1000), int(CAST_COOLDOWN_S * 1000) + 200)


if __name__ == "__main__":
    run()
