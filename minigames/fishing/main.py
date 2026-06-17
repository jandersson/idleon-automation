"""Fishing minigame bot — main loop and config.

The fishing minigame is a hold-to-cast distance game (see
docs/fishing_minigame.md): hold LMB to charge, release to land the lure on
a fish (Green/Eel/Squid/Whale) while avoiding mines. The bot's control is
the hold duration; its learned model is hold_ms <-> cast distance
(cast_model), fitted at startup from logged casts and refined by
exploration — the darts pattern.

SCAFFOLD STATUS: structurally complete and runnable in observe mode, but
detection (HSV ranges in detector.py) and the cast geometry (origin,
hold<->distance) need live calibration before it actually scores. First
steps: `fishing-pick-play-region`, then `fishing-observe` to verify
detection, `fishing-calibrate` to tune the colour masks, and capture
assets/lure.png so landings (and thus the cast model) can be measured.
"""
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import hold, random_delay, check_failsafe
from common.regions import get_region
from common.session_log import session_log
from common.window import get_bounds, WindowNotFoundError
from common.git_info import current_code_commit
from common.auto_commit import commit_file_if_changed
from minigames.fishing.detector import (
    find_fish, find_mines, find_lure, find_game_over, kind_at,
)
from minigames.fishing.fish_log import (
    open_db, log_cast, set_outcome, log_run, fetch_cast_samples,
)
from minigames.fishing.cast_model import (
    FISH_VALUE, MIN_SAMPLES, fit_cast_model, hold_for_distance,
    distance_for_hold, choose_target,
)

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
FISH_DB_PATH = _HERE / "assets" / "fishing.db"
REPO_ROOT = _HERE.parent.parent

WINDOW_TITLE = "Legends Of Idleon"
POLL_INTERVAL = 0.05

# Exploration hold range (ms): random holds within this span sample the
# hold->distance curve until the cast model is fitted. PLACEHOLDER bounds —
# widen/narrow once a real cast's hold-to-reach is observed.
EXPLORE_HOLD_MIN_MS = 150
EXPLORE_HOLD_MAX_MS = 1200
# Fire an exploration cast every Nth cast even after the model is fitted,
# so the hold->distance surface keeps getting sampled (darts EXPLORE_EVERY_N).
EXPLORE_EVERY_N = 10

# Time for the lure to fly + land before measuring the outcome. PLACEHOLDER.
CAST_SETTLE_S = 0.7
# Wait between casts (the next charge can't start until the lure resets).
CAST_COOLDOWN_S = 0.6
# Bail if no fish are seen this long — the minigame scene is gone (the
# darts no-pose timeout analogue). Game-over template detection is primary
# once assets/game_over.png exists.
NO_FISH_TIMEOUT_S = 20.0


def _cast_origin(play_region: dict) -> tuple[int, int]:
    """Lure cast origin (play-region-relative). PLACEHOLDER: horizontal
    centre, lower third — the rod tip. Calibrate from a captured frame; the
    target distance is measured from here, so it must be right for the cast
    model to mean anything."""
    return (play_region["width"] // 2, int(play_region["height"] * 0.66))


def run():
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
            print(f"cast model: not fitted ({len(samples)} measured casts "
                  f"< {MIN_SAMPLES}) — exploring holds "
                  f"[{EXPLORE_HOLD_MIN_MS},{EXPLORE_HOLD_MAX_MS}] ms")
        else:
            lo, hi = model.reach_px()
            print(f"cast model: fitted on {model.n} casts — "
                  f"hold {model.hold_min_ms}-{model.hold_max_ms} ms reaches "
                  f"{lo:.0f}-{hi:.0f} px")
        started_at = time.time()
        stats = {"n_casts": 0, "points_total": 0, "max_streak": 0,
                 "end_reason": "process_exit"}
        try:
            _run_inner(session_started, db, code_commit, model, stats)
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


def _run_inner(session_started, db, code_commit, model, stats):
    print(f"Fishing bot starting — tracking window {WINDOW_TITLE!r}. "
          f"Move mouse to a corner to abort.")
    time.sleep(2)

    streak = 0
    last_fish_time = time.time()
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

        is_over, go_conf = find_game_over(frame)
        if is_over:
            print(f"Game over detected (conf={go_conf:.2f}). "
                  f"Casts={stats['n_casts']} points={stats['points_total']}.")
            return

        fish = find_fish(frame)
        mines = find_mines(frame)
        if not fish:
            if time.time() - last_fish_time > NO_FISH_TIMEOUT_S:
                print(f"No fish seen for {NO_FISH_TIMEOUT_S:.0f}s — assuming the "
                      f"minigame ended. Casts={stats['n_casts']}.")
                return
            time.sleep(POLL_INTERVAL)
            continue
        last_fish_time = time.time()

        origin_x, origin_y = _cast_origin(play)
        explore = model is None or (stats["n_casts"] + 1) % EXPLORE_EVERY_N == 0

        if explore:
            hold_ms = random.randint(EXPLORE_HOLD_MIN_MS, EXPLORE_HOLD_MAX_MS)
            target, aim_mode = None, "explore"
            predicted = distance_for_hold(model, hold_ms) if model else None
        else:
            target = choose_target(fish, model, origin_x)
            if target is None:
                hold_ms = random.randint(EXPLORE_HOLD_MIN_MS, EXPLORE_HOLD_MAX_MS)
                aim_mode = "fallback"
                predicted = distance_for_hold(model, hold_ms) if model else None
            else:
                hold_ms = hold_for_distance(model, target["target_dist"])
                aim_mode = "model"
                predicted = distance_for_hold(model, hold_ms)

        # Fire immediately on the decision — the fish move, so any latency
        # between sampling positions and casting biases the target (the
        # hoops/darts click-timing rule). Bookkeeping runs after the cast.
        fired_at = datetime.now().isoformat(timespec="milliseconds")
        cx = win_left + play["left"] + play["width"] // 2
        cy = win_top + play["top"] + play["height"] // 2
        hold(cx, cy, hold_ms)
        stats["n_casts"] += 1

        tag = (f"{target['kind']}@{target['target_dist']}px" if target else "explore")
        print(f"cast #{stats['n_casts']} hold={hold_ms}ms [{aim_mode}] -> {tag} "
              f"(fish={len(fish)} mines={len(mines)} streak={streak})")

        row_id = log_cast(
            db,
            session_started=session_started,
            attempt_idx=1,
            cast_idx=stats["n_casts"],
            fired_at=fired_at,
            hold_ms=hold_ms,
            aim_mode=aim_mode,
            cast_origin_x=origin_x,
            cast_origin_y=origin_y,
            target_kind=(target["kind"] if target else "explore"),
            target_x=(target["x"] if target else None),
            target_dist_px=(target["target_dist"] if target else None),
            predicted_dist_px=(int(predicted) if predicted is not None else None),
            streak_before=streak,
            n_fish=len(fish),
            n_mines=len(mines),
            window_w=win_w,
            window_h=win_h,
            code_commit=code_commit,
            source="bot",
        )

        # Measure the landing after the lure settles. Needs assets/lure.png;
        # until that template exists find_lure returns None and the outcome
        # stays unmeasured (the cast model then can't train — see docs).
        time.sleep(CAST_SETTLE_S)
        post = grab_region(
            win_left + play["left"], win_top + play["top"],
            play["width"], play["height"],
        )
        lure = find_lure(post)
        if lure is not None:
            landed_x, landed_y = lure
            landed_kind = kind_at(post, landed_x, landed_y)
            points = FISH_VALUE.get(landed_kind, 0) if landed_kind else 0
            made = 1 if landed_kind in FISH_VALUE and points > 0 else 0
            set_outcome(
                db, row_id,
                landed_x=landed_x,
                landed_dist_px=abs(landed_x - origin_x),
                landed_kind=landed_kind or "miss",
                points=points,
                made=made,
            )
            stats["points_total"] += points
            # Streak: any fish landing extends it; a Whale catch resets to 1
            # (wiki); a miss breaks it.
            if made:
                streak = 1 if landed_kind == "whale" else streak + 1
            else:
                streak = 0
            stats["max_streak"] = max(stats["max_streak"], streak)

        random_delay(int(CAST_COOLDOWN_S * 1000), int(CAST_COOLDOWN_S * 1000) + 200)


if __name__ == "__main__":
    run()
