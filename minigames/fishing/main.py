"""Fishing minigame bot — main loop and config.

The fishing minigame is a hold-to-cast distance game (see
docs/fishing_minigame.md): hold LMB to charge, release to land the lure on
a fish (Green/Eel/Squid/Whale) while avoiding mines. The bot's control is
the hold duration; its learned model is hold_ms <-> cast distance
(cast_model), fitted at startup from logged casts and refined by
exploration — the darts pattern.

STATUS: auto-starts (clicks the PLAY GAME prompt, like catching/mining — the
cast bar is the 'minigame active' signal), plays the picked cast-bar region
with calibrated colour detection, and explores hold durations. Still needed to
SCORE: assets/lure.png (capture via fishing-capture) so landings — and thus the
hold<->distance cast model — can be measured, and the #63 detection refinement
(eel/megalodon over-detect on score text / mine cores) so target selection is
reliable. Until the model fits the bot casts random holds and logs them.
"""
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import hold, click, random_delay, check_failsafe
from common.regions import get_region
from common.session_log import session_log
from common.window import get_bounds, WindowNotFoundError
from common.git_info import current_code_commit
from common.auto_commit import commit_file_if_changed
from minigames.fishing.detector import (
    find_fish, find_mines, find_lure, find_game_over, kind_at, find_cast_bar,
    find_play_button,
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

# Landing measurement: POLL find_lure over this window after a cast and take
# the detection lowest on the bar (largest y) as the landing — a single grab
# at a fixed delay missed the bobber mid-arc (run 2: only 3/24 casts measured).
# The window must cover the longest cast's flight + settle.
CAST_SETTLE_S = 1.6
LANDING_POLL_S = 0.1
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


def _measure_landing(win_left, win_top, play, settle_s, poll_s):
    """Poll find_lure over settle_s; return (landed (x, y), the frame it was
    in) for the LANDED bobber — the detection lowest on the bar (largest y),
    since mid-flight the bobber arcs high (small y / off the crop). (None,
    None) if it's never seen (the cast landed off the play region)."""
    best, best_frame = None, None
    deadline = time.time() + settle_s
    while time.time() < deadline:
        post = grab_region(
            win_left + play["left"], win_top + play["top"],
            play["width"], play["height"],
        )
        lure = find_lure(post)
        if lure is not None and (best is None or lure[1] > best[1]):
            best, best_frame = lure, post
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

    game_running = False        # confirmed active: the cast bar is present
    start_attempts = 0          # consecutive PLAY GAME clicks with no game started
    last_start_click = 0.0
    last_active_time = time.time()   # wall-clock the cast bar was last seen
    streak = 0
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

        # Measure the landing by polling for the bobber over the settle window
        # (a single grab missed it mid-arc). Returns the landed position + the
        # frame, so kind_at classifies what it landed on.
        lure, post = _measure_landing(win_left, win_top, play,
                                      CAST_SETTLE_S, LANDING_POLL_S)
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
