"""Watch the user play hoops; log each manual shot to shots.db.

Uses the generic observer framework (`common.observer`) for the listener
+ overlay; this module is just the hoops-specific click handler that
runs detectors, OCRs the score, and writes a row.

Rows are tagged `source="human"` so future predictor fits can
distinguish bot-derived data from observed-human data. Useful for
seeding the predictor with makes in regions the bot can't reach.
"""
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.observer import ObserverConfig, run_observer
from common.regions import get_region
from common.score_diff import score_region
from common.score_ocr import read_score
from common.session_log import session_log
from common.shot_log import open_db, log_shot, current_code_commit
from common.window import get_bounds, WindowNotFoundError
from minigames.hoops.detector import find_rim, find_platform, find_game_over
from minigames.hoops.main import (
    SHOT_DB_PATH, LOGS_DIR, REPO_ROOT, WINDOW_TITLE, POST_SHOT_COOLDOWN,
)

_HERE = Path(__file__).parent


def _capture_score_region(left: int, top: int, width: int, height: int):
    region = get_region(_HERE, "score", width, height)
    if region is None:
        return None
    frame = grab_region(left, top, width, height)
    return score_region(
        frame, region["left"], region["top"],
        region["width"], region["height"],
    )


def run():
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        session_started = datetime.now().isoformat(timespec="seconds")
        code_commit = current_code_commit(REPO_ROOT)
        if code_commit:
            print(f"Code commit: {code_commit}")
        shot_db = open_db(SHOT_DB_PATH)
        pending_timers: list[threading.Timer] = []
        try:
            on_click = _make_handler(shot_db, session_started, code_commit, pending_timers)
            print(
                f"Hoops OBSERVE — click on {WINDOW_TITLE!r} to record shots.\n"
                "  Overlay: GO (green) = ready, countdown (red) = wait.\n"
                "  Move mouse to a screen corner to abort.\n"
            )
            n = run_observer(
                ObserverConfig(
                    window_title=WINDOW_TITLE,
                    cooldown_seconds=POST_SHOT_COOLDOWN,
                    stop_check=_check_game_over,
                ),
                on_click,
            )
            print(f"Observe session ended. Captured {n} clicks.")
        finally:
            # Wait for any post_shot timers still in flight before
            # closing the DB. Without this, click N's row never gets
            # written if the session ends within POST_SHOT_COOLDOWN
            # seconds of the click (game-over auto-exit, fail-safe).
            in_flight = [t for t in pending_timers if t.is_alive()]
            if in_flight:
                print(f"  Waiting for {len(in_flight)} pending post-shot timer(s) to finish…")
                for t in in_flight:
                    t.join(timeout=POST_SHOT_COOLDOWN + 2)
            shot_db.close()


def _check_game_over() -> bool:
    """Return True if the game-over banner is on screen. Called every
    couple of seconds by the observer's overlay loop so we exit
    cleanly when the trial ends instead of letting the user fail-safe."""
    try:
        left, top, w, h = get_bounds(WINDOW_TITLE)
    except WindowNotFoundError:
        return False
    frame = grab_region(left, top, w, h)
    is_over, _conf = find_game_over(frame)
    return is_over


def _make_handler(shot_db, session_started: str, code_commit: str | None,
                  pending_timers: list[threading.Timer]):
    def on_click(idx: int, click_x_rel: int, click_y_rel: int,
                 frame: np.ndarray) -> None:
        # Re-fetch bounds for post-shot work (they shouldn't change but
        # we want a stable reference for the timer).
        try:
            left, top, width, height = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError:
            return
        hoop_pos, hoop_conf = find_rim(frame)
        platform_pos, _ = find_platform(frame)
        score_before = _capture_score_region(left, top, width, height)
        score_before_int = read_score(score_before) if score_before is not None else None
        if hoop_pos is None or platform_pos is None:
            print(f"  click {idx} @ ({click_x_rel},{click_y_rel}) — skipped (rim or platform not detected)")
            return
        hoop_x, hoop_y = hoop_pos
        platform_x, platform_y = platform_pos
        print(
            f"  click {idx}: hoop=({hoop_x},{hoop_y}) py={platform_y} "
            f"sb={score_before_int} — waiting {POST_SHOT_COOLDOWN}s for score…"
        )

        def post_shot():
            score_after = _capture_score_region(left, top, width, height)
            score_after_int = read_score(score_after) if score_after is not None else None
            inc = (
                score_after_int - score_before_int
                if score_before_int is not None and score_after_int is not None
                else None
            )
            made = inc is not None and 1 <= inc <= 2
            log_shot(
                shot_db,
                session_started=session_started,
                shot_idx=idx,
                fired_at=datetime.now().isoformat(timespec="seconds"),
                hoop_x=int(hoop_x),
                hoop_y=int(hoop_y),
                hoop_conf=float(hoop_conf),
                platform_x=int(platform_x),
                platform_y=int(platform_y),
                offset=int(platform_y - hoop_y),
                target_y=int(platform_y),
                eff_target_y=int(platform_y),
                clamped=0,
                direction="any",
                required_direction="any",
                made=int(made),
                score_before_int=score_before_int,
                score_after_int=score_after_int,
                score_increment=inc,
                code_commit=code_commit,
                predictor_kind=None,
                source="human",
            )
            label = f"MAKE +{inc}" if made else "miss"
            print(f"  click {idx} → {label} (sa={score_after_int}, inc={inc})")

        timer = threading.Timer(POST_SHOT_COOLDOWN, post_shot)
        pending_timers.append(timer)
        timer.start()

    return on_click


if __name__ == "__main__":
    run()
