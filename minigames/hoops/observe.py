"""Watch the user play hoops and log each manual shot to shots.db.

The bot fires no clicks here — pynput's mouse listener catches the
user's clicks on the game window, captures the rim + platform state at
that moment, then 4 seconds later checks the score region for a make.
Rows are logged with `source="human"` so future predictor fits can
distinguish bot-derived data from observed-human data.

Useful for seeding the predictor with makes in regions the bot can't
reach yet (transition zones between training-data clusters).
"""
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyautogui
import pynput

from common.capture import grab_region
from common.input import check_failsafe
from common.regions import get_region
from common.score_diff import score_region
from common.score_ocr import read_score
from common.session_log import session_log
from common.shot_log import open_db, log_shot, current_code_commit
from common.window import get_bounds, WindowNotFoundError
from minigames.hoops.detector import find_rim, find_platform
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
        try:
            _observe_inner(shot_db, session_started, code_commit)
        finally:
            shot_db.close()


def _observe_inner(shot_db, session_started: str, code_commit: str | None):
    print(
        f"Hoops OBSERVE mode — tracking window {WINDOW_TITLE!r}.\n"
        "  Click on the game window to play; each click is recorded.\n"
        "  Move mouse to a screen corner to abort.\n"
    )
    time.sleep(2)

    shot_idx = [0]  # mutable counter for the closure

    def on_click(screen_x: int, screen_y: int, button, pressed: bool) -> None:
        # Only react to LEFT-button presses (not releases, not other buttons).
        if not pressed or button != pynput.mouse.Button.left:
            return
        # Locate the game window now so we can ignore clicks outside it.
        try:
            left, top, width, height = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError:
            return
        if not (left <= screen_x <= left + width and top <= screen_y <= top + height):
            return  # click outside the game window — ignore

        # Snapshot pre-shot state synchronously so platform_y is fresh.
        click_x_rel = screen_x - left
        click_y_rel = screen_y - top
        frame = grab_region(left, top, width, height)
        hoop_pos, hoop_conf = find_rim(frame)
        platform_pos, _ = find_platform(frame)
        score_before = _capture_score_region(left, top, width, height)
        score_before_int = read_score(score_before) if score_before is not None else None

        if hoop_pos is None or platform_pos is None:
            print(f"  click @ ({click_x_rel},{click_y_rel}) — skipped (rim or platform not detected)")
            return

        shot_idx[0] += 1
        idx = shot_idx[0]
        hoop_x, hoop_y = hoop_pos
        platform_x, platform_y = platform_pos
        print(
            f"  click {idx}: hoop=({hoop_x},{hoop_y}) py={platform_y} "
            f"sb={score_before_int}  — waiting {POST_SHOT_COOLDOWN}s for score…"
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
                target_y=int(platform_y),  # at "fire time" platform was here
                eff_target_y=int(platform_y),
                clamped=0,
                direction="any",  # we don't track upstroke/downstroke for human clicks
                required_direction="any",
                made=int(made),
                score_before_int=score_before_int,
                score_after_int=score_after_int,
                score_increment=inc,
                code_commit=code_commit,
                predictor_kind=None,  # human shot, no predictor used
                source="human",
            )
            label = f"MAKE +{inc}" if made else "miss"
            print(
                f"  click {idx} → {label} "
                f"(sa={score_after_int}, inc={inc})"
            )

        threading.Timer(POST_SHOT_COOLDOWN, post_shot).start()

    listener = pynput.mouse.Listener(on_click=on_click)
    listener.start()
    try:
        while True:
            check_failsafe()
            time.sleep(0.1)
    except pyautogui.FailSafeException:
        print("Fail-safe triggered — exiting observe mode.")
    except KeyboardInterrupt:
        print("Interrupted — exiting observe mode.")
    finally:
        listener.stop()
        listener.join(timeout=1)
        print(f"Observe session ended. Captured {shot_idx[0]} clicks.")


if __name__ == "__main__":
    run()
