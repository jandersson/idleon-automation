import time
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay, check_failsafe, press_key
from common.monitor import make_shot_dir, save_frame, save_meta
from common.regions import get_region
from common.session_log import session_log
from common.window import get_bounds, WindowNotFoundError
from minigames.darts.detector import find_release_pose, find_game_over, score_region, score_changed

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"

WINDOW_TITLE = "Legends Of Idleon"
POLL_INTERVAL = 0.02

# Template-match confidence threshold for the release pose. The hand sweeps
# through other angles where the template matches weakly; threshold gates
# firing to the moments when it's in the captured release angle.
RELEASE_THRESHOLD = 0.7

# Mean Otsu-binarized pixel-diff threshold for "the score digits changed".
# Was 3.0 (the common.score_diff default); spot-check on 30 recent throws
# found two real hits at diff=2.25 (3→8) and diff=2.63 (5→6) being marked
# as misses. Real misses sit at exactly 0.0, so the floor for "something
# happened" is well below 1.0. Single-digit score region (40×17 px) means
# even a full digit change only flips a small fraction of pixels.
SCORE_CHANGE_THRESHOLD = 1.0

# Steam screenshot on Nine Dart Finish completion. When enabled, the bot
# presses F12 (Steam's default screenshot binding) after detecting 9
# consecutive hits — the requirement for the in-game Nine Dart Finish
# trophy. Limitation: counts any hit as a streak-keeper; if the trophy
# strictly requires *bullseyes* (and a non-bullseye hit resets the
# streak in-game), we'd false-fire on a 9-hit run that wasn't all
# bullseyes. Tighten to bullseye-only via score-magnitude or visual
# streak-counter detection if false fires happen.
STEAM_SCREENSHOT_ON_NINE_DART = False
NINE_DART_STREAK = 9

# Wait after throwing for: dart to land, score/animation to settle, new dart to
# load, and the player+platform to teleport to a new spawn position.
POST_THROW_COOLDOWN = 1.5

# Score region for make/miss diff. Calibrate via darts-pick-score-region (TODO)
# or set to None to skip score logging.
# Score and wind regions are loaded fresh from regions.json each call (using
# current window dims) so they survive the user resizing the game window.
# Pick via darts-pick-score-region / darts-pick-wind-region.
WIND_SAMPLES_DIR = Path(__file__).parent / "assets" / "wind_samples"
WIND_DEDUP_THRESHOLD = 5.0  # mean pixel diff above this = new wind state

# When enabled, every throw writes a per-throw subfolder under assets/monitor/
# with pre/post screenshots, the wind crop, and a metadata file. User can zip
# and share for offline review (since the bot can't watch the screen live).
MONITOR_MODE = True
MONITOR_DIR = Path(__file__).parent / "assets" / "monitor"
POST_LAND_DELAY = 0.6  # how long to wait after the cooldown before post-screenshot

# Capture flight frames during the post-throw cooldown so we can later
# extract dart-landing position relative to the bullseye and study how
# wind state correlates with landing offset. Heavyweight (~50KB/frame
# × ~30 frames per throw) but the data is what unlocks any future
# release-angle predictor. Mirrors hoops' MONITOR_FLIGHT pattern.
MONITOR_FLIGHT = True
FLIGHT_POLL = 0.05

# If no release pose has matched in this many seconds, assume the game-over
# screen has replaced the dartboard scene (the entire scene is replaced at
# game over, so the player avatar disappears and the template can't match).
# 10s was too tight — false-fired between normal throws after a 7-throw run
# while the game was still active. 25s gives plenty of headroom for slow
# cycles (wind change, score animations) while still terminating quickly
# when the dartboard is actually gone.
GAME_OVER_NO_POSE_SEC = 25.0


def _crop_wind(frame_bgra) -> np.ndarray | None:
    bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    h_img, w_img = bgr.shape[:2]
    region = get_region(_HERE, "wind", w_img, h_img)
    if region is None:
        return None
    x0 = max(0, region["left"])
    y0 = max(0, region["top"])
    x1 = min(w_img, region["left"] + region["width"])
    y1 = min(h_img, region["top"] + region["height"])
    return bgr[y0:y1, x0:x1]


def _maybe_save_wind_sample(wind_crop: np.ndarray, seen: list) -> bool:
    """Save wind_crop to wind_samples dir if it differs from every prior sample.

    Mutates `seen` in place. Returns True if saved.
    """
    if wind_crop is None or wind_crop.size == 0:
        return False
    for ref in seen:
        if ref.shape != wind_crop.shape:
            continue
        diff = float(cv2.absdiff(wind_crop, ref).astype(np.float32).mean())
        if diff < WIND_DEDUP_THRESHOLD:
            return False
    WIND_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S")
    cv2.imwrite(str(WIND_SAMPLES_DIR / f"sample_{stamp}.png"), wind_crop)
    seen.append(wind_crop)
    return True


def _load_existing_wind_samples() -> list:
    if not WIND_SAMPLES_DIR.exists():
        return []
    samples = []
    for p in sorted(WIND_SAMPLES_DIR.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            samples.append(img)
    return samples


def _save_monitor_throw(
    throw_idx: int,
    pre_frame_bgra: np.ndarray,
    pose: tuple[int, int],
    conf: float,
    wind_crop: np.ndarray | None,
    post_frame_bgra: np.ndarray,
    score_before,
    score_after,
    score_diff: float | None,
    score_changed_flag: bool | None,
    sub: Path | None = None,
) -> Path:
    # When the caller has already allocated the throw folder (e.g. to
    # write flight frames during the cooldown), reuse it instead of
    # creating a new one — keeps everything for one throw in one place.
    if sub is None:
        sub = make_shot_dir(MONITOR_DIR, throw_idx, prefix="throw")
    save_frame(sub / "pre_throw.png", pre_frame_bgra)
    save_frame(sub / "post_throw.png", post_frame_bgra)
    if wind_crop is not None and wind_crop.size > 0:
        cv2.imwrite(str(sub / "wind.png"), wind_crop)
    save_meta(
        sub / "meta.txt",
        timestamp=datetime.now().isoformat(),
        release_pose=f"({pose[0]},{pose[1]})",
        release_conf=f"{conf:.3f}",
        score_diff=score_diff if score_diff is not None else "n/a",
        score_changed=score_changed_flag if score_changed_flag is not None else "n/a",
    )
    return sub


def _capture_score(left: int, top: int, width: int, height: int):
    region = get_region(_HERE, "score", width, height)
    if region is None:
        return None
    frame = grab_region(left, top, width, height)
    return score_region(
        frame,
        region["left"],
        region["top"],
        region["width"],
        region["height"],
    )


def _log_shot_result(stats: dict, before, after) -> None:
    if before is None or after is None:
        return
    changed, diff = score_changed(before, after, threshold=SCORE_CHANGE_THRESHOLD)
    stats["attempts"] += 1
    if changed:
        stats["makes"] += 1
        stats["streak"] = stats.get("streak", 0) + 1
        streak = stats["streak"]
        suffix = f" | streak {streak}" if streak >= 2 else ""
        print(f"  [score] HIT (diff={diff:.1f}) | session {stats['makes']}/{stats['attempts']}{suffix}")
        if STEAM_SCREENSHOT_ON_NINE_DART and streak == NINE_DART_STREAK:
            print(f"  [nine-dart] {NINE_DART_STREAK}-hit streak reached — pressing F12 for Steam screenshot")
            press_key("f12")
            # Don't reset — Steam dedupes its own screenshot key, but we
            # also don't want to spam if the streak continues past 9.
            # Reset the counter so the next screenshot only fires after
            # another full streak.
            stats["streak"] = 0
    else:
        if stats.get("streak", 0) > 0:
            print(f"  [streak] reset (was {stats['streak']})")
        stats["streak"] = 0
        print(f"  [score] miss (diff={diff:.1f}) | session {stats['makes']}/{stats['attempts']}")


def run():
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        _run_inner()


def _run_inner():
    print(f"Darts bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    shot_stats: dict = {"makes": 0, "attempts": 0}
    throws_taken = 0  # increments every throw, independent of score detection
    best_recent_conf = 0.0  # for visibility into how close the matcher is getting between shots
    wind_seen = _load_existing_wind_samples()
    if wind_seen:
        print(f"Loaded {len(wind_seen)} existing wind samples; will only save new states.")
    last_pose_time = time.time()
    last_wind_crop: np.ndarray | None = None

    while True:
        check_failsafe()
        try:
            left, top, width, height = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        frame = grab_region(left, top, width, height)

        # Fast template check for the game-over screen first. Returns
        # (False, 0.0) if the template hasn't been captured yet — falls
        # through to the no-pose timeout heuristic below.
        is_over, go_conf = find_game_over(frame)
        if is_over:
            print(f"Game over detected (conf={go_conf:.2f}). Final session: "
                  f"{shot_stats['makes']}/{shot_stats['attempts']} hits.")
            return

        pose, conf = find_release_pose(frame, threshold=RELEASE_THRESHOLD)

        if pose is None:
            best_recent_conf = max(best_recent_conf, conf)
            # Game-over signal: the entire dartboard scene is replaced when
            # the trial ends, so the player avatar disappears and the release
            # template can't match. If we haven't seen the player in a while,
            # bail.
            if time.time() - last_pose_time > GAME_OVER_NO_POSE_SEC:
                print(f"No release pose detected for {GAME_OVER_NO_POSE_SEC:.0f}s — "
                      f"assuming game over. Final session: "
                      f"{shot_stats['makes']}/{shot_stats['attempts']} makes.")
                return
            time.sleep(POLL_INTERVAL)
            continue

        last_pose_time = time.time()
        px, py = pose
        print(f"Release pose at ({px},{py}), conf={conf:.2f} (recent best while waiting={best_recent_conf:.2f}) — throwing")
        # Fire immediately. Bookkeeping (score capture, wind crop +
        # diff + sample save) runs after the click — every ms between
        # pose detection and click landing drifts the arm a few degrees
        # off the captured release angle. Same principle as hoops; see
        # CLAUDE.md "Click timing".
        click(left + width // 2, top + height // 2)
        # Snapshot wind state from the pre-click frame (still valid —
        # `frame` is the BGRA buffer captured before the pose match).
        wind_crop = _crop_wind(frame)
        # Capture the score region post-click: the in-game score only
        # updates after the dart lands, so the region still shows the
        # pre-throw value here.
        score_before = _capture_score(left, top, width, height)
        # Log wind change between throws (vs the previous throw's reading,
        # not vs the saved library — that one only fires on never-seen-before
        # samples). Useful for correlating bullseye-or-not with wind shifts.
        if wind_crop is not None and last_wind_crop is not None \
                and wind_crop.shape == last_wind_crop.shape:
            wind_diff = float(cv2.absdiff(wind_crop, last_wind_crop).astype(np.float32).mean())
            if wind_diff >= WIND_DEDUP_THRESHOLD:
                print(f"  [wind] changed since last throw (diff={wind_diff:.1f})")
        if wind_crop is not None:
            last_wind_crop = wind_crop
        if _maybe_save_wind_sample(wind_crop, wind_seen):
            print(f"  [wind] new wind state saved (total samples: {len(wind_seen)})")
        # Per-throw monitor folder is allocated up-front so flight frames
        # have a destination. throws_taken hasn't been incremented yet —
        # do it now so the folder index matches the meta written below.
        throws_taken += 1
        flight_dir: Path | None = None
        if MONITOR_MODE:
            flight_dir = make_shot_dir(MONITOR_DIR, throws_taken, prefix="throw")
        # Sample flight frames during the cooldown instead of just sleeping.
        # Frames go to the throw folder so a later trajectory module can
        # extract dart-landing-x without a second live session.
        if MONITOR_FLIGHT and flight_dir is not None:
            flight_deadline = time.time() + POST_THROW_COOLDOWN + POST_LAND_DELAY
            flight_idx = 0
            while time.time() < flight_deadline:
                check_failsafe()
                flight_idx += 1
                f = grab_region(left, top, width, height)
                bgr = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
                cv2.imwrite(str(flight_dir / f"flight_{flight_idx:03d}.png"), bgr)
                time.sleep(FLIGHT_POLL)
        else:
            time.sleep(POST_THROW_COOLDOWN)
            time.sleep(POST_LAND_DELAY)
        post_frame = grab_region(left, top, width, height)
        score_after = _capture_score(left, top, width, height)
        # Compute the diff once so we can log AND save it to meta.
        diff_val = None
        diff_changed = None
        if score_before is not None and score_after is not None:
            diff_changed, diff_val = score_changed(score_before, score_after, threshold=SCORE_CHANGE_THRESHOLD)
        _log_shot_result(shot_stats, score_before, score_after)
        if MONITOR_MODE:
            sub = _save_monitor_throw(
                throw_idx=throws_taken,
                pre_frame_bgra=frame,
                pose=(px, py),
                conf=conf,
                wind_crop=wind_crop,
                post_frame_bgra=post_frame,
                score_before=score_before,
                score_after=score_after,
                score_diff=diff_val,
                score_changed_flag=diff_changed,
                sub=flight_dir,
            )
            print(f"  [monitor] saved {sub.name}")
        best_recent_conf = 0.0


if __name__ == "__main__":
    run()
