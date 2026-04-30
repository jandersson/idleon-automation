import time
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay
from common.monitor import make_shot_dir, save_frame, save_meta
from common.regions import get_region
from common.session_log import session_log
from common.window import get_bounds, WindowNotFoundError
from minigames.darts.detector import find_release_pose, score_region, score_changed

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"

WINDOW_TITLE = "Legends Of Idleon"
POLL_INTERVAL = 0.02

# Template-match confidence threshold for the release pose. The hand sweeps
# through other angles where the template matches weakly; threshold gates
# firing to the moments when it's in the captured release angle.
RELEASE_THRESHOLD = 0.7

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
) -> Path:
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
    changed, diff = score_changed(before, after)
    stats["attempts"] += 1
    if changed:
        stats["makes"] += 1
        print(f"  [score] HIT (diff={diff:.1f}) | session {stats['makes']}/{stats['attempts']}")
    else:
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

    while True:
        try:
            left, top, width, height = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        frame = grab_region(left, top, width, height)
        pose, conf = find_release_pose(frame, threshold=RELEASE_THRESHOLD)

        if pose is None:
            best_recent_conf = max(best_recent_conf, conf)
            time.sleep(POLL_INTERVAL)
            continue

        px, py = pose
        print(f"Release pose at ({px},{py}), conf={conf:.2f} (recent best while waiting={best_recent_conf:.2f}) — throwing")
        score_before = _capture_score(left, top, width, height)
        # Snapshot wind state right before this throw — useful even now (so the
        # library accumulates) and later (so we can pick the right release angle).
        wind_crop = _crop_wind(frame)
        if _maybe_save_wind_sample(wind_crop, wind_seen):
            print(f"  [wind] new wind state saved (total samples: {len(wind_seen)})")
        random_delay(20, 60)
        click(left + width // 2, top + height // 2)
        time.sleep(POST_THROW_COOLDOWN)
        time.sleep(POST_LAND_DELAY)
        post_frame = grab_region(left, top, width, height)
        score_after = _capture_score(left, top, width, height)
        # Compute the diff once so we can log AND save it to meta.
        diff_val = None
        diff_changed = None
        if score_before is not None and score_after is not None:
            diff_changed, diff_val = score_changed(score_before, score_after)
        _log_shot_result(shot_stats, score_before, score_after)
        throws_taken += 1
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
            )
            print(f"  [monitor] saved {sub.name}")
        best_recent_conf = 0.0


if __name__ == "__main__":
    run()
