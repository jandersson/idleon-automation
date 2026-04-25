import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay
from common.window import get_bounds, WindowNotFoundError
from minigames.darts.detector import find_release_pose, score_region, score_changed

WINDOW_TITLE = "Idleon"
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
SCORE_REGION_REL: dict | None = None

# Wind indicator region (set via darts-pick-wind-region). Crop tight around
# just the wind value/arrow, not the "Wind:" label, so we're sensitive to the
# state, not the static label.
WIND_REGION_REL: dict | None = {"left": 747, "top": 357, "width": 57, "height": 51}


def _capture_score(left: int, top: int, width: int, height: int):
    if SCORE_REGION_REL is None:
        return None
    frame = grab_region(left, top, width, height)
    return score_region(
        frame,
        SCORE_REGION_REL["left"],
        SCORE_REGION_REL["top"],
        SCORE_REGION_REL["width"],
        SCORE_REGION_REL["height"],
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
    print(f"Darts bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    shot_stats: dict = {"makes": 0, "attempts": 0}
    best_recent_conf = 0.0  # for visibility into how close the matcher is getting between shots

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
        random_delay(20, 60)
        click(left + width // 2, top + height // 2)
        time.sleep(POST_THROW_COOLDOWN)
        score_after = _capture_score(left, top, width, height)
        _log_shot_result(shot_stats, score_before, score_after)
        best_recent_conf = 0.0


if __name__ == "__main__":
    run()
