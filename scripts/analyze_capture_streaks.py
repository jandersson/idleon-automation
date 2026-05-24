"""Analyze a darts captures burst for template-match streak distribution.

Runs `find_release_pose` over every capture frame in
`minigames/darts/assets/captures/` (or a specified subdir/timestamp prefix),
counts consecutive frames above threshold, and reports the streak length
distribution. Used to test whether the top-of-swing apex match is sustained
at the capture rate — if all streaks are length 1, the discriminator isn't
match duration at that resolution.

Usage:
  uv run python scripts/analyze_capture_streaks.py
  uv run python scripts/analyze_capture_streaks.py --prefix capture_20260524_140000_
  uv run python scripts/analyze_capture_streaks.py --threshold 0.65
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from minigames.darts.detector import find_release_pose

CAPTURES_DIR = ROOT / "minigames" / "darts" / "assets" / "captures"


def _arg(name: str, default):
    if name in sys.argv:
        i = sys.argv.index(name) + 1
        if i < len(sys.argv):
            return type(default)(sys.argv[i])
    return default


def main() -> None:
    prefix = _arg("--prefix", "")
    threshold = _arg("--threshold", 0.7)

    pattern = f"{prefix}*.png" if prefix else "capture_*.png"
    frames = sorted(CAPTURES_DIR.glob(pattern))
    if not frames:
        print(f"No frames matching {pattern} in {CAPTURES_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"Analyzing {len(frames)} frames with threshold {threshold}")

    results: list[tuple[int, float, int | None, int | None, bool]] = []
    for i, p in enumerate(frames):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        pose, conf = find_release_pose(bgra, threshold=0.0)
        matched = conf >= threshold
        # find_release_pose returns None below its internal threshold (0.6)
        # — fall back to running match_multiscale_center directly so we
        # always have a dart-y for trajectory analysis.
        if pose is not None:
            x, y = pose
        else:
            from common.templates import match_multiscale_center
            template = cv2.imread(
                str(ROOT / "minigames" / "darts" / "assets" / "release.png"),
                cv2.IMREAD_COLOR,
            )
            (x, y), _, _ = match_multiscale_center(bgr, template)
        results.append((i, conf, int(x), int(y), matched))

    print()
    print("frame#  conf   dart=(x,y)  matched")
    for i, conf, x, y, matched in results:
        marker = "MATCH" if matched else ""
        print(f"  {i:>4}  {conf:.2f}   ({x},{y})    {marker}")

    streaks: list[int] = []
    cur = 0
    for r in results:
        m = r[4]
        if m:
            cur += 1
        elif cur > 0:
            streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)

    # Per-streak y trajectory: which direction was the dart moving in the
    # few frames leading up to (and during) the match? Down-swing = y
    # increasing (arm descending on screen) -> hit. Up-swing = y decreasing
    # -> miss. Sample 3 frames before streak start to capture pre-match
    # motion direction.
    print()
    print("Streak y-trajectories (3 frames pre + during streak):")
    streak_idx = 0
    cur_start: int | None = None
    for j, r in enumerate(results):
        m = r[4]
        if m and cur_start is None:
            cur_start = j
        elif not m and cur_start is not None:
            streak_idx += 1
            lead_start = max(0, cur_start - 3)
            ys = [results[k][3] for k in range(lead_start, j)]
            confs = [results[k][1] for k in range(lead_start, j)]
            direction = (
                "DESCENDING (down-swing -> hit?)" if ys[-1] > ys[0]
                else "ASCENDING (up-swing -> miss?)" if ys[-1] < ys[0]
                else "FLAT"
            )
            print(f"  Streak {streak_idx} (frames {cur_start}-{j-1}):")
            print(f"    y trajectory: {ys}  -> {direction}")
            print(f"    conf: {[f'{c:.2f}' for c in confs]}")
            cur_start = None

    print()
    print(f"Match streaks ({len(streaks)} total): {streaks}")
    if streaks:
        from collections import Counter
        dist = Counter(streaks)
        print("Streak-length distribution:")
        for length in sorted(dist):
            print(f"  len={length}: {dist[length]} streak(s)")
    if all(s == 1 for s in streaks):
        print()
        print("All streaks are length 1 at this threshold/interval. The")
        print("apex-vs-forward-release discriminator is NOT match duration at")
        print("this sampling rate. Try a finer interval or look for a different")
        print("signal (template tightness, dart rotation, etc.).")
    else:
        print()
        print("Mixed streak lengths! Apex matches might be the longer streaks.")
        print("Worth re-investigating the wait-and-verify approach with a")
        print("longer wait that catches the apex sustain.")


if __name__ == "__main__":
    main()
