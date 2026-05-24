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

    results: list[tuple[int, float, bool]] = []
    for i, p in enumerate(frames):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        _, conf = find_release_pose(bgra, threshold=0.0)
        matched = conf >= threshold
        results.append((i, conf, matched))

    print()
    print("frame#  conf   matched")
    for i, conf, matched in results:
        marker = "MATCH" if matched else ""
        print(f"  {i:>4}  {conf:.2f}   {marker}")

    streaks: list[int] = []
    cur = 0
    for _, _, m in results:
        if m:
            cur += 1
        elif cur > 0:
            streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)

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
