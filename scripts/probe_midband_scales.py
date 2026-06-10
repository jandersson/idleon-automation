"""One-off: in the 0.58-0.68 conf band, which matches are the genuine
player pose (center in the spawn area) vs static junk, and at which scale
does each win? The scale-lock design excludes far-off scales from most
polls, so genuine sub-threshold matches (the adaptive-threshold signal)
must win at the locked scale, not at the junk-dominated 0.6.
"""
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.templates import match_multiscale_center  # noqa: E402

ROOT = Path(__file__).parent.parent
ASSETS = ROOT / "minigames" / "darts" / "assets"

template = cv2.imread(str(ASSETS / "release.png"))
frames = sorted((ASSETS / "monitor").rglob("*.png"))
stride = max(1, len(frames) // 1200)
frames = frames[::stride]

# Player area on the 960x572 window: spawn x ~350-450 (template centers
# spread wider with arm swing), pose_y support ~(240, 404).
def in_player_area(c):
    return c is not None and 230 <= c[0] <= 560 and 180 <= c[1] <= 460

player = Counter()
junk = Counter()
junk_centers = Counter()
for p in frames:
    img = cv2.imread(str(p))
    if img is None or img.shape[1] != 960:
        continue
    h, w = img.shape[:2]
    center, val, scale = match_multiscale_center(img, template, region=(0, 0, int(w * 0.65), h))
    if not (0.58 <= val <= 0.68):
        continue
    if in_player_area(center):
        player[scale] += 1
    else:
        junk[scale] += 1
        junk_centers[center] += 1

print(f"player-area matches in band: n={sum(player.values())}  "
      + ", ".join(f"{s}: {c}" for s, c in sorted(player.items())))
print(f"junk matches in band:        n={sum(junk.values())}  "
      + ", ".join(f"{s}: {c}" for s, c in sorted(junk.items())))
print("top junk centers:", junk_centers.most_common(8))
