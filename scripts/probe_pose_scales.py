"""One-off: which template scale wins find_release_pose matches, and how
stable is it? Informs the scale-cache design for the poll-loop speedup —
if genuine (high-conf) matches always win at one or two adjacent scales
within a session, most polls can skip the other five.
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
stride = max(1, len(frames) // 600)
frames = frames[::stride]

by_band: dict[str, Counter] = {"genuine (>=0.65)": Counter(), "mid (0.55-0.65)": Counter(), "junk (<0.55)": Counter()}
sizes = Counter()
for p in frames:
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    sizes[(w, h)] += 1
    center, val, scale = match_multiscale_center(img, template, region=(0, 0, int(w * 0.65), h))
    band = "genuine (>=0.65)" if val >= 0.65 else ("mid (0.55-0.65)" if val >= 0.55 else "junk (<0.55)")
    by_band[band][scale] += 1

print(f"frames: {sum(sizes.values())}, window sizes: {dict(sizes)}")
for band, counter in by_band.items():
    total = sum(counter.values())
    dist = ", ".join(f"{s}: {c}" for s, c in sorted(counter.items()))
    print(f"{band:<18} n={total}  {dist}")
