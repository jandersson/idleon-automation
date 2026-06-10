"""One-off: validate ScaleLockMatcher against the full multi-scale sweep
on recorded darts frames before switching find_release_pose over.

Simulates sequential polls (frames in within-throw temporal order:
pre_throw, flight_*, post_throw) through a stateful matcher configured
the way find_release_pose will use it, and compares every poll's
(confidence, center) to the full 7-scale sweep. The matcher returns
exact full-resolution confidences over the scales it tries, so any
disagreement comes from the winning scale falling outside the locked
neighborhood — the thing to measure, especially on fire-relevant polls
(full-sweep conf >= MIN_RELEASE_THRESHOLD).

A prior coarse-to-fine candidate (half-res localization + full-res
crops) was validated here first and rejected 2026-06-10: with the tiny
9x40 release template, only 66/449 frames agreed exactly and the median
confidence under-report was -0.03 for a 2.9x speedup. The scale-lock
approach replaced it.

Usage: python scripts/validate_pose_coarse_fine.py [max_frames]
"""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.templates import ScaleLockMatcher, match_multiscale_center  # noqa: E402

ROOT = Path(__file__).parent.parent
ASSETS = ROOT / "minigames" / "darts" / "assets"
MIN_RELEASE_THRESHOLD = 0.55  # mirror of minigames/darts/main.py

max_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def _throw_order(p: Path) -> tuple:
    rank = {"pre_throw": 0, "post_throw": 2}.get(p.stem, 1)
    return (str(p.parent), rank, p.name)


frames = sorted((ASSETS / "monitor").rglob("*.png"), key=_throw_order)
stride = max(1, len(frames) // max_frames)
frames = frames[::stride]
print(f"{len(frames)} monitor frames (stride {stride}), simulated as one poll stream")

template = cv2.imread(str(ASSETS / "release.png"))
assert template is not None

matcher = ScaleLockMatcher(always_scales=(0.6,))

n = 0
deltas = []
disagreements = []  # (path, old_conf, new_conf, old_center, new_center)
fire_relevant = []
t_old = t_new = 0.0

for p in frames:
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    region = (0, 0, int(w * 0.65), h)

    t0 = time.perf_counter()
    old_c, old_v, old_s = match_multiscale_center(img, template, region=region)
    t1 = time.perf_counter()
    new_c, new_v, new_s = matcher.match_center(img, template, region=region)
    t2 = time.perf_counter()
    t_old += t1 - t0
    t_new += t2 - t1
    n += 1

    deltas.append(new_v - old_v)
    if new_v < old_v - 0.005 or (old_c != new_c and old_v >= MIN_RELEASE_THRESHOLD):
        disagreements.append((p, old_v, new_v, old_c, new_c))
        if old_v >= MIN_RELEASE_THRESHOLD:
            fire_relevant.append((p, old_v, new_v, old_c, new_c))
    if n % 400 == 0:
        print(f"  {n}/{len(frames)}... (locked scale: {matcher.locked_scale})")

deltas.sort()
print(f"\nframes compared: {n}")
print(f"timing: full sweep {t_old / n * 1000:.1f} ms/frame, "
      f"scale-locked {t_new / n * 1000:.1f} ms/frame ({t_old / t_new:.1f}x)")
print(f"conf delta (new - old): min {deltas[0]:+.4f}  p1 {deltas[n // 100]:+.4f}  "
      f"median {deltas[n // 2]:+.4f}  max {deltas[-1]:+.4f}")
print(f"exact conf agreement (|delta| < 1e-6): "
      f"{sum(1 for d in deltas if abs(d) < 1e-6)}/{n}")
print(f"disagreements (conf dropped >0.005, or center moved on a fire-relevant poll): "
      f"{len(disagreements)}")
print(f"fire-relevant (old conf >= {MIN_RELEASE_THRESHOLD}): {len(fire_relevant)}")
for p, ov, nv, oc, nc in fire_relevant[:30]:
    print(f"  {p.parent.name}/{p.name}: conf {ov:.4f} -> {nv:.4f}, center {oc} -> {nc}")
