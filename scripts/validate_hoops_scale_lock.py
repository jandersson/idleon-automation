"""One-off: validate the scale-locked find_rim / find_platform against the
full multi-scale sweep on recorded hoops frames (mirror of
scripts/validate_pose_scale_lock.py for darts).

Platform center is THE aim signal (py drives the crossing detector), so
the bar is center agreement on accepted matches, not just confidence.
Simulates sequential polls (frames in within-shot temporal order)
through stateful matchers and compares every poll to the full sweep.

VERDICT (2026-06-10, 1207-frame run): platform 1199/1207 bit-identical
(the 8 misses are cross-window-size artifacts of the simulation — the
stream interleaves sessions captured at different window sizes, which
live polls never do). Rim REJECTED for scale-locking: the recurring
"(691,550) conf 0.45-0.55" disagreements turned out to be genuine
extreme-low-spawn rims, clipped by the window bottom edge and matching
at scale 0.6 while native scales read ~0.25 — the documented case the
0.45 accept threshold exists for. find_rim keeps the full sweep; it
isn't on the steady-state tick.

Usage: python scripts/validate_hoops_scale_lock.py [max_frames]
"""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.templates import ScaleLockMatcher, match_multiscale_center  # noqa: E402

ROOT = Path(__file__).parent.parent
ASSETS = ROOT / "minigames" / "hoops" / "assets"

max_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 1200


def _shot_order(p: Path) -> tuple:
    rank = {"pre_shot": 0}.get(p.stem, 1 if p.stem.startswith("flight") else 2)
    return (str(p.parent), rank, p.name)


frames = sorted((ASSETS / "monitor").rglob("*.png"), key=_shot_order)
stride = max(1, len(frames) // max_frames)
frames = frames[::stride]
print(f"{len(frames)} monitor frames (stride {stride}), simulated as one poll stream")

CASES = {
    "rim": {
        "template": cv2.imread(str(ASSETS / "rim.png")),
        "region": lambda w, h: (w // 2, 0, w, h),
        "accept": 0.45,
        "matcher": ScaleLockMatcher(),
    },
    "platform": {
        "template": cv2.imread(str(ASSETS / "platform.png")),
        "region": lambda w, h: (0, 0, w // 2, h),
        "accept": 0.5,
        "matcher": ScaleLockMatcher(),
    },
}

stats = {k: {"n": 0, "exact": 0, "deltas": [], "bad": []} for k in CASES}
t_old = t_new = 0.0

for i, p in enumerate(frames):
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    for name, case in CASES.items():
        region = case["region"](w, h)
        t0 = time.perf_counter()
        old_c, old_v, _ = match_multiscale_center(img, case["template"], region=region)
        t1 = time.perf_counter()
        new_c, new_v, _ = case["matcher"].match_center(img, case["template"], region=region)
        t2 = time.perf_counter()
        t_old += t1 - t0
        t_new += t2 - t1
        s = stats[name]
        s["n"] += 1
        s["deltas"].append(new_v - old_v)
        if abs(new_v - old_v) < 1e-6 and old_c == new_c:
            s["exact"] += 1
        elif old_v >= case["accept"] and (new_c != old_c or new_v < case["accept"]):
            # An accepted match whose center moved (aim shifts) or that the
            # locked poll would now reject (detection lost) — the failures
            # that matter.
            s["bad"].append((p, old_v, new_v, old_c, new_c))
    if (i + 1) % 300 == 0:
        print(f"  {i + 1}/{len(frames)}...")

n_total = sum(s["n"] for s in stats.values())
print(f"\ntiming (rim+platform per frame): full sweep "
      f"{t_old / (n_total / 2) * 1000:.1f} ms, scale-locked "
      f"{t_new / (n_total / 2) * 1000:.1f} ms ({t_old / t_new:.1f}x)")
for name, s in stats.items():
    d = sorted(s["deltas"])
    n = s["n"]
    print(f"\n{name}: exact agreement {s['exact']}/{n}, "
          f"conf delta min {d[0]:+.4f} / p1 {d[n // 100]:+.4f} / median {d[n // 2]:+.4f}")
    print(f"  accepted-match center moves or losses: {len(s['bad'])}")
    for p, ov, nv, oc, nc in s["bad"][:15]:
        print(f"    {p.parent.name}/{p.name}: conf {ov:.4f} -> {nv:.4f}, center {oc} -> {nc}")
