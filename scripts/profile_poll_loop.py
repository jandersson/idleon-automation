"""One-off: profile the darts poll-loop pipeline on a real frame."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from minigames.darts.detector import find_game_over, find_release_pose  # noqa: E402
from minigames.darts.arm_motion import compute_arm_centroid  # noqa: E402

root = Path(__file__).parent.parent
f1 = cv2.imread(str(root / "minigames/darts/assets/monitor/throw_029_202521/flight_001.png"))
f2 = cv2.imread(str(root / "minigames/darts/assets/monitor/throw_029_202521/flight_002.png"))
frame_bgra = cv2.cvtColor(f1, cv2.COLOR_BGR2BGRA)
print(f"frame: {f1.shape}")


def bench(name, fn, n=20):
    fn()  # warm
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"  {name:<28} {dt:7.1f} ms")
    return dt


total = 0.0
total += bench("cvtColor BGRA->BGR", lambda: cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR))
total += bench("find_game_over", lambda: find_game_over(frame_bgra))
total += bench("find_release_pose", lambda: find_release_pose(frame_bgra, threshold=0.0))
total += bench("compute_arm_centroid", lambda: compute_arm_centroid(f1, f2))
print(f"  {'CV subtotal':<28} {total:7.1f} ms")

# mss grab cost at the same window size
import mss  # noqa: E402

with mss.mss() as sct:
    region = {"left": 0, "top": 0, "width": f1.shape[1], "height": f1.shape[0]}

    def grab():
        return np.array(sct.grab(region))

    g = bench("mss grab (live screen)", grab)
print(f"  {'TOTAL per poll':<28} {total + g:7.1f} ms")
