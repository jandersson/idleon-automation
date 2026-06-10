"""One-off: profile the hoops main-loop pipeline on a real frame.

The hoops findings doc records a 200-1000ms observed tick vs the
15-30ms the POLL_INTERVAL comment promises; this measures where it
goes, mirroring scripts/profile_poll_loop.py (darts).
"""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from minigames.hoops.detector import (  # noqa: E402
    find_game_over, find_game_prompt, find_platform, find_rim, _PLATFORM_MATCHER,
)

# Steady state: the platform matcher locks its scale within the first
# few ticks of a live session (platform conf ~0.8+), so bench the locked
# path — the benchmark frame would lock on its own here anyway, but be
# explicit about what's being measured.
_PLATFORM_MATCHER.locked_scale = 1.0

root = Path(__file__).parent.parent
frame_path = root / "minigames/hoops/assets/monitor/shot_001_001657/flight_001.png"
f1 = cv2.imread(str(frame_path))
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


t_cvt = bench("cvtColor BGRA->BGR", lambda: cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR))
t_go = bench("find_game_over", lambda: find_game_over(frame_bgra))
t_prompt = bench("find_game_prompt", lambda: find_game_prompt(frame_bgra))
t_plat = bench("find_platform", lambda: find_platform(frame_bgra))
t_rim = bench("find_rim", lambda: find_rim(frame_bgra))

import mss  # noqa: E402
import numpy as np  # noqa: E402

with mss.mss() as sct:
    region = {"left": 0, "top": 0, "width": f1.shape[1], "height": f1.shape[0]}

    def grab():
        return np.array(sct.grab(region))

    t_grab = bench("mss grab (live screen)", grab)

# Steady state (target set, waiting to fire): grab + platform, with
# game_over amortized over GAME_OVER_CHECK_EVERY ticks. find_rim runs
# between shots and on >=10 drift refresh; find_game_prompt only in the
# pre-game phase and post-click.
from minigames.hoops.main import GAME_OVER_CHECK_EVERY  # noqa: E402

steady = t_grab + t_plat + t_go / GAME_OVER_CHECK_EVERY
print(f"  {'steady-state tick':<28} {steady:7.1f} ms "
      f"(game_over /{GAME_OVER_CHECK_EVERY})")
print(f"  {'between-shots tick (+rim)':<28} {steady + t_rim:7.1f} ms")
