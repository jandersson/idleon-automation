import mss
import numpy as np


def grab_region(left: int, top: int, width: int, height: int) -> np.ndarray:
    with mss.mss() as sct:
        region = {"left": left, "top": top, "width": width, "height": height}
        frame = sct.grab(region)
        return np.array(frame)


def grab_fullscreen() -> np.ndarray:
    with mss.mss() as sct:
        frame = sct.grab(sct.monitors[1])
        return np.array(frame)
