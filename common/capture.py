"""Screen capture with HiDPI normalization.

Repo-wide invariant: 1 captured pixel == 1 window coordinate unit ==
1 pyautogui click unit. On Retina/HiDPI displays mss returns physical
pixels (e.g. 2x the requested point size), which would corrupt
detected→click coordinates, every pixel-unit threshold, template
matching scale, and DB unit comparability across machines. So every
grab is resized back to the requested logical size at this one
chokepoint — downstream code never sees physical pixels. The scale is
derived per grab (not hardcoded 2.0) to handle external 1x monitors
and fractional scaling.
"""
import threading

import cv2
import mss
import numpy as np

# Reuse one mss instance per thread rather than `with mss.mss()` per grab —
# the documented-efficient pattern (avoids re-opening the display connection
# each call). mss isn't thread-safe and an instance is bound to its creating
# thread, so cache it thread-locally (outcome callbacks can grab from worker
# threads). NOTE: the grab() capture itself is the dominant cost (~13ms here,
# size-independent), so the real lever against capture-induced game lag is
# grab FREQUENCY/COUNT, not instance reuse.
_local = threading.local()


def _sct() -> "mss.base.MSSBase":
    sct = getattr(_local, "sct", None)
    if sct is None:
        sct = mss.mss()
        _local.sct = sct
    return sct


def _normalize_size(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a grabbed frame to the requested logical size if the OS
    returned it at a different (physical-pixel) size. No-op at 1x."""
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def grab_region(left: int, top: int, width: int, height: int) -> np.ndarray:
    region = {"left": left, "top": top, "width": width, "height": height}
    frame = _sct().grab(region)
    return _normalize_size(np.array(frame), width, height)


def grab_fullscreen() -> np.ndarray:
    sct = _sct()
    monitor = sct.monitors[1]
    frame = sct.grab(monitor)
    return _normalize_size(np.array(frame), monitor["width"], monitor["height"])
