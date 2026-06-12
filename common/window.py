"""Window lookup, platform-dispatched.

The only API bots use is get_bounds(title_substring) -> (left, top,
width, height) in screen coordinates. On Windows that's pygetwindow; on
macOS it's a Quartz window-list query (pygetwindow's macOS backend is
mostly NotImplementedError, so its import lives inside the Windows
branch — keeping module import safe everywhere).

macOS coordinate note: kCGWindowBounds is in points with a top-left
origin in global coords — the same space pyautogui and mss region
requests use on macOS, so the window-relative convention carries over
unchanged (capture normalization in common/capture.py handles Retina
pixel density).
"""
import sys


class WindowNotFoundError(RuntimeError):
    pass


def get_bounds(title_substring: str = "Idleon") -> tuple[int, int, int, int]:
    """Return (left, top, width, height) of the first visible window whose title contains the substring."""
    if sys.platform == "darwin":
        return _get_bounds_quartz(title_substring)
    return _get_bounds_pygetwindow(title_substring)


def _get_bounds_pygetwindow(title_substring: str) -> tuple[int, int, int, int]:
    import pygetwindow as gw

    for w in gw.getAllWindows():
        if title_substring.lower() in w.title.lower() and w.visible and w.width > 0 and w.height > 0:
            return w.left, w.top, w.width, w.height
    raise WindowNotFoundError(f"No visible window found with title containing {title_substring!r}")


def _get_bounds_quartz(title_substring: str) -> tuple[int, int, int, int]:
    import Quartz

    needle = title_substring.lower()
    # Until Screen Recording permission is granted, kCGWindowName comes
    # back empty — so also match the owner (process) name, which is
    # always visible. The Idleon process name is likely "LegendsOfIdleon"
    # (no spaces), so compare space-stripped too.
    needle_tight = needle.replace(" ", "")
    window_list = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    for info in window_list or []:
        if info.get(Quartz.kCGWindowLayer, 0) != 0:  # 0 = normal app windows
            continue
        name = (info.get(Quartz.kCGWindowName) or "").lower()
        owner = (info.get(Quartz.kCGWindowOwnerName) or "").lower()
        if needle not in name and needle_tight not in owner.replace(" ", ""):
            continue
        bounds = info.get(Quartz.kCGWindowBounds) or {}
        left, top = int(bounds.get("X", 0)), int(bounds.get("Y", 0))
        width, height = int(bounds.get("Width", 0)), int(bounds.get("Height", 0))
        if width > 0 and height > 0:
            return left, top, width, height
    raise WindowNotFoundError(f"No visible window found with title containing {title_substring!r}")
