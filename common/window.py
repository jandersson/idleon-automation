import pygetwindow as gw


class WindowNotFoundError(RuntimeError):
    pass


def get_bounds(title_substring: str = "Idleon") -> tuple[int, int, int, int]:
    """Return (left, top, width, height) of the first visible window whose title contains the substring."""
    for w in gw.getAllWindows():
        if title_substring.lower() in w.title.lower() and w.visible and w.width > 0 and w.height > 0:
            return w.left, w.top, w.width, w.height
    raise WindowNotFoundError(f"No visible window found with title containing {title_substring!r}")
