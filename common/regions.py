"""Per-minigame region storage in assets/regions.json.

Regions are stored as fractions of the game window's width and height (instead
of absolute pixels), so they survive the user resizing the window. Pickers
write the fractions; readers convert back to pixels at runtime using the
window's current dimensions.

JSON shape:
    {
      "score": {"left_frac": 0.04, "top_frac": 0.27, "width_frac": 0.06, "height_frac": 0.013},
      "wind":  {"left_frac": 0.56, "top_frac": 0.25, "width_frac": 0.10, "height_frac": 0.064}
    }
"""
import json
from pathlib import Path


def regions_path(minigame_dir: Path) -> Path:
    return minigame_dir / "assets" / "regions.json"


def load_regions(minigame_dir: Path) -> dict:
    p = regions_path(minigame_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def get_region(minigame_dir: Path, name: str, win_w: int, win_h: int) -> dict | None:
    """Return region as pixel coords {left, top, width, height} for the given
    window size, or None if not picked yet.

    Supports both the legacy pixel format ({"left", "top", ...}) and the new
    fraction format ({"left_frac", ...}). Legacy is treated as absolute and
    returned as-is — fine for backward compatibility but won't survive resize.
    """
    raw = load_regions(minigame_dir).get(name)
    if raw is None:
        return None
    if "left_frac" in raw:
        return {
            "left": int(raw["left_frac"] * win_w),
            "top": int(raw["top_frac"] * win_h),
            "width": int(raw["width_frac"] * win_w),
            "height": int(raw["height_frac"] * win_h),
        }
    return raw  # legacy pixel format


def save_region(minigame_dir: Path, name: str, region_px: dict, win_w: int, win_h: int) -> Path:
    """Save a pixel region as fractions of the given window size."""
    p = regions_path(minigame_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = load_regions(minigame_dir)
    data[name] = {
        "left_frac": region_px["left"] / win_w,
        "top_frac": region_px["top"] / win_h,
        "width_frac": region_px["width"] / win_w,
        "height_frac": region_px["height"] / win_h,
    }
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p
