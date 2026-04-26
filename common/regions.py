"""Per-minigame region storage in assets/regions.json.

Pickers write directly here instead of printing copy-paste values; main.py
loads from the file. Result: any region picked is immediately live without
the user editing source code.
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


def get_region(minigame_dir: Path, name: str) -> dict | None:
    return load_regions(minigame_dir).get(name)


def save_region(minigame_dir: Path, name: str, region: dict) -> Path:
    p = regions_path(minigame_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = load_regions(minigame_dir)
    data[name] = region
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p
