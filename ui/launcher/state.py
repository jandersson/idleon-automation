"""Persisted launcher UI state (bot option selections, log preferences).

Selections like chopping's Aim=plan/gate or Save frames=on/off used to
reset to defaults on every launcher start; anyone A/B-testing an option
had to re-select it each session. This keeps them in a tiny JSON under
<repo>/data/ (gitignored — per-user state, same convention as
global_tries.json). Load once at startup, save on every change (the file
is tiny; eager writes mean a crash can't lose a selection).
"""
from __future__ import annotations

import json

from ui.launcher.config import PROJECT_ROOT

STATE_PATH = PROJECT_ROOT / "data" / "launcher_state.json"


def load() -> dict:
    """Return the persisted state dict ({} when missing/corrupt)."""
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save(state: dict) -> None:
    """Persist the state dict. Best-effort — never raises into the UI."""
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass


def get_option(state: dict, game: str, env: str, default: str) -> str:
    return state.get("bot_options", {}).get(f"{game}:{env}", default)


def set_option(state: dict, game: str, env: str, value: str) -> None:
    state.setdefault("bot_options", {})[f"{game}:{env}"] = value
    save(state)
