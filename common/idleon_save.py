"""Read the local Idleon save file from Steam's LevelDB cache.

Idleon stores its save in:
    %APPDATA%\\legends-of-idleon\\Local Storage\\leveldb

The interesting value is encoded in Haxe's custom serialization format
(see https://haxe.org/manual/std-serialization-format.html). We vendor a
minimal decoder here so we don't drag in idleon-saver's full GUI/CLI deps.

Usage:
    from common.idleon_save import load_save
    data = load_save()
    # `data` is a Python dict with the full game state.
"""
import os
import shutil
import tempfile
from typing import Any
from urllib.parse import unquote

# Soft import — the launcher reads this lazily; failing import means the
# user hasn't installed the wheel-friendly leveldb wrapper.
try:
    import plyvel
    _HAVE_PLYVEL = True
except ImportError:
    plyvel = None
    _HAVE_PLYVEL = False


SAVE_DIR = os.path.expandvars(r"%APPDATA%\legends-of-idleon\Local Storage\leveldb")
GAME_KEY_PREFIX = b"_file://\x00\x01/"  # values keyed by file:// URL of the game


# Constants from Haxe serialization format.
_CONSTANTS: dict[str, Any] = {
    "n": None,
    "z": 0,
    "k": float("nan"),
    "m": float("-inf"),
    "p": float("inf"),
    "t": True,
    "f": False,
}


def _decode_haxe(text: str) -> Any:
    """Parse one Haxe-serialized value out of `text` from index 0.

    Implements the subset of the format that Idleon emits:
      i<digits>           — int
      d<digits>           — float
      y<len>:<chars>      — string (URL-decoded; cached)
      R<int>              — string-cache reference
      o ... g             — dict (key/value pairs)
      a ... h , l ... h   — list
      single chars in _CONSTANTS (n/t/f/z/k/m/p)
    """
    pos = 0
    strcache: list[str] = []

    def peek() -> str:
        return text[pos]

    def take() -> str:
        nonlocal pos
        ch = text[pos]
        pos += 1
        return ch

    def read_int() -> int:
        nonlocal pos
        start = pos
        while pos < len(text) and text[pos] in "0123456789-":
            pos += 1
        return int(text[start:pos])

    def read_float() -> float:
        nonlocal pos
        start = pos
        while pos < len(text) and text[pos] in "0123456789.-+e":
            pos += 1
        return float(text[start:pos])

    def read_string() -> str:
        nonlocal pos
        # length, then ':', then characters (URL-encoded).
        n = read_int()
        assert text[pos] == ":", f"expected ':' at pos {pos}, got {text[pos]!r}"
        pos += 1
        s = unquote(text[pos:pos + n])
        pos += n
        strcache.append(s)
        return s

    def read_until(end_char: str, parse_one):
        out = []
        while peek() != end_char:
            out.append(parse_one())
        take()  # consume end_char
        return out

    def parse_one() -> Any:
        ch = take()
        if ch in _CONSTANTS:
            return _CONSTANTS[ch]
        if ch == "i":
            return read_int()
        if ch == "d":
            return read_float()
        if ch == "y":
            return read_string()
        if ch == "R":
            return strcache[read_int()]
        if ch == "o":
            # struct/object: pairs of key/value until 'g'.
            return dict(read_until("g", lambda: (parse_one(), parse_one())))
        if ch in ("b", "q", "M"):
            # StringMap, IntMap, ObjectMap: pairs of key/value until 'h'.
            return dict(read_until("h", lambda: (parse_one(), parse_one())))
        if ch in ("a", "l"):
            return read_until("h", parse_one)
        if ch == "u":
            # null-elements run in arrays: u<n> means n nulls in a row.
            return [None] * read_int()
        raise ValueError(f"Unknown tag {ch!r} at index {pos - 1}")

    return parse_one()


def load_save(save_dir: str = SAVE_DIR) -> dict | None:
    """Decode the largest game-state value out of Idleon's LevelDB cache.

    Copies the leveldb files to a temp dir first so we don't fight the
    game's LOCK file when Idleon is running. Returns None if plyvel isn't
    installed or no game-state key was found.
    """
    if not _HAVE_PLYVEL:
        return None
    if not os.path.exists(save_dir):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, "idleon_save")
        shutil.copytree(save_dir, dst, ignore=shutil.ignore_patterns("LOCK"))
        db = plyvel.DB(dst, create_if_missing=False)
        try:
            game_value: bytes | None = None
            for k, v in db:
                if k.startswith(GAME_KEY_PREFIX) and len(v) > 1000:
                    game_value = v
                    break
        finally:
            db.close()
    if game_value is None:
        return None
    # leveldb prepends a single tag byte (0x01 = JSON-ish blob in chrome's
    # Local Storage format); strip it.
    text = game_value[1:].decode("utf-8", errors="replace")
    return _decode_haxe(text)


def read_minigame_plays(save_dir: str = SAVE_DIR) -> dict[str, int] | None:
    """Read MinigamePlays per character from the save. Returns
    {character_name: plays_remaining} or None if the save can't be read.

    Chopping and catching share this counter in-game. Per-character
    (each character has its own quota that resets daily)."""
    data = load_save(save_dir)
    if data is None:
        return None
    out: dict[str, int] = {}
    for name, info in (data.get("PlayerDATABASE") or {}).items():
        plays = (info.get("PersonalValuesMap") or {}).get("MinigamePlays")
        if isinstance(plays, (int, float)):
            out[name] = int(plays)
    return out
