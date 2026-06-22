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


# Idleon's `OptionsListAccount` array stores per-account counters. These
# indices were identified empirically (2026-05-05) by diffing the save
# across a 30s wait — the live values shift in 5-tick-per-second units,
# i.e. 1 tick = 0.2s. The mapping matches cheat-table dumps that flag
# OLA[440] as the darts daily-play counter.
_OLA_DARTS_COOLDOWN_TICKS = 439  # negative = playable; positive = ticks of cooldown left
_OLA_DARTS_PLAYS_TODAY = 440     # 0..N, grows with each scored darts session
# OLA[442] was previously mislabelled here as a darts "cooldown base" (#22).
# The authoritative community index map (MrJoiny/Idleon-Injector
# optionsAccountSchema.json) identifies it as the darts minigame HIGH SCORE,
# and the data confirms it: it sits flat at 125 while the darts cooldown
# still escalates with plays_today. The minigame cooldown is COMPUTED from
# plays-today (see minigame_cooldown_ticks_for_plays), not read from any
# stored base — so OLA[442] is not used for the cooldown.
_OLA_TICKS_PER_SECOND = 5

# Accumulated Library book-checkout pool (the shared count you spend to
# generate Talent Books for any character). IdleonToolbox reads it from
# OptionsListAccount[55] (`timeAway.BookLib` / book count); confirmed
# against the local save (29). The checkout pool is account-wide, so the
# book recommender ranks candidates across all characters (#44).
_OLA_BOOK_CHECKOUTS = 55


def read_book_checkouts(save_dir: str = SAVE_DIR) -> int | None:
    """Accumulated Library book-checkout pool (OLA[55]) — the shared count
    spent to check out Talent Books. None if the save can't be read or the
    array is too short / non-numeric."""
    data = load_save(save_dir)
    if data is None:
        return None
    ola = data.get("OptionsListAccount")
    if not isinstance(ola, list) or len(ola) <= _OLA_BOOK_CHECKOUTS:
        return None
    n = ola[_OLA_BOOK_CHECKOUTS]
    if not isinstance(n, (int, float)):
        return None
    return int(n)


# Account max Talent Book level (#94). It isn't one save field — it's a base
# plus a sum of unlock sources (wiki: Talent Book Library, "Maximum level can
# be boosted by"). Base 125 = the top of the un-boosted checkout range
# (101-125); the sources stack up to the 396 ceiling.
#
# VERIFIED against the local save (computes 127): base + the World 3 merit.
#   World 3 Merit Shop #3 ("+10 Max possible Lv of Talent books") = +2 per
#   level, max 5 levels, stored at Tasks[2][2][2] — Tasks[2] is the merit
#   shop, [2] selects World 3, [2] the 3rd merit. Local save has 1 level
#   (+2) → 125 + 2 = 127.
#
# NOT YET SUMMED — all zero on the local save, so their array indices can't
# be verified here (see #95): Salt Lick "Spontaneity Salts" (+2/lvl, max 10),
# the W3 "Checkout Takeout" achievement (+5), Atom Collider "Oxygen - Library
# Booker" (+10), the Sailing "Fury Relic" artifact (+25 x6), and the Summoning
# winner bonus (+10, up to +75). Until those are wired, detection undercounts
# once they're unlocked — the Books tab keeps the value editable as a manual
# override/backstop.
_MAX_BOOK_BASE = 125
_W3_MERIT_BOOK_PER_LEVEL = 2
_W3_MERIT_BOOK_MAX_LEVELS = 5


def _max_book_level_from_data(data: dict) -> int:
    """Sum the detected max-book-level sources from a decoded save dict.

    Split from `read_max_book_level` so it can be unit-tested on synthetic
    saves. Currently: base + the World 3 book merit (the one source verified
    against the local save). Missing/short/non-numeric fields contribute 0."""
    total = _MAX_BOOK_BASE
    tasks = data.get("Tasks")
    try:
        merit_lvl = tasks[2][2][2]
    except (TypeError, IndexError, KeyError):
        merit_lvl = 0
    if isinstance(merit_lvl, (int, float)):
        lvl = max(0, min(int(merit_lvl), _W3_MERIT_BOOK_MAX_LEVELS))
        total += lvl * _W3_MERIT_BOOK_PER_LEVEL
    return total


def read_max_book_level(save_dir: str = SAVE_DIR) -> int | None:
    """Detected account max Talent Book level (base + summed unlock sources).

    None if the save can't be read. Sums the base and the World 3 book merit
    (the only source verified against the local save → 127); other sources
    aren't included yet (see the comment above / #95), so this can undercount
    once they're unlocked. The Books tab pre-fills this but keeps the field
    editable as a manual override."""
    data = load_save(save_dir)
    if data is None:
        return None
    return _max_book_level_from_data(data)


# Stamp levels live in `StampLevel`: a list of 3 per-tab arrays — Combat,
# Skills, Misc (Idleon's three stamp pages). Each entry is one stamp's current
# level; some are stored as strings (a Haxe quirk), so coerce. Stamps level up
# by spending gold + a material with an escalating cost — there's no per-stamp
# "cap" in the save (StampLevel == StampLevelMAX), so a recommender ranks by
# cost/value (game-data upgrade tables), not by headroom. This accessor is the
# save-side half of that (discovery map: Stamps).
_STAMP_TABS = ("combat", "skills", "misc")


def _coerce_int(x: Any) -> int:
    """Haxe stores some numeric save fields as strings; coerce to int, 0 on
    anything non-numeric/empty."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def read_stamps(save_dir: str = SAVE_DIR) -> dict[str, list[int]] | None:
    """Per-tab stamp levels: ``{"combat": [...], "skills": [...], "misc": [...]}``.

    None if the save can't be read or ``StampLevel`` is missing/malformed.
    A level of 0 means the stamp is unowned/un-leveled.
    """
    data = load_save(save_dir)
    if data is None:
        return None
    raw = data.get("StampLevel")
    if not isinstance(raw, list) or len(raw) < len(_STAMP_TABS):
        return None
    return {
        tab: [_coerce_int(x) for x in sub] if isinstance(sub, list) else []
        for tab, sub in zip(_STAMP_TABS, raw)
    }


# Card star recommender inputs. Counts live in Cards[0] (code -> copies);
# the account star cap is 4, +1 once Rift level >= 45 unlocks five-star
# cards (Rift[0] is the rift level), +1 once a Spelunking lore boss
# unlocks six-star (not decoded here — defaults off; only very-late
# accounts hit the 6th star). The "five-star list" of auto-maxed cards is
# a comma-string in OptionsListAccount[155]. Mirrors IdleonToolbox
# parsers/cards.ts (parseCards).
_RIFT_FIVE_STAR_LEVEL = 45
_OLA_FIVE_STAR_CARDS = 155
_CARD_BASE_MAX_STARS = 4


def read_card_counts(save_dir: str = SAVE_DIR) -> dict[str, int] | None:
    """Per-card collected copy counts (save `Cards[0]`: code -> copies).
    None if the save can't be read or the structure is unexpected."""
    data = load_save(save_dir)
    if data is None:
        return None
    cards = data.get("Cards")
    if not isinstance(cards, list) or not cards or not isinstance(cards[0], dict):
        return None
    return {k: int(v) for k, v in cards[0].items() if isinstance(v, (int, float))}


def read_card_five_star_set(save_dir: str = SAVE_DIR) -> set[str]:
    """Card codes auto-pinned to five stars via the five-star list
    (OptionsListAccount[155], a comma-string). Empty when unset (the field
    reads 0 until the mechanic is unlocked)."""
    data = load_save(save_dir)
    if data is None:
        return set()
    ola = data.get("OptionsListAccount")
    if not isinstance(ola, list) or len(ola) <= _OLA_FIVE_STAR_CARDS:
        return set()
    raw = ola[_OLA_FIVE_STAR_CARDS]
    if not isinstance(raw, str):
        return set()
    return {s for s in raw.split(",") if s}


def read_card_star_cap(save_dir: str = SAVE_DIR) -> int:
    """Account card-star cap: 4 base, +1 once Rift level (Rift[0]) >= 45.
    The six-star Spelunking unlock isn't decoded yet (defaults off), so
    this can under-cap a very-late account by one — safe (it just won't
    suggest farming toward a 6th star). Defaults to 4 if unreadable."""
    data = load_save(save_dir)
    if data is None:
        return _CARD_BASE_MAX_STARS
    rift = data.get("Rift")
    level = rift[0] if isinstance(rift, list) and rift and isinstance(rift[0], (int, float)) else 0
    return _CARD_BASE_MAX_STARS + (1 if level >= _RIFT_FIVE_STAR_LEVEL else 0)


def read_darts_cooldown(save_dir: str = SAVE_DIR) -> dict[str, float | int] | None:
    """Read the live darts cooldown state from the save.

    Returns a dict with:
        cooldown_seconds  : seconds of cooldown remaining (negative = playable)
        plays_today       : darts sessions played since the last daily reset

    Returns None if the save can't be read or the OLA array is too short.
    The unit conversion (5 ticks per second) was verified empirically.
    """
    data = load_save(save_dir)
    if data is None:
        return None
    ola = data.get("OptionsListAccount")
    if not isinstance(ola, list) or len(ola) <= _OLA_DARTS_PLAYS_TODAY:
        return None
    cd_ticks = ola[_OLA_DARTS_COOLDOWN_TICKS]
    plays = ola[_OLA_DARTS_PLAYS_TODAY]
    if not all(isinstance(v, (int, float)) for v in (cd_ticks, plays)):
        return None
    return {
        "cooldown_seconds": float(cd_ticks) / _OLA_TICKS_PER_SECOND,
        "plays_today": int(plays),
    }


def read_minigame_plays(save_dir: str = SAVE_DIR) -> dict[str, int] | None:
    """Read MinigamePlays per character from the save. Returns
    {character_name: plays_remaining} or None if the save can't be read.

    NOTE: the count is account-wide (chopping + catching + mining share
    one pool, refreshed daily). Each character's PersonalValuesMap
    field is a stale per-character cache that updates when that
    character becomes active — same pattern as the hoops cooldown's
    OL[6]. For the live shared count, prefer `read_minigame_plays_shared`.
    Kept for tooling that wants the raw per-character snapshots.
    """
    data = load_save(save_dir)
    if data is None:
        return None
    out: dict[str, int] = {}
    for name, info in (data.get("PlayerDATABASE") or {}).items():
        plays = (info.get("PersonalValuesMap") or {}).get("MinigamePlays")
        if isinstance(plays, (int, float)):
            out[name] = int(plays)
    return out


def read_minigame_plays_shared(save_dir: str = SAVE_DIR) -> int | None:
    """Return the shared (account-wide) minigame plays count by reading
    the most-recently-active character's MinigamePlays cache. None if
    the save can't be read or no character has the field.
    """
    import time
    data = load_save(save_dir)
    if data is None:
        return None
    now = time.time()
    best_afk: float | None = None
    best_plays: int | None = None
    for info in (data.get("PlayerDATABASE") or {}).values():
        plays = (info.get("PersonalValuesMap") or {}).get("MinigamePlays")
        away = info.get("PlayerAwayTime")
        if not isinstance(plays, (int, float)):
            continue
        if not isinstance(away, (int, float)):
            continue
        afk = now - float(away)
        if best_afk is None or afk < best_afk:
            best_afk = afk
            best_plays = int(plays)
    return best_plays


# Hoops cooldown: account-wide, lives in OptionsListAccount[423]. Same
# convention as the darts cooldown at OLA[439] — negative = playable,
# positive = ticks of cooldown remaining, 5 ticks/sec. Identified
# empirically (2026-05-09) by diffing two save dumps ~66s apart and
# finding OLA[423] decrementing at exactly 5 ticks/sec; the prior
# attempt at PlayerDATABASE[name].OptionsList[6] turned out to be a
# per-character snapshot of the cooldown duration, not a live counter.
_OLA_HOOPS_COOLDOWN_TICKS = 423


def read_hoops_cooldown(save_dir: str = SAVE_DIR) -> float | None:
    """Return the shared hoops cooldown in seconds — negative when
    playable, positive when on cooldown. None if the save can't be read
    or the OLA array is too short.

    Decrements at 5 ticks/sec in real time (independent of which
    character is active), matching the in-game cooldown indicator.
    """
    data = load_save(save_dir)
    if data is None:
        return None
    ola = data.get("OptionsListAccount")
    if not isinstance(ola, list) or len(ola) <= _OLA_HOOPS_COOLDOWN_TICKS:
        return None
    ticks = ola[_OLA_HOOPS_COOLDOWN_TICKS]
    if not isinstance(ticks, (int, float)):
        return None
    return float(ticks) / _OLA_TICKS_PER_SECOND


# Hoops plays-today: OLA[424] ("Hoops Played Today" in the community
# schema). Resets at the daily reset, increments once per play, and
# escalates the reset timer — the input to the cooldown formula (#22).
_OLA_HOOPS_PLAYS_TODAY = 424

# Minigame play-cooldown formula (#22). Hoops and darts run on one shared
# subsystem (ActorEvents_510) that COMPUTES the reset timer from the
# number of plays today — it is not stored in any OLA base index (the
# darts "base" at OLA[442] is actually a high score; see above). The
# cooldown escalates quadratically with plays-today and caps at ~900
# ticks = 180s = 3 minutes. (NOT 15 minutes — the save stores ticks at
# 5/sec, so 900 ticks reads like "900" but is 180s; that 5x is the source
# of the anecdotal "15 min".)
#
# CONFIDENCE: the shape (quadratic), the driver (plays-today), and the cap
# (~900t/180s) are high-confidence — 530 save observations across 6 days
# plus a darts cross-check all agree, and the index meanings match the
# community schema. The exact constants below are PROVISIONAL: the
# observations are lower bounds (sampled ~1s after each save flush), so
# this best-fit polynomial can't be pinned to the game's exact expression.
# To finalise, capture cooldown_ticks at save_age < 0.5s for plays 1-7 in
# one clean day. Full derivation + alternatives: docs/hoops_cooldown.md.
_MINIGAME_COOLDOWN_CAP_TICKS = 900


def minigame_cooldown_ticks_for_plays(plays_today: int) -> int:
    """Predicted reset-timer ticks set after the Nth play of the day.

    `plays_today` is 1-indexed: the first play of the day sets the n=1
    value. n <= 0 returns 0 (no cooldown before the first play). Result
    is clamped to the cap. Provisional empirical fit (see module note)."""
    n = int(plays_today)
    if n <= 0:
        return 0
    ticks = round(12 * n * n + 66 * n - 26)
    return max(0, min(_MINIGAME_COOLDOWN_CAP_TICKS, ticks))


def predict_hoops_cooldown_seconds(plays_today: int) -> float:
    """Predicted hoops reset-timer duration in seconds for a given
    plays-today count (#22). Convenience wrapper over
    `minigame_cooldown_ticks_for_plays`."""
    return minigame_cooldown_ticks_for_plays(plays_today) / _OLA_TICKS_PER_SECOND


def read_hoops_plays_today(save_dir: str = SAVE_DIR) -> int | None:
    """Return hoops plays-today (OLA[424]) — resets daily, escalates the
    cooldown. None if the save can't be read or the array is too short."""
    data = load_save(save_dir)
    if data is None:
        return None
    ola = data.get("OptionsListAccount")
    if not isinstance(ola, list) or len(ola) <= _OLA_HOOPS_PLAYS_TODAY:
        return None
    n = ola[_OLA_HOOPS_PLAYS_TODAY]
    if not isinstance(n, (int, float)):
        return None
    return int(n)


def read_talents(save_dir: str = SAVE_DIR) -> dict[str, dict] | None:
    """Per-character talent state for the book recommender (#44).

    Returns {character_name: {
        "character_class": int | None,   # raw class id (e.g. 32)
        "skill_levels": list[int],       # current talent levels (SkillLevels)
        "skill_levels_max": list[int],   # talent caps (SkillLevelsMAX; -1 inactive)
    }} or None if the save can't be read. Books raise entries of
    skill_levels_max; a talent AT cap (level >= max) is a book candidate.
    """
    data = load_save(save_dir)
    if data is None:
        return None
    out: dict[str, dict] = {}
    for name, info in (data.get("PlayerDATABASE") or {}).items():
        sl = info.get("SkillLevels")
        slm = info.get("SkillLevelsMAX")
        if not isinstance(sl, list) or not isinstance(slm, list):
            continue
        cls = info.get("CharacterClass")
        out[name] = {
            "character_class": int(cls) if isinstance(cls, (int, float)) else None,
            "skill_levels": sl,
            "skill_levels_max": slm,
        }
    return out
