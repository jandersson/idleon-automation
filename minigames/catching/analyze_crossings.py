"""Extract ring-centre crossings from catching --trace CSVs (#62).

The collision-model calibration tool: for each trace it builds per-hoop
tracks from the sparse gap detections, fits each track's linear scroll,
extrapolates the ring-centre crossing time past fly_x (find_next_gap only
reports hoops whose centre is still RIGHT of the fly, so the crossing
itself is never directly observed), interpolates fly_y at that moment from
the dense per-poll series, and reports the avatar's offset from the hole
centre plus its excursion over the full ring-overlap window.

Score alignment: trace rows with fired=1 correspond in order to the flaps
table rows for the same session (matched by nearest session_started to the
trace stamp), giving a wall-clock <-> trace-time offset; the flaps table's
throttled score reads then land on the trace timeline as increment events.
Only digits with captured templates read (see assets/digit_templates/) —
other values hold the last readable score.

    uv run catching-analyze-crossings                 # newest trace
    uv run catching-analyze-crossings trace_*.csv     # specific traces

This is how the #62 numbers were measured (2026-07-11): passable hole =
centre +/- ~15px (threads up to |off| 14, deaths at +15/+18), rim lethal
across the FULL x-overlap (not just the centre plane), dodging entirely
above/below the band safe, ring width ~17px, band ~52px, and the scroll
speed ramp v(t) ~ 63 + 0.87t px/s. See docs/catching_timing.md.
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_HERE = Path(__file__).parent
ASSETS = _HERE / "assets"
DB_PATH = ASSETS / "catching.db"
TRACES_DIR = ASSETS / "traces"

APPROACH0 = 78.0    # px/s prior for track gating only (fits use per-track slope)
TRACK_TOL = 16.0    # px, detection-to-track-prediction gate
CLEAN_H = (46, 62)  # single un-merged ring bbox heights; merged contours run 75-110
CROP_W = 520        # play-crop width; right-edge-clipped boxes excluded from geometry


def _f(row: dict, key: str) -> float | None:
    v = row.get(key)
    return None if v is None or v == "" else float(v)


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _slope_fit(ts, xs):
    """(intercept, slope) least squares, or (None, None)."""
    n = len(ts)
    if n < 2:
        return None, None
    mx, my = sum(ts) / n, sum(xs) / n
    den = sum((t - mx) ** 2 for t in ts)
    if den < 1e-9:
        return None, None
    b = sum((t - mx) * (x - my) for t, x in zip(ts, xs)) / den
    return my - b * mx, b


def build_tracks(rows: list[dict]) -> list[list[tuple]]:
    """Greedy nearest-prediction assignment of gap detections to hoop tracks.
    Each track is a list of (t, gl, gr, gt, gb)."""
    tracks: list[list[tuple]] = []
    for row in rows:
        t, gl = _f(row, "t"), _f(row, "gap_left_x")
        if t is None or gl is None:
            continue
        det = (t, gl, _f(row, "gap_right_x"), _f(row, "gap_top"),
               _f(row, "gap_bottom"))
        best = None
        for tr in tracks:
            lt, lgl = tr[-1][0], tr[-1][1]
            if t - lt > 2.5:
                continue
            err = abs(gl - (lgl - APPROACH0 * (t - lt)))
            if err < TRACK_TOL and (best is None or err < best[1]):
                best = (tr, err)
        if best is not None:
            best[0].append(det)
        else:
            tracks.append([det])
    return tracks


def _interp_fly(pts, tc, max_gap=0.3):
    before = [(t, y) for t, y in pts if t <= tc]
    after = [(t, y) for t, y in pts if t >= tc]
    if not before or not after:
        return None
    t0, y0 = before[-1]
    t1, y1 = after[0]
    if t1 - t0 > max_gap:
        return None
    return y0 if t1 == t0 else y0 + (y1 - y0) * (tc - t0) / (t1 - t0)


def _session_for(trace_path: Path) -> str | None:
    """Nearest runs.session_started to the trace file's timestamp stamp."""
    stamp = trace_path.stem.replace("trace_", "")
    try:
        t0 = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    if not DB_PATH.exists():
        return None
    db = sqlite3.connect(DB_PATH)
    best, best_dt = None, 120.0
    for (s,) in db.execute("SELECT session_started FROM runs"):
        dt = abs((datetime.fromisoformat(s) - t0).total_seconds())
        if dt < best_dt:
            best, best_dt = s, dt
    return best


def score_events(session: str | None, rows: list[dict]):
    """[(trace_t, old_score, new_score)] via fired-row <-> flap-row pairing."""
    if session is None or not DB_PATH.exists():
        return []
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    flaps = list(db.execute(
        "SELECT clicked_at, score FROM flaps WHERE session_started=? "
        "ORDER BY flap_idx", (session,)))
    fired_ts = [_f(r, "t") for r in rows if (_f(r, "fired") or 0) >= 1.0]
    if not flaps or not fired_ts:
        return []
    n = min(len(flaps), len(fired_ts))
    off = _median([
        datetime.fromisoformat(flaps[i]["clicked_at"]).timestamp() - fired_ts[i]
        for i in range(n)])
    events, prev = [], None
    for r in flaps:
        s = r["score"]
        if s is not None and s != prev:
            wall = datetime.fromisoformat(r["clicked_at"]).timestamp()
            events.append((round(wall - off, 2), prev, s))
            prev = s
    return events


def extract_crossings(rows: list[dict]) -> list[dict]:
    """All ring-centre crossings recoverable from one trace's rows."""
    fly_pts = [(_f(r, "t"), _f(r, "fly_y")) for r in rows]
    fly_pts = [(t, y) for t, y in fly_pts if t is not None and y is not None]
    if not fly_pts:
        return []
    fly_x = _median([_f(r, "fly_x") for r in rows
                     if _f(r, "fly_x") is not None])
    tmax = max(t for t, _ in fly_pts)

    cands = []
    for tr in build_tracks(rows):
        clean = [d for d in tr
                 if d[3] is not None and d[4] is not None
                 and CLEAN_H[0] <= d[4] - d[3] <= CLEAN_H[1]
                 and (d[2] is None or d[2] < CROP_W - 4)]
        if len(clean) < 3:
            continue
        fit_from = clean if len(clean) >= 4 else tr
        a, b = _slope_fit([d[0] for d in fit_from], [d[1] for d in fit_from])
        if a is None or b is None or b > -35:
            continue  # stationary plateau / clutter, not a scrolling ring
        w = _median([d[2] - d[1] for d in clean if d[2] is not None])
        gt = _median([d[3] for d in clean])
        gb = _median([d[4] for d in clean])
        hc = 0.5 * (gt + gb)
        tc = (fly_x - w / 2.0 - a) / b        # ring centre reaches fly_x
        t_enter = (fly_x - w - a) / b         # ring right edge reaches fly
        t_exit = (fly_x - a) / b              # ring left edge leaves fly
        if tc < fit_from[0][0] - 0.2 or tc > tmax + 1.5:
            continue
        fy = _interp_fly(fly_pts, tc)
        oys = [y for t, y in fly_pts
               if min(t_enter, t_exit) <= t <= max(t_enter, t_exit)]
        cands.append({
            "t_cross": tc, "hole_center": hc, "band_top": gt, "band_bot": gb,
            "width": w, "speed": -b, "fly_y": fy,
            "off": None if fy is None else fy - hc,
            "ovl_min": min(oys) if oys else None,
            "ovl_max": max(oys) if oys else None,
            "n_clean": len(clean),
        })

    # dedup: alternating contour variants split one ring into parallel tracks
    cands.sort(key=lambda c: c["t_cross"])
    merged: list[dict] = []
    for c in cands:
        if merged and abs(c["t_cross"] - merged[-1]["t_cross"]) < 1.0 \
                and abs(c["hole_center"] - merged[-1]["hole_center"]) < 10:
            if c["n_clean"] > merged[-1]["n_clean"]:
                merged[-1] = c
            continue
        merged.append(c)
    return merged


def _verdict(c: dict) -> str:
    if c["fly_y"] is None:
        return "?"
    if c["fly_y"] > c["band_bot"] + 2:
        return "under-ring"
    if c["fly_y"] < c["band_top"] - 2:
        return "over-ring"
    return f"in-band off={c['off']:+.0f}"


def run() -> None:
    names = sys.argv[1:]
    if not names:
        newest = sorted(TRACES_DIR.glob("trace_*.csv"))
        if not newest:
            print(f"No traces in {TRACES_DIR}")
            return
        names = [newest[-1].name]
    for name in names:
        path = TRACES_DIR / name
        if not path.exists():
            print(f"missing: {path}")
            continue
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        session = _session_for(path)
        print(f"\n=== {name} (session {session}) ===")
        events = score_events(session, rows)
        print(f"score events (trace t): {events}")
        print(f"{'t_cross':>7} {'holeC':>6} {'band':>10} {'w':>3} {'spd':>4} "
              f"{'flyY@c':>6} {'off':>6} {'overlap':>11} verdict")
        for c in extract_crossings(rows):
            fy = f"{c['fly_y']:6.1f}" if c["fly_y"] is not None else "     ?"
            off = f"{c['off']:+6.1f}" if c["off"] is not None else "     ?"
            ovl = (f"[{c['ovl_min']:3.0f},{c['ovl_max']:3.0f}]"
                   if c["ovl_min"] is not None else "          ?")
            print(f"{c['t_cross']:7.2f} {c['hole_center']:6.1f} "
                  f"[{c['band_top']:3.0f},{c['band_bot']:3.0f}] "
                  f"{c['width']:3.0f} {c['speed']:4.0f} {fy} {off} {ovl} "
                  f"{_verdict(c)}")


if __name__ == "__main__":
    run()
