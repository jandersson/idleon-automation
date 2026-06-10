"""Tell the chopping speed story from a session's polls.

Settles the open speed-attribution question (docs/chopping_notes.md):
does the leaf speed up per successful CHOP (maintainer's intuition,
DigitalTQ's claim), per edge BOUNCE (Steam tips thread), or both — and
is the motion eased (slower near the edges, like the hoops platform
bob) rather than constant-speed?

Segments the leaf track into sweeps (runs of constant direction),
reports each sweep's peak/mean speed alongside cumulative chops and
bounces at that point, and prints a position-vs-speed profile for the
easing check. Peak-per-sweep is the robust speed metric if motion is
eased; mean would conflate easing with the ramp.

Needs full-rate polls (POLL_LOG_INTERVAL=0, 2026-06-11) to be
trustworthy — 10Hz sessions alias sweeps and understate peaks.

Usage: python scripts/analyze_chop_speed.py [session_started]
       (default: the latest session in the polls table)
"""
import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).parent.parent / "minigames" / "chopping" / "assets" / "chopping.db"

conn = sqlite3.connect(DB)
session = sys.argv[1] if len(sys.argv) > 1 else conn.execute(
    "SELECT MAX(session_started) FROM polls"
).fetchone()[0]
rows = conn.execute(
    "SELECT t_ms, pointer_x, fired FROM polls "
    "WHERE session_started = ? AND pointer_x IS NOT NULL ORDER BY t_ms",
    (session,),
).fetchall()
print(f"session {session}: {len(rows)} leaf samples")
if len(rows) < 3:
    sys.exit("not enough samples")

bar_w = max(x for _, x, _ in rows) + 1

# Per-pair velocities, segmented into sweeps at direction reversals.
sweeps: list[dict] = []
cur: dict | None = None
chops_so_far = 0
for (t0, x0, _), (t1, x1, fired1) in zip(rows, rows[1:]):
    dt = (t1 - t0) / 1000.0
    if dt <= 0 or dt > 0.5:  # gap (cooldown hold at 10Hz-era data, or leaf dropout)
        cur = None
    else:
        dx = x1 - x0
        if dx != 0:
            direction = 1 if dx > 0 else -1
            v = abs(dx) / dt
            if cur is None or cur["dir"] != direction:
                cur = {"dir": direction, "t0": t0, "t1": t1, "vs": [],
                       "xs": [], "chops_at_start": chops_so_far}
                sweeps.append(cur)
            cur["t1"] = t1
            cur["vs"].append(v)
            cur["xs"].append((x0 + x1) / 2)
    if fired1:
        chops_so_far += 1

print(f"\n{'sweep':>5} {'t0..t1 ms':>15} {'dir':>4} {'peak px/s':>10} "
      f"{'mean px/s':>10} {'n':>4} {'chops':>6} {'bounces':>8}")
for i, s in enumerate(sweeps):
    if not s["vs"]:
        continue
    peak = max(s["vs"])
    mean = sum(s["vs"]) / len(s["vs"])
    arrow = ">" if s["dir"] > 0 else "<"
    print(f"{i:>5} {s['t0']:>7}..{s['t1']:<7} {arrow:>4} {peak:>10.0f} "
          f"{mean:>10.0f} {len(s['vs']):>4} {s['chops_at_start']:>6} {i:>8}")

# Easing check: mean |v| by bar-position quintile, pooled over sweeps.
# Eased motion shows mid-bar quintiles clearly faster than edge ones.
bins: list[list[float]] = [[] for _ in range(5)]
for s in sweeps:
    for x, v in zip(s["xs"], s["vs"]):
        bins[min(4, int(x / bar_w * 5))].append(v)
print("\nposition profile (easing check):")
for i, b in enumerate(bins):
    label = f"x {i * 20}-{(i + 1) * 20}%"
    if b:
        print(f"  {label:<12} mean {sum(b) / len(b):>6.0f} px/s  (n={len(b)})")
    else:
        print(f"  {label:<12} (no samples)")

print("\nreading the story: peak-per-sweep flat across bounces but stepping "
      "up after fired-chops => chops ramp it; rising with sweep index "
      "regardless of chops => bounces ramp it; both => both.")
