"""Back-test the chopping fire gate against every recorded fire.

Replays each logged chop (x, vx, red_ahead at fire) through the eased
time-to-red model (detector.eased_time_to_red_ms with V_max inferred
from the session's own polls in the 3s before the click) and reports
what the gate would have decided. The bar: every red-click DEATH must
classify SKIP, and as many survived fires as possible must classify
FIRE (a skipped survival costs throughput, not safety).

Context (2026-06-11): the previous flat speed floor (0.75 x recent-max
|vx|) starved the bot at ~20 points because detection-jitter spikes of
1600-1800 px/s poisoned the recent-max; a human clicking greens scores
66. See docs/chopping_notes.md.

Usage: python scripts/validate_chop_gate.py
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from minigames.chopping.detector import eased_time_to_red_ms, infer_vmax  # noqa: E402
from minigames.chopping.main import MIN_TIME_TO_RED_MS  # noqa: E402

DB = Path(__file__).parent.parent / "minigames" / "chopping" / "assets" / "chopping.db"
BAR_W = 222

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

# The 00:13 session's death predates the vx/red_ahead columns; its fire
# state was reconstructed from the 10Hz polls during the original
# debrief: x=152 moving right at ~+275 px/s, red ahead 19px. V_max for
# that round measured ~700-790 (peak per sweep).
MANUAL_CASES = [
    {"label": "00:13 chop3 DEATH (reconstructed)", "x": 152, "vx": 275.0,
     "red_ahead": 19, "v_max": 750.0, "death": True},
]

rows = conn.execute(
    "SELECT session_started s, chop_idx, pointer_x x, leaf_vx_px_s vx, "
    "red_ahead_px ahead, clicked_at, outcome, zone "
    "FROM chops WHERE leaf_vx_px_s IS NOT NULL AND red_ahead_px IS NOT NULL "
    "ORDER BY session_started, chop_idx"
).fetchall()

print(f"{len(rows)} replayable fires + {len(MANUAL_CASES)} manual cases; "
      f"budget {MIN_TIME_TO_RED_MS}ms\n")
print(f"{'session/chop':<24} {'zone':<6} {'x':>4} {'vx':>6} {'ahead':>5} "
      f"{'vmax':>5} {'ttr_ms':>6} {'gate':>5}  outcome")

results = {"death_skipped": 0, "death_fired": 0, "surv_fired": 0, "surv_skipped": 0}


def classify(label, zone, x, vx, ahead, v_max, ttr, death, outcome):
    fires = ttr is not None and ttr >= MIN_TIME_TO_RED_MS
    gate = "FIRE" if fires else "skip"
    print(f"{label:<24} {zone:<6} {x:>4} {vx:>6.0f} {ahead:>5} "
          f"{v_max:>5.0f} {str(ttr):>6} {gate:>5}  {outcome}")
    if death:
        results["death_fired" if fires else "death_skipped"] += 1
    else:
        results["surv_fired" if fires else "surv_skipped"] += 1


for m in MANUAL_CASES:
    ttr = eased_time_to_red_ms(m["x"], m["vx"], m["red_ahead"],
                               max(m["v_max"], abs(m["vx"])), BAR_W)
    classify(m["label"], "?", m["x"], m["vx"], m["red_ahead"], m["v_max"],
             ttr, m["death"], "DEATH" if m["death"] else "survived")

for r in rows:
    # V_max from the session's polls in the 3s before this click.
    t_click = conn.execute(
        "SELECT t_ms FROM polls WHERE session_started=? AND fired=1 "
        "ORDER BY ABS(t_ms - (SELECT MIN(t_ms) FROM polls WHERE session_started=?)) ",
        (r["s"], r["s"])).fetchone()
    # Simpler: take the fired poll matching this chop by order.
    fired_polls = [p[0] for p in conn.execute(
        "SELECT t_ms FROM polls WHERE session_started=? AND fired=1 ORDER BY t_ms",
        (r["s"],))]
    idx = r["chop_idx"] - 1
    if idx >= len(fired_polls):
        continue
    t0 = fired_polls[idx]
    samples = [(p["pointer_x"], abs(p["vx"])) for p in conn.execute(
        "SELECT pointer_x, leaf_vx_px_s vx FROM polls "
        "WHERE session_started=? AND t_ms BETWEEN ? AND ? "
        "AND pointer_x IS NOT NULL AND leaf_vx_px_s IS NOT NULL",
        (r["s"], t0 - 3000, t0))]
    v_max = infer_vmax(samples, BAR_W)
    if v_max is None:
        continue
    ttr = eased_time_to_red_ms(r["x"], r["vx"], r["ahead"],
                               max(v_max, abs(r["vx"])), BAR_W)
    # Deaths: the only recorded red-click death with vx logged is the
    # 01:08 chop 20 (outcome round_ended ~1s later, dissolve signature).
    death = r["outcome"] == "round_ended" and r["s"].startswith("2026-06-11T01:08")
    classify(f"{r['s'][11:16]} chop{r['chop_idx']}", r["zone"], r["x"], r["vx"],
             r["ahead"], v_max, ttr, death,
             "DEATH" if death else (r["outcome"] or "?"))

print(f"\ndeaths skipped {results['death_skipped']}, deaths fired "
      f"{results['death_fired']} (must be 0); survivals fired "
      f"{results['surv_fired']}, survivals skipped {results['surv_skipped']} "
      f"(throughput cost only)")
