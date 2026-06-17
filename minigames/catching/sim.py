"""Closed-loop SIM of the catching control policy against the fitted dynamics.

NOT the real game — a self-consistency harness. It runs main.py's model-path
decision (floor rescue -> should_flap_now launch -> coast -> apex-hover) on a
synthetic hoop stream using assets/dynamics.json, and reports how many hoops
the policy threads before clipping. Use it to iterate the launch/phase-lock
controller without burning live plays: if a change can't thread in its OWN
physics model, it won't live.

    uv run catching-sim                 # default sweep over hoop spacings/phases
    uv run catching-sim --verbose

Caveat: the geometry constants (ring width, detected-band vs passable-hole
height) are estimates from the wiki sprites + the 2026-06-17 trace, not exact
game measurements; treat absolute thread counts as indicative, comparisons
between controller variants as the signal. See docs/catching_timing.md.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from minigames.catching.controller import (
    load_dynamics, should_flap_now, hover_target_y, floor_rescue_due, step,
)
from minigames.catching import main as M

_HERE = Path(__file__).parent
DYNAMICS_PATH = _HERE / "assets" / "dynamics.json"

# --- sim geometry (estimates; see module docstring) ----------------------
DT = 0.027            # poll cadence (measured ~37Hz)
PLAY_H = 185          # play-crop height
FLY_X = 256           # avatar's fixed x in the crop
RING_HALF_W = 12      # ring horizontal half-width (~24px ring)
DET_HALF = 28         # detected band half-height (find_next_gap bbox)
PASS_HALF = 15        # passable hole half-height (the lethal gate)
SPAWN_X = 520         # hoop spawn x (right of the crop)


def run_one(dyn, hole_centers, spacing_px, seed_phase=0.0, verbose=False,
            max_t=80.0):
    """Run the policy over a hoop stream; return (threaded, crash_reason)."""
    hoops = [{"cx": SPAWN_X + i * spacing_px, "hc": hc, "passed": False}
             for i, hc in enumerate(hole_centers)]
    y = hole_centers[0] + dyn.rise_height_px * seed_phase
    vy = 0.0
    last_click = -1.0
    coast_until = -1.0
    t = 0.0
    threaded = 0
    while t < max_t:
        t += DT
        for h in hoops:
            h["cx"] -= dyn.approach_speed * DT
        nxt = next((h for h in hoops
                    if not h["passed"] and h["cx"] + RING_HALF_W >= FLY_X), None)
        if nxt is None and all(h["passed"] for h in hoops):
            break

        gl = gr = gt = gb = dc = None
        if nxt is not None:
            gl, gr = nxt["cx"] - RING_HALF_W, nxt["cx"] + RING_HALF_W
            gt, gb, dc = nxt["hc"] - DET_HALF, nxt["hc"] + DET_HALF, nxt["hc"]

        # --- decision: mirrors main.py's model path (launch before floor) ---
        launch = (nxt is not None
                  and should_flap_now(FLY_X, y, gl, gr, gt, gb, dyn)
                  and t >= coast_until)
        if launch:
            do, where = True, "model"
        elif floor_rescue_due(y, vy, PLAY_H):
            do, where = True, "floor"
        elif t < coast_until:
            floor = (gb - M.COAST_RESCUE_PX) if gb is not None else PLAY_H - 25
            do, where = y > floor, "coast"
        else:
            target = hover_target_y(dc, dyn) if dc is not None else PLAY_H * 0.5
            do, where = y > target + M.FLAP_MARGIN, "hover"

        if do and t - last_click >= M.MIN_CLICK_INTERVAL:
            vy = dyn.flap_vy
            last_click = t
            if launch:
                coast_until = t + M.COAST_S
            if verbose and where in ("model", "floor"):
                print(f"  t={t:5.2f} y={y:4.0f} apex->{y - dyn.rise_height_px:4.0f} "
                      f"{where} hc={nxt['hc'] if nxt else '-'}")
        y, vy = step(y, vy, DT, dyn)

        if y >= PLAY_H:
            return threaded, f"floor t={t:.2f}"
        if y <= 0:
            return threaded, f"ceiling t={t:.2f}"
        if nxt is not None:
            if gl <= FLY_X <= gr:
                lo, hi = nxt["hc"] - PASS_HALF, nxt["hc"] + PASS_HALF
                if not (lo <= y <= hi):
                    return threaded, (f"clip hc={nxt['hc']} y={y:.0f} "
                                      f"hole[{lo:.0f},{hi:.0f}] t={t:.2f}")
            if nxt["cx"] + RING_HALF_W < FLY_X and not nxt["passed"]:
                nxt["passed"] = True
                threaded += 1
    return threaded, "completed"


def run():
    parser = argparse.ArgumentParser(description="Catching control sim (fitted dynamics)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    dyn = load_dynamics(DYNAMICS_PATH)
    if dyn is None:
        print(f"No {DYNAMICS_PATH.name} — fit a --trace run first (catching-fit-dynamics).")
        return
    print(f"dyn: g={dyn.gravity:.0f} flap_vy={dyn.flap_vy:.0f} "
          f"approach={dyn.approach_speed:.0f} rise={dyn.rise_height_px:.0f} "
          f"apex={dyn.time_to_apex_s * 1000:.0f}ms")
    centers = [80, 70, 90, 75, 85] * 6
    import statistics
    print("\nthreaded per run (5 bob phases) by hoop spacing:")
    for spacing in (180, 220, 260, 320, 400):
        res = [run_one(dyn, centers, spacing, p) for p in (0.0, 0.2, 0.4, 0.6, 0.8)]
        thr = [r[0] for r in res]
        print(f"  spacing {spacing:>3}px: {thr}  median={statistics.median(thr)}  "
              f"e.g. {res[0][1]}")
    if args.verbose:
        print("\nverbose (spacing 320, phase 0.4):")
        th, c = run_one(dyn, centers, 320, 0.4, verbose=True)
        print(f"  threaded={th} {c}")


if __name__ == "__main__":
    run()
