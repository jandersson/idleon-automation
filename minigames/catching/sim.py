"""Closed-loop SIM of the catching control policy against the fitted dynamics.

NOT the real game — a self-consistency harness. It runs main.py's model-path
decision (launch -> floor rescue -> coast -> apex-hover) on a synthetic hoop
stream using assets/dynamics.json, and reports how many hoops the policy
threads before clipping. Use it to iterate the launch/phase-lock controller
without burning live plays: if a change can't thread in its OWN physics model,
it won't live.

    uv run catching-sim                 # default sweep over hoop spacings/phases
    uv run catching-sim --replay        # replay run 17's measured hoop stream
    uv run catching-sim --verbose

Collision model (#62, recalibrated 2026-07-11 from the four 2026-06-17
trace runs — see docs/catching_timing.md "Collision model, measured"):

  - The ring is a small oval, NOT a full-height pipe pair. Passing entirely
    ABOVE or BELOW the ring's outer band is a safe dodge (no score, no
    death) — run 17 dodged under 6 rings at +34..+53px below centre and
    lived. The old sim scored any non-hole crossing as a clip, which is why
    it threaded 0 while reality threaded 9.
  - While the ring's ~17px-wide band x-overlaps the avatar, touching a RIM
    band (outer band minus the hole) at ANY poll is death — not just at the
    centre plane. Run 16 died at hole_centre-15 early in the overlap while
    at the plane instant it was safely above the band; run 18 fell through
    the bottom rim mid-overlap.
  - The passable hole is hole_centre ± PASS_HALF (=15): threads survived up
    to |offset| 14, deaths measured at +15 and +18.
  - Hoops scroll left with a RAMP: v(t) = APPROACH_V0 + APPROACH_RAMP*t
    (run 17: 63 + 0.87t px/s; the fitted dynamics.json approach_speed=78 is
    the run median). The controller still believes the constant fit — the
    sim deliberately models that mismatch, since the live bot has it too.

Ring geometry, measured from the traces: outer band ~52px tall (DET_HALF
26), drawn width ~17px (RING_HALF_W 8.5 — the old 12 was a wiki-sprite
estimate). Hole centres 47..120 in play-crop coords; spacing ~270px.
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from minigames.catching.controller import (
    load_dynamics, should_flap_now, hover_target_y, floor_rescue_due,
    coast_rescue_due, step,
)
from minigames.catching import main as M

_HERE = Path(__file__).parent
DYNAMICS_PATH = _HERE / "assets" / "dynamics.json"

# --- sim geometry (measured from the 2026-06-17 traces; see module doc) ----
DT = 0.027            # median decision-poll cadence (measured from the trace)
PHYS_DT = 0.005       # physics/collision substep (the game never blinks)
PLAY_H = 185          # play-crop height
FLY_X = 256           # avatar's fixed x in the crop
RING_HALF_W = 8.5     # ring drawn half-width (measured w ~17px)
DET_HALF = 26         # detected band half-height (measured ~52px bands)
PASS_HALF = 15        # passable hole half-height (threads |off|<=14, deaths 15+)
SPAWN_X = 520         # hoop spawn x (right of the crop)
APPROACH_V0 = 63.0    # scroll speed at t=0 (run 17 ramp fit)
APPROACH_RAMP = 0.87  # px/s per second (run 17: 63 + 0.87t)

# Actuation timing, measured from the 2026-06-17 traces (fall-depth during
# the post-fire blind window vs descent speed): the in-game flap lands
# ~75ms after the fire decision, and click() + bookkeeping BLOCK the poll
# loop ~160ms per fired flap. The stall is why live flaps come in pairs
# 160-215ms apart: the loop wakes, the (already-turned) avatar still reads
# below the floor bound, and the rescue refires — an echo double-flap that
# lifts the apex ~13px above a single bob. Both are systematic, so they
# apply in clean mode too.
ACT_LATENCY = 0.075   # decision -> in-game vy reset
POST_FIRE_STALL = 0.16  # decision -> next possible decision poll

# Run 17's measured hoop stream (crossing time at the ramp above, hole
# centre) — the extraction from trace_20260617_192505.csv. Replay spawns
# each ring so its centre reaches FLY_X at that time. Reality: 9 threads,
# 6 dodges, death on the last ring (bottom rim, off +18).
RUN17_STREAM = [
    (4.85, 81), (8.74, 81), (12.65, 87), (16.25, 66), (19.77, 58),
    (24.11, 63), (26.91, 95), (29.73, 118.5), (32.16, 110), (32.78, 89),
    (34.88, 78), (37.00, 61), (41.83, 63), (43.77, 106), (46.03, 103),
    (48.22, 73),
]


def _dist_at(t):
    """Horizontal distance a ring scrolls between sim time 0 and t."""
    return APPROACH_V0 * t + 0.5 * APPROACH_RAMP * t * t


def run_one(dyn, hoops, seed_phase=0.0, verbose=False, max_t=80.0,
            ramp=True, rng=None):
    """Run the policy over a hoop stream; hoops = [{cx0, hc}] with cx0 the
    spawn x at t=0. Returns (threaded, dodged, crash_reason).

    Physics and collision advance on a fine fixed step (the game never
    blinks); DECISIONS happen at poll times. With `rng` set, poll cadence
    and detection follow the measured live noise (run 17 trace): median
    poll 27ms with 19% stretched gaps to 40-230ms (detector dropouts — no
    fly seen means no decision, and the avatar free-falls through them,
    which is where the real bot's deep 140+ bob bottoms come from), and
    the ring visible on only 67% of polls (main.py then HOLDS the last
    hole centre for the hover aim but cannot launch or coast-rescue).
    Without rng, every poll is the median cadence and detection is
    perfect — the clean deterministic mode."""
    hoops = [{"cx0": h["cx0"], "hc": h["hc"], "passed": False} for h in hoops]
    y = hoops[0]["hc"] + dyn.rise_height_px * seed_phase
    vy = 0.0
    last_click = -1.0
    coast_until = -1.0
    t = 0.0
    next_poll = 0.0
    act_at = None      # pending flap: in-game vy reset lands at this time
    held_dc = None
    prev_sample = None  # (t, y) of the previous decision poll — the live
                        # loop's finite-difference vy (None across gaps
                        # >0.2s, e.g. right after every click stall)
    threaded = dodged = 0
    while t < max_t:
        # --- decision poll -------------------------------------------------
        if t >= next_poll:
            if rng is None:
                next_poll = t + DT
            elif rng.random() < 0.13:
                # detector dropout / slow grab — short, 1-2 missed polls
                # (the long 140-230ms gaps in the trace are all post-fire
                # click stalls, modeled separately via POST_FIRE_STALL)
                next_poll = t + rng.uniform(0.04, 0.07)
            else:
                next_poll = t + rng.gauss(DT, 0.003)
            d = _dist_at(t) if ramp else dyn.approach_speed * t
            for h in hoops:
                h["cx"] = h["cx0"] - d
            nxt = next((h for h in hoops
                        if not h["passed"] and h["cx"] + RING_HALF_W >= FLY_X),
                       None)
            ring_seen = nxt is not None and (rng is None or rng.random() < 0.67)
            gl = gr = gt = gb = None
            if ring_seen:
                gl, gr = nxt["cx"] - RING_HALF_W, nxt["cx"] + RING_HALF_W
                gt, gb = nxt["hc"] - DET_HALF, nxt["hc"] + DET_HALF
                held_dc = nxt["hc"]
            dc = held_dc

            # The live loop never sees true vy — only a finite difference of
            # detected positions, None across gaps > 0.2s (every post-click
            # stall). The rescue decisions must run on the same degraded
            # signal or the sim's floor anchor is unrealistically tight.
            vy_est = None
            if prev_sample is not None:
                dt_s = t - prev_sample[0]
                if 0 < dt_s <= 0.2:
                    vy_est = (y - prev_sample[1]) / dt_s
            prev_sample = (t, y)

            # --- mirrors main.py's model path (launch before floor) --------
            launch = (ring_seen
                      and should_flap_now(FLY_X, y, gl, gr, gt, gb, dyn)
                      and t >= coast_until)
            if launch:
                do, where = True, "model"
            elif floor_rescue_due(y, vy_est, PLAY_H):
                do, where = True, "floor"
            elif t < coast_until:
                floor = (gb - M.COAST_RESCUE_PX) if gb is not None \
                    else PLAY_H - 25
                do, where = coast_rescue_due(y, vy_est, floor), "coast"
            else:
                target = hover_target_y(dc, dyn) if dc is not None \
                    else PLAY_H * 0.5
                # model path: no margin (apex centres on hole)
                do, where = y > target, "hover"

            if do and t - last_click >= M.MIN_CLICK_INTERVAL:
                act_at = t + ACT_LATENCY   # flap lands after actuation latency
                last_click = t
                next_poll = t + POST_FIRE_STALL  # click blocks the loop
                if launch:
                    coast_until = t + M.COAST_S
                if verbose and where in ("model", "floor"):
                    print(f"  t={t:5.2f} y={y:4.0f} "
                          f"apex->{y - dyn.rise_height_px:4.0f} "
                          f"{where} hc={nxt['hc'] if nxt else '-'}")

        # --- physics + collision substep -----------------------------------
        t += PHYS_DT
        if act_at is not None and t >= act_at:
            vy = dyn.flap_vy
            act_at = None
        y, vy = step(y, vy, PHYS_DT, dyn)
        d = _dist_at(t) if ramp else dyn.approach_speed * t
        for h in hoops:
            h["cx"] = h["cx0"] - d
        if all(h["passed"] for h in hoops):
            break

        if y >= PLAY_H:
            return threaded, dodged, f"floor t={t:.2f}"
        if y <= 0:
            return threaded, dodged, f"ceiling t={t:.2f}"
        # Collision (#62): while the ring x-overlaps the avatar, a RIM band
        # (outer band minus hole) is lethal at any instant; the hole and the
        # space entirely above/below the band are safe. Classify the pass
        # when the ring's trailing edge clears the avatar: min |y - hc| over
        # the overlap inside the hole = thread, else dodge.
        for h in hoops:
            if h["passed"]:
                continue
            if h["cx"] - RING_HALF_W <= FLY_X <= h["cx"] + RING_HALF_W:
                band_top, band_bot = h["hc"] - DET_HALF, h["hc"] + DET_HALF
                lo, hi = h["hc"] - PASS_HALF, h["hc"] + PASS_HALF
                if band_top <= y <= band_bot and not (lo <= y <= hi):
                    return threaded, dodged, (
                        f"clip hc={h['hc']} y={y:.0f} "
                        f"hole[{lo:.0f},{hi:.0f}] band[{band_top:.0f},"
                        f"{band_bot:.0f}] t={t:.2f}")
                h.setdefault("min_off", 1e9)
                off = abs(y - h["hc"])
                if off < h["min_off"]:
                    h["min_off"] = off
            elif h["cx"] + RING_HALF_W < FLY_X:
                h["passed"] = True
                if h.get("min_off", 1e9) <= PASS_HALF:
                    threaded += 1
                else:
                    dodged += 1
    return threaded, dodged, "completed"


def synthetic_hoops(centers, spacing_px):
    return [{"cx0": SPAWN_X + i * spacing_px, "hc": hc}
            for i, hc in enumerate(centers)]


def replay_hoops():
    """Run 17's stream: spawn each ring so its centre crosses FLY_X at the
    measured time under the measured ramp."""
    return [{"cx0": FLY_X + _dist_at(tc), "hc": hc} for tc, hc in RUN17_STREAM]


def run():
    parser = argparse.ArgumentParser(description="Catching control sim (fitted dynamics)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--replay", action="store_true",
                        help="replay run 17's measured hoop stream instead of "
                             "the synthetic sweep")
    parser.add_argument("--clean", action="store_true",
                        help="deterministic mode: median poll cadence, perfect "
                             "detection (default replays with measured noise)")
    args = parser.parse_args()
    dyn = load_dynamics(DYNAMICS_PATH)
    if dyn is None:
        print(f"No {DYNAMICS_PATH.name} — fit a --trace run first (catching-fit-dynamics).")
        return
    print(f"dyn: g={dyn.gravity:.0f} flap_vy={dyn.flap_vy:.0f} "
          f"approach={dyn.approach_speed:.0f} rise={dyn.rise_height_px:.0f} "
          f"apex={dyn.time_to_apex_s * 1000:.0f}ms")
    print(f"world: v(t) = {APPROACH_V0} + {APPROACH_RAMP}t px/s, "
          f"ring w={2 * RING_HALF_W:.0f}px band={2 * DET_HALF}px "
          f"hole=+/-{PASS_HALF}px, dodges safe")

    if args.replay:
        import random
        n_runs = 200 if not args.clean else 40
        mode = "clean (deterministic)" if args.clean else \
            "noisy (measured dropout/visibility)"
        print(f"\nreplay of run 17, {n_runs} runs, {mode} "
              "(reality: 9 threads, 6 dodges, clip on ring 16 hc=73):")
        results = []
        for i in range(n_runs):
            phase = (i % 40) / 40.0
            rng = None if args.clean else random.Random(1000 + i)
            th, dg, why = run_one(dyn, replay_hoops(), phase,
                                  verbose=args.verbose, rng=rng)
            results.append((th, dg, why))
        thr = sorted(r[0] for r in results)
        dgs = sorted(r[1] for r in results)
        completed = sum(1 for r in results if r[2] == "completed")
        floors = sum(1 for r in results if r[2].startswith("floor"))
        print(f"  threads: min={thr[0]} median={statistics.median(thr)} "
              f"p90={thr[9 * len(thr) // 10]} max={thr[-1]}")
        print(f"  dodges:  median={statistics.median(dgs)} max={dgs[-1]}")
        print(f"  full-run survivals={completed}/{n_runs}  "
              f"floor deaths={floors}")
        from collections import Counter
        print(f"  thread counts: {dict(sorted(Counter(thr).items()))}")
        for th, dg, why in results[:5]:
            print(f"    e.g. threaded={th} dodged={dg} {why}")
        return

    centers = [80, 70, 90, 75, 85] * 6
    print("\nthreaded/dodged per run (5 bob phases) by hoop spacing:")
    for spacing in (180, 220, 260, 320, 400):
        res = [run_one(dyn, synthetic_hoops(centers, spacing), p)
               for p in (0.0, 0.2, 0.4, 0.6, 0.8)]
        thr = [r[0] for r in res]
        print(f"  spacing {spacing:>3}px: thr={thr} median={statistics.median(thr)} "
              f"dodged={[r[1] for r in res]}  e.g. {res[0][2]}")
    if args.verbose:
        print("\nverbose (spacing 320, phase 0.4):")
        th, dg, c = run_one(dyn, synthetic_hoops(centers, 320), 0.4, verbose=True)
        print(f"  threaded={th} dodged={dg} {c}")


if __name__ == "__main__":
    run()
