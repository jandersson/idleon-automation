"""Fit the catching avatar's vertical dynamics from a dense --trace CSV (#60).

Reads a trajectory trace (written by `catching --trace` into assets/traces/),
estimates (gravity, flap_vy, approach_speed), prints diagnostics, and writes
assets/dynamics.json — the model the predictive flap timer (`catching --model`)
flies on. Run after one good --trace session:

    uv run catching-fit-dynamics                 # newest trace in assets/traces/
    uv run catching-fit-dynamics path/to/trace.csv
    uv run catching-fit-dynamics --dry-run       # fit + print, don't write

Fitting is offline and pure (controller.fit_dynamics); this is just the CLI
that picks the file, reports, and persists.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from minigames.catching.controller import (
    fit_dynamics, fit_gravity, fit_flap_vy, fit_approach_speed,
    load_trace, save_dynamics,
)

_HERE = Path(__file__).parent
TRACES_DIR = _HERE / "assets" / "traces"
DYNAMICS_PATH = _HERE / "assets" / "dynamics.json"


def _newest_trace() -> Path | None:
    traces = sorted(TRACES_DIR.glob("trace_*.csv"))
    return traces[-1] if traces else None


def run():
    parser = argparse.ArgumentParser(description="Fit catching dynamics from a trace CSV")
    parser.add_argument("trace", nargs="?", help="trace CSV (default: newest in assets/traces/)")
    parser.add_argument("--dry-run", action="store_true", help="print the fit; don't write dynamics.json")
    args = parser.parse_args()

    trace_path = Path(args.trace) if args.trace else _newest_trace()
    if trace_path is None or not trace_path.exists():
        print(f"No trace found. Run `catching --trace` first (writes to {TRACES_DIR}).")
        return
    rows = load_trace(trace_path)
    print(f"Loaded {len(rows)} poll rows from {trace_path.name}")
    n_fired = sum(1 for r in rows if (r.get("fired") or "0") not in ("0", "", None))
    print(f"  flaps fired: {n_fired}")

    # Per-parameter so a None is attributable to which segment was too thin.
    g = fit_gravity(rows)
    fv = fit_flap_vy(rows)
    ap = fit_approach_speed(rows)
    print(f"  gravity:        {g if g is None else round(g, 1)} px/s^2")
    print(f"  flap_vy:        {fv if fv is None else round(fv, 1)} px/s (up)")
    print(f"  approach_speed: {ap if ap is None else round(ap, 1)} px/s")

    dyn = fit_dynamics(rows)
    if dyn is None:
        print("Could not fit all three parameters — need a longer/cleaner trace "
              "(more free-fall runs, flaps, and a hoop scrolling). Not writing.")
        return
    print(f"  -> rise {dyn.rise_height_px:.0f}px, apex {dyn.time_to_apex_s*1000:.0f}ms")
    if args.dry_run:
        print("(--dry-run) not writing dynamics.json")
        return
    save_dynamics(DYNAMICS_PATH, dyn)
    print(f"Wrote {DYNAMICS_PATH}. Run with `catching --model` to use it.")


if __name__ == "__main__":
    run()
