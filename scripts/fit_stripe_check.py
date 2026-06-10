"""#41 step 3: fit the E[stripe] GP offline and sanity-check the surface.

Checks the two falsifiable predictions from the issue thread before the
model goes live:
  (1) headwind (wind_x < 0) should depress EV at every dy — the
      hit-rate penalty observed empirically (96% -> 82%);
  (2) the best dy should shift with wind_y — vertical wind moves the
      arc, so the dy target compensates.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from minigames.darts.stripe_model import (  # noqa: E402
    fetch_stripe_rows,
    fit_stripe_gp,
    model_dy_band,
)

root = Path(__file__).parent.parent


def main() -> None:
    db = sqlite3.connect(root / "minigames/darts/assets/darts.db")
    rows = fetch_stripe_rows(db)
    n_hits = sum(1 for r in rows if r[4] > 0)
    print(f"training rows: {len(rows)} ({n_hits} hits, {len(rows) - n_hits} misses)")
    dys = sorted(r[2] for r in rows)
    poses = sorted(r[3] for r in rows)
    print(f"dy range: {dys[0]:.0f}..{dys[-1]:.0f} (median {dys[len(dys)//2]:.0f})")
    print(f"pose_y range: {poses[0]:.0f}..{poses[-1]:.0f} (median {poses[len(poses)//2]:.0f})")

    model = fit_stripe_gp(rows)
    if model is None:
        print("below sample floor — no fit")
        return
    print(f"kernel: {model._gp.kernel_}")

    pose_y = poses[len(poses) // 2]
    print(f"\nEV surface at pose_y={pose_y:.0f} (rows: wind, cols: dy)")
    header = "  wind (x,y)    " + " ".join(f"{dy:>5}" for dy in range(-16, 1, 2))
    print(header)
    scenarios = [
        ("calm", 0.0, 0.0),
        ("tail 6mph", 6.0, 0.0),
        ("head 6mph", -6.0, 0.0),
        ("head 10mph", -6.6, 7.5),  # the streak-breaker state from c0925b2
        ("up 6mph", 0.0, 6.0),
        ("down 6mph", 0.0, -6.0),
    ]
    for name, wx, wy in scenarios:
        evs = model.predict_grid(wx, wy, pose_y, list(range(-16, 1, 2)))
        band, best_dy, best_ev = model_dy_band(model, wx, wy, pose_y)
        print(
            f"  {name:<12} " + " ".join(f"{v:5.2f}" for v in evs)
            + f"   best dy={best_dy} ev={best_ev:.2f} band={sorted(band)}"
        )

    print("\nsanity (1) — headwind depresses EV vs calm at matched dy:")
    for dy in (-7, -10):
        calm = model.predict(0.0, 0.0, dy, pose_y)
        head = model.predict(-6.0, 0.0, dy, pose_y)
        print(f"  dy={dy}: calm {calm:.2f} vs head {head:.2f}  {'OK' if head < calm else 'VIOLATED'}")

    print("\nsanity (2) — best dy shifts with wind_y:")
    for wy in (-6.0, 0.0, 6.0):
        _, best_dy, _ = model_dy_band(model, 0.0, wy, pose_y)
        print(f"  wind_y={wy:+.0f}: best dy = {best_dy}")


if __name__ == "__main__":
    main()
