"""#41 step 3: fit the E[stripe] GP offline and sanity-check the surface.

Checks the two falsifiable predictions from the issue thread before the
model goes live:
  (1) headwind (wind_x < 0) should depress EV at every vy — the
      hit-rate penalty observed empirically (96% -> 82%);
  (2) the best vy should shift with wind_y — vertical wind moves the
      arc, so the vy target compensates.

vy units are px/s (cadence-invariant; legacy px/poll rows are converted
by fetch_stripe_rows via each fire's actual poll gap).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from minigames.darts.shot_log import open_db  # noqa: E402
from minigames.darts.stripe_model import (  # noqa: E402
    fetch_stripe_rows,
    fit_stripe_gp,
    model_vy_band,
)

root = Path(__file__).parent.parent


def main() -> None:
    # open_db (not raw sqlite3.connect) so schema migrations run —
    # fetch_stripe_rows reads the late-added vy column.
    db = open_db(root / "minigames/darts/assets/darts.db")
    rows = fetch_stripe_rows(db)
    n_hits = sum(1 for r in rows if r[4] > 0)
    print(f"training rows: {len(rows)} ({n_hits} hits, {len(rows) - n_hits} misses)")
    vys = sorted(r[2] for r in rows)
    poses = sorted(r[3] for r in rows)
    print(f"vy range px/s: {vys[0]:.0f}..{vys[-1]:.0f} (median {vys[len(vys)//2]:.0f})")
    print(f"pose_y range: {poses[0]:.0f}..{poses[-1]:.0f} (median {poses[len(poses)//2]:.0f})")

    model = fit_stripe_gp(rows)
    if model is None:
        print("below sample floor — no fit")
        return
    print(f"kernel: {model._gp.kernel_}")
    print(f"pose support: {model.pose_support}")

    pose_y = poses[len(poses) // 2]
    grid = list(range(-64, 1, 8))
    print(f"\nEV surface at pose_y={pose_y:.0f} (rows: wind, cols: vy px/s)")
    print("  wind (x,y)    " + " ".join(f"{vy:>5}" for vy in grid))
    scenarios = [
        ("calm", 0.0, 0.0),
        ("tail 6mph", 6.0, 0.0),
        ("head 6mph", -6.0, 0.0),
        ("head 10mph", -6.6, 7.5),  # the streak-breaker state from c0925b2
        ("up 6mph", 0.0, 6.0),
        ("down 6mph", 0.0, -6.0),
    ]
    for name, wx, wy in scenarios:
        evs = model.predict_grid(wx, wy, pose_y, grid)
        band, best_vy, best_ev = model_vy_band(model, wx, wy, pose_y)
        print(
            f"  {name:<12} " + " ".join(f"{v:5.2f}" for v in evs)
            + f"   best vy={best_vy} ev={best_ev:.2f} band=[{band[0]},{band[1]}]"
        )

    print("\nsanity (1) — headwind depresses EV vs calm at matched vy:")
    for vy in (-28, -40):
        calm = model.predict(0.0, 0.0, vy, pose_y)
        head = model.predict(-6.0, 0.0, vy, pose_y)
        print(f"  vy={vy}: calm {calm:.2f} vs head {head:.2f}  {'OK' if head < calm else 'VIOLATED'}")

    print("\nsanity (2) — best vy shifts with wind_y:")
    for wy in (-6.0, 0.0, 6.0):
        _, best_vy, _ = model_vy_band(model, 0.0, wy, pose_y)
        print(f"  wind_y={wy:+.0f}: best vy = {best_vy}")


if __name__ == "__main__":
    main()
