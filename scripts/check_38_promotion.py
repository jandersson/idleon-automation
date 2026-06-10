"""#38 promotion check: make_prob vs the #37 policy-stack baseline.

Same filter as the #37 measurement gate (docs/hoops_findings.md):
post-policy attempts with exploration excluded. Promotion bar (from
the #38 plan): >=50 cumulative model shots with the make_prob Wilson
95% CI still separated from the policy stack's [0.273, 0.433].
"""
import math
import sqlite3
from pathlib import Path

root = Path(__file__).parent.parent


def wilson(makes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = makes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def main() -> None:
    db = sqlite3.connect(root / "minigames/hoops/assets/shots.db")
    cur = db.cursor()
    for kind in ("make_prob", "gp"):
        cur.execute(
            """
            SELECT COUNT(*), SUM(made)
            FROM shots
            WHERE predictor_kind = ?
              AND (target_source IS NULL OR target_source != 'explore')
              AND fired_at >= '2026-06-09'
            """,
            (kind,),
        )
        n, makes = cur.fetchone()
        if not n:
            print(f"{kind}: no shots")
            continue
        lo, hi = wilson(makes or 0, n)
        print(
            f"{kind}: {makes}/{n} = {100 * (makes or 0) / n:.1f}%  "
            f"Wilson95 [{lo:.3f}, {hi:.3f}]"
        )

    # Per-session breakdown for make_prob, to see drift
    print("\nmake_prob by session (explore excluded):")
    cur.execute(
        """
        SELECT session_started, COUNT(*), SUM(made)
        FROM shots
        WHERE predictor_kind = 'make_prob'
          AND (target_source IS NULL OR target_source != 'explore')
        GROUP BY session_started ORDER BY session_started
        """
    )
    for s, n, makes in cur.fetchall():
        print(f"  {s}: {makes}/{n}")


if __name__ == "__main__":
    main()
