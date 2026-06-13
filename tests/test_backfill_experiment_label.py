"""Tests for the experiment_label backfill heuristic (#8).

The rule labels only 2026-05-03 click-position rig sessions: a session
that logged click coordinates, fired >=3 shots, and held the hoop at a
single position. Everything else (normal gameplay, non-click controlled
runs, other dates, tiny sessions) must stay NULL.
"""
from minigames.hoops.shot_log import open_db, log_shot
from scripts.backfill_experiment_label import label_click_experiments, CLICK_RIG_LABEL


def _open_with_click_cols(path):
    """Fresh DBs no longer carry click_x/click_y (dropped post-experiment);
    the historical shots.db still has them. Re-add them so the test DB
    matches the one the backfill actually runs against."""
    conn = open_db(path)
    conn.execute("ALTER TABLE shots ADD COLUMN click_x INTEGER")
    conn.execute("ALTER TABLE shots ADD COLUMN click_y INTEGER")
    conn.commit()
    return conn


def _labelled_sessions(conn):
    return {
        r[0] for r in conn.execute(
            "SELECT DISTINCT session_started FROM shots "
            "WHERE experiment_label IS NOT NULL"
        )
    }


def test_labels_only_fixed_hoop_click_sessions(tmp_path):
    conn = _open_with_click_cols(tmp_path / "shots.db")

    # (1) click-position rig: fixed hoop, click logged, >=3 shots → labelled.
    for i in range(4):
        log_shot(conn, session_started="2026-05-03T07:21:17", shot_idx=i,
                 hoop_x=586, hoop_y=410, click_x=586, click_y=410,
                 made=0, required_direction="up")
    # (2) the sweep variant: fixed hoop, click_y varied → also labelled.
    for cy in (170, 250, 330, 410, 490):
        log_shot(conn, session_started="2026-05-03T07:44:50", shot_idx=cy,
                 hoop_x=586, hoop_y=410, click_x=586, click_y=cy,
                 made=0, required_direction="up")
    # (3) normal gameplay same day: hoop moves every shot → NOT labelled.
    for i, hx in enumerate((600, 712, 588, 733)):
        log_shot(conn, session_started="2026-05-03T05:26:05", shot_idx=i,
                 hoop_x=hx, hoop_y=300 + i, click_x=hx, click_y=300,
                 made=i % 2, required_direction="up")
    # (4) non-click controlled run: fixed hoop but no click data → NOT labelled.
    for i in range(29):
        log_shot(conn, session_started="2026-05-03T10:01:57", shot_idx=i,
                 hoop_x=588, hoop_y=384, made=0, required_direction="up")
    # (5) tiny fixed-hoop session (n=2) → NOT labelled.
    for i in range(2):
        log_shot(conn, session_started="2026-05-03T06:16:57", shot_idx=i,
                 hoop_x=715, hoop_y=450, click_x=715, click_y=450,
                 made=1, required_direction="up")
    # (6) fixed-hoop click session on a DIFFERENT date → NOT labelled.
    for i in range(5):
        log_shot(conn, session_started="2026-05-04T09:00:00", shot_idx=i,
                 hoop_x=586, hoop_y=410, click_x=586, click_y=410,
                 made=0, required_direction="up")

    n = label_click_experiments(conn)
    assert n == 4 + 5  # sessions (1) and (2)
    assert _labelled_sessions(conn) == {
        "2026-05-03T07:21:17", "2026-05-03T07:44:50",
    }
    # All labelled rows carry the single label value.
    labels = {r[0] for r in conn.execute(
        "SELECT DISTINCT experiment_label FROM shots WHERE experiment_label IS NOT NULL"
    )}
    assert labels == {CLICK_RIG_LABEL}
    conn.close()


def test_backfill_is_idempotent(tmp_path):
    conn = _open_with_click_cols(tmp_path / "shots.db")
    for i in range(3):
        log_shot(conn, session_started="2026-05-03T07:21:17", shot_idx=i,
                 hoop_x=586, hoop_y=410, click_x=586, click_y=410,
                 made=0, required_direction="up")
    assert label_click_experiments(conn) == 3
    assert label_click_experiments(conn) == 0  # nothing new on re-run
    conn.close()
