"""Round-trip test for the shot SQLite log."""
import sqlite3
import threading

from common.shot_log import (
    open_db, log_shot, fetch_makes, fetch_shots, fetch_clean_trajectories,
    MIN_TRAJECTORY_DISTANCE_PX,
)


def test_open_db_creates_schema(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    # Newly-added diagnostic columns from the metrics-backlog batch must
    # exist on a fresh DB so callers can write them without an
    # OperationalError. Catches schema drift if the SCHEMA constant and
    # _LATE_COLUMNS list disagree.
    cur = conn.execute("PRAGMA table_info(shots)")
    cols = {row[1] for row in cur.fetchall()}
    conn.close()
    assert "shots" in tables
    for col in ("rim_match_scale", "time_since_last_shot_ms",
                "ball_flight_ms", "bob_ymin", "bob_ymax", "bob_period_ms"):
        assert col in cols, f"missing column {col!r}"


def test_late_column_migration_on_existing_db(tmp_path):
    """Sanity check: an existing DB without the new diagnostic columns
    gets them added by open_db's _migrate. Build an old-style schema
    (the columns the backfill UPDATE references plus a few extras),
    then re-open and verify the new columns appear and a log_shot
    using them round-trips."""
    db_path = tmp_path / "old_shots.db"
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        "CREATE TABLE shots ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_started TEXT, hoop_x INTEGER, "
        "made INTEGER, ball_landing_x INTEGER)"
    )
    raw.commit()
    raw.close()
    conn = open_db(db_path)
    log_shot(
        conn, session_started="x", hoop_x=700,
        rim_match_scale=0.92, time_since_last_shot_ms=1234,
        ball_flight_ms=2900, bob_ymin=290, bob_ymax=510,
        bob_period_ms=410,
    )
    row = conn.execute(
        "SELECT rim_match_scale, time_since_last_shot_ms, ball_flight_ms, "
        "bob_ymin, bob_ymax, bob_period_ms FROM shots"
    ).fetchone()
    conn.close()
    assert row == (0.92, 1234, 2900, 290, 510, 410)


def test_log_shot_inserts_partial_row(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    log_shot(
        conn,
        session_started="2026-05-01T16:00:00",
        shot_idx=1,
        hoop_x=710,
        hoop_y=448,
        offset=14,
        target_y=462,
        made=0,
        score_diff=0.0,
    )
    rows = conn.execute("SELECT shot_idx, hoop_y, \"offset\", made FROM shots").fetchall()
    conn.close()
    assert rows == [(1, 448, 14, 0)]


def test_log_shot_handles_offset_keyword(tmp_path):
    """`offset` is a SQLite reserved-ish word — make sure quoting works."""
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, shot_idx=1, offset=42)
    val = conn.execute("SELECT \"offset\" FROM shots").fetchone()[0]
    conn.close()
    assert val == 42


def test_open_db_migrates_existing_db_with_late_columns(tmp_path):
    """An old DB created before perturbation/lives_diff/etc. existed should
    pick them up via _migrate when reopened, not error out."""
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        'CREATE TABLE shots ('
        '  id INTEGER PRIMARY KEY, session_started TEXT, shot_idx INTEGER, '
        '  hoop_x INTEGER, "offset" INTEGER, made INTEGER'
        ')'
    )
    conn.execute("INSERT INTO shots (shot_idx, hoop_x) VALUES (1, 700)")
    conn.commit()
    conn.close()

    conn = open_db(db_path)
    log_shot(conn, shot_idx=2, perturbation=-16, lives_diff=222.4)
    rows = list(conn.execute(
        "SELECT shot_idx, perturbation, lives_diff FROM shots ORDER BY id"
    ))
    conn.close()
    assert rows == [(1, None, None), (2, -16, 222.4)]


def test_log_shot_records_perturbation(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, shot_idx=1, perturbation=0)
    log_shot(conn, shot_idx=2, perturbation=-8)
    log_shot(conn, shot_idx=3, perturbation=24)
    rows = list(conn.execute("SELECT perturbation FROM shots ORDER BY id"))
    conn.close()
    assert rows == [(0,), (-8,), (24,)]


def test_log_shot_records_lives_diff(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, shot_idx=1, lives_diff=0.0)        # no tick
    log_shot(conn, shot_idx=2, lives_diff=222.4)      # tick down
    log_shot(conn, shot_idx=3, lives_diff=None)       # region invisible
    rows = list(conn.execute("SELECT lives_diff FROM shots ORDER BY id"))
    conn.close()
    assert rows == [(0.0,), (222.4,), (None,)]


def test_log_shot_records_window_dims(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, shot_idx=1, window_w=960, window_h=572)
    log_shot(conn, shot_idx=2, window_w=1280, window_h=720)  # user resized
    rows = list(conn.execute("SELECT window_w, window_h FROM shots ORDER BY id"))
    conn.close()
    assert rows == [(960, 572), (1280, 720)]


def test_log_shot_records_predicted_offset_and_code_commit(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, shot_idx=1, predicted_offset=18, code_commit="abc123def456")
    log_shot(conn, shot_idx=2, predicted_offset=22, code_commit="abc123def456-dirty")
    rows = list(conn.execute("SELECT predicted_offset, code_commit FROM shots ORDER BY id"))
    conn.close()
    assert rows == [(18, "abc123def456"), (22, "abc123def456-dirty")]


def test_log_shot_records_predictor_kind(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, shot_idx=1, predictor_kind="knn")
    log_shot(conn, shot_idx=2, predictor_kind="bivariate")
    log_shot(conn, shot_idx=3, predictor_kind=None)  # cold start
    rows = list(conn.execute("SELECT predictor_kind FROM shots ORDER BY id"))
    conn.close()
    assert rows == [("knn",), ("bivariate",), (None,)]


def test_log_shot_records_source(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, shot_idx=1, source="bot")
    log_shot(conn, shot_idx=2, source="human")  # hoops-observe shot
    log_shot(conn, shot_idx=3, source=None)  # legacy row
    rows = list(conn.execute("SELECT source FROM shots ORDER BY id"))
    conn.close()
    assert rows == [("bot",), ("human",), (None,)]


def test_log_shot_works_from_a_background_thread(tmp_path):
    """Regression: observe mode opens the DB on the main thread but logs
    each shot from a threading.Timer worker thread (post-shot OCR fires
    POST_SHOT_COOLDOWN seconds after the click). SQLite connections in
    Python default to check_same_thread=True which raises
    sqlite3.ProgrammingError on cross-thread use; that exception is
    swallowed by threading.Timer and the row silently never gets written.
    open_db must use check_same_thread=False so the worker thread can
    write."""
    conn = open_db(tmp_path / "shots.db")
    error: list[BaseException] = []

    def worker():
        try:
            log_shot(conn, shot_idx=99, source="human", made=1)
        except BaseException as e:
            error.append(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=2)
    assert not error, f"log_shot from background thread raised: {error[0]!r}"
    rows = list(conn.execute("SELECT shot_idx, source, made FROM shots"))
    conn.close()
    assert rows == [(99, "human", 1)]


def test_fetch_shots_returns_makes_and_misses_with_landing(tmp_path):
    """fetch_shots returns every shot with a tracked ball_landing_x —
    misses included, clamped included — so the trajectory predictor
    can learn from both make/miss and bob-extreme rows."""
    conn = open_db(tmp_path / "shots.db")
    # Make with landing
    log_shot(conn, hoop_y=300, hoop_x=600, platform_y=320, ball_landing_x=602,
             made=1, clean_make=1, clamped=0, required_direction="up")
    # Miss with landing — should still be returned
    log_shot(conn, hoop_y=400, hoop_x=700, platform_y=420, ball_landing_x=540,
             made=0, clean_make=0, clamped=0, required_direction="up")
    # Clamped make: kept (#20). The logged platform_y is what actually
    # fired, so it's a usable training input.
    log_shot(conn, hoop_y=500, hoop_x=800, platform_y=509, ball_landing_x=750,
             made=1, clean_make=1, clamped=1, required_direction="up")
    # Pollution: wrong direction (excluded)
    log_shot(conn, hoop_y=400, hoop_x=700, platform_y=420, ball_landing_x=700,
             made=1, clean_make=1, clamped=0, required_direction="down")
    # Pollution: ball_landing_x missing (excluded — no trajectory signal)
    log_shot(conn, hoop_y=400, hoop_x=700, platform_y=420,
             made=0, clean_make=0, clamped=0, required_direction="up")
    rows = fetch_shots(conn, "up")
    conn.close()
    assert sorted(rows) == [
        (300.0, 600.0, 320.0, 602.0),
        (400.0, 700.0, 420.0, 540.0),
        (500.0, 800.0, 509.0, 750.0),
    ]


def test_fetch_clean_trajectories_drops_bounces_keeps_clean_shots(tmp_path):
    """A bounce lands close to platform_x; a clean shot lands far away.
    The filter cuts the former but keeps the latter."""
    conn = open_db(tmp_path / "shots.db")
    # Clean shot: ball flew 500 px right of launcher.
    log_shot(conn, hoop_y=400, hoop_x=700, platform_y=420, platform_x=136,
             ball_landing_x=636, made=1, clean_make=1, clamped=0,
             required_direction="up")
    # Bounce: ball came back to within 100 px of launcher (rim ricochet).
    log_shot(conn, hoop_y=400, hoop_x=700, platform_y=420, platform_x=136,
             ball_landing_x=236, made=0, clean_make=0, clamped=0,
             required_direction="up")
    # Borderline at the threshold (kept).
    log_shot(conn, hoop_y=400, hoop_x=700, platform_y=420, platform_x=136,
             ball_landing_x=136 + MIN_TRAJECTORY_DISTANCE_PX,
             made=0, clean_make=0, clamped=0, required_direction="up")
    # Just below threshold (dropped).
    log_shot(conn, hoop_y=400, hoop_x=700, platform_y=420, platform_x=136,
             ball_landing_x=136 + MIN_TRAJECTORY_DISTANCE_PX - 1,
             made=0, clean_make=0, clamped=0, required_direction="up")
    rows = fetch_clean_trajectories(conn, "up")
    conn.close()
    landings = sorted(int(r[3]) for r in rows)
    assert landings == [136 + MIN_TRAJECTORY_DISTANCE_PX, 636]


def test_fetch_clean_trajectories_inherits_other_filters(tmp_path):
    """The bounce filter doesn't override wrong-direction or
    missing-landing filters from the underlying fetch_shots logic.
    Clamped rows are NOT excluded (see #20)."""
    conn = open_db(tmp_path / "shots.db")
    # Wrong direction → excluded.
    log_shot(conn, hoop_y=400, hoop_x=700, platform_y=420, platform_x=136,
             ball_landing_x=636, made=0, clamped=0, required_direction="down")
    rows = fetch_clean_trajectories(conn, "up")
    conn.close()
    assert rows == []


def test_fetch_clean_trajectories_keeps_clamped_rows(tmp_path):
    """Clamped rows used to be filtered out; now they're kept so the
    predictor can learn from perturbation discoveries at the bob limit
    (#20). The logged platform_y is the actual fire value, not the
    nominal target_y, so it's still factual training data."""
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, hoop_y=400, hoop_x=700, platform_y=509, platform_x=136,
             ball_landing_x=636, made=1, clean_make=1, clamped=1,
             required_direction="up")
    rows = fetch_clean_trajectories(conn, "up")
    conn.close()
    assert len(rows) == 1
    assert rows[0] == (400.0, 700.0, 509.0, 636.0)


def test_fetch_clean_trajectories_threshold_is_overridable(tmp_path):
    """Caller can pass a custom min_distance_px (e.g. for analysis tools
    inspecting the bounce cluster itself)."""
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, hoop_y=400, hoop_x=700, platform_y=420, platform_x=136,
             ball_landing_x=200, made=0, clamped=0, required_direction="up")
    # Default threshold 300 → row excluded.
    assert fetch_clean_trajectories(conn, "up") == []
    # Custom threshold 50 → row kept.
    rows = fetch_clean_trajectories(conn, "up", min_distance_px=50)
    conn.close()
    assert len(rows) == 1


def test_fetch_makes_excludes_misses_and_wrong_direction(tmp_path):
    """fetch_makes returns clean makes (clean_make=1) in the matching
    direction, including clamped rows (#20)."""
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, hoop_y=300, hoop_x=600, platform_y=320, made=1,
             clean_make=1, clamped=0, required_direction="up")
    log_shot(conn, hoop_y=400, hoop_x=700, platform_y=420, made=1,
             clean_make=1, clamped=0, required_direction="up")
    # Clamped clean make: kept (see #20). The logged platform_y is the
    # actual fire value — useful training data.
    log_shot(conn, hoop_y=500, hoop_x=800, platform_y=509, made=1,
             clean_make=1, clamped=1, required_direction="up")
    # Pollution
    log_shot(conn, hoop_y=400, hoop_x=700, platform_y=9999, made=0,
             clean_make=0, clamped=0, required_direction="up")  # miss
    log_shot(conn, hoop_y=400, hoop_x=700, platform_y=9999, made=1,
             clean_make=1, clamped=0, required_direction="down")  # wrong direction
    log_shot(conn, hoop_y=400, hoop_x=700, platform_y=9999, made=1,
             clean_make=0, clamped=0, required_direction="up")  # rattle-in
    rows = fetch_makes(conn, "up")
    conn.close()
    assert sorted(rows) == [
        (300.0, 600.0, 320.0),
        (400.0, 700.0, 420.0),
        (500.0, 800.0, 509.0),
    ]


def test_migrate_backfills_clean_make_for_existing_rows(tmp_path):
    """Old DB rows without clean_make get backfilled at open_db time:
    - made=0 → clean_make=0
    - made=1, no trajectory → clean_make=1 (trust historical data)
    - made=1, ball_landing_x within tolerance → clean_make=1
    - made=1, ball_landing_x outside tolerance → clean_make=0 (rattle)
    """
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        'CREATE TABLE shots ('
        '  id INTEGER PRIMARY KEY, shot_idx INTEGER, hoop_x INTEGER, '
        '  made INTEGER, ball_landing_x INTEGER'
        ')'
    )
    conn.executemany(
        "INSERT INTO shots (shot_idx, hoop_x, made, ball_landing_x) VALUES (?,?,?,?)",
        [
            (1, 700, 0, 500),       # miss
            (2, 700, 1, None),      # historical make, no trajectory → trust
            (3, 700, 1, 720),       # clean make (20px past)
            (4, 700, 1, 900),       # rattle-in (200px past)
        ],
    )
    conn.commit()
    conn.close()

    conn = open_db(db_path)
    rows = list(conn.execute(
        "SELECT shot_idx, made, clean_make FROM shots ORDER BY shot_idx"
    ))
    conn.close()
    assert rows == [(1, 0, 0), (2, 1, 1), (3, 1, 1), (4, 1, 0)]


def test_migrate_classifies_backboard_bounce_as_not_clean(tmp_path):
    """A make with ball_peak_x >> ball_landing_x means the ball flew past
    the hoop, bounced off the backboard, and dropped back in. The new
    rule should classify this as clean_make=0 — the bot got lucky, but
    the trajectory shouldn't be used as a learning target."""
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        'CREATE TABLE shots ('
        '  id INTEGER PRIMARY KEY, shot_idx INTEGER, hoop_x INTEGER, '
        '  made INTEGER, ball_landing_x INTEGER, ball_peak_x INTEGER'
        ')'
    )
    conn.executemany(
        "INSERT INTO shots (shot_idx, hoop_x, made, ball_landing_x, ball_peak_x) VALUES (?,?,?,?,?)",
        [
            # Made, landing near hoop, peak near landing → clean
            (1, 589, 1, 600, 605),
            # Made, landing near hoop, but peak 284px past → bounce, not clean
            (2, 589, 1, 603, 887),
            # Made, landing near hoop, peak NULL (no trajectory) → trust make
            (3, 589, 1, 600, None),
            # Made, landing near hoop, peak only 30px past → borderline, still clean
            (4, 589, 1, 600, 630),
        ],
    )
    conn.commit()
    conn.close()

    conn = open_db(db_path)
    rows = list(conn.execute(
        "SELECT shot_idx, clean_make FROM shots ORDER BY shot_idx"
    ))
    conn.close()
    assert rows == [(1, 1), (2, 0), (3, 1), (4, 1)]


def test_migrate_does_not_overwrite_explicit_clean_make(tmp_path):
    """If clean_make is already set on a row, the migration leaves it
    alone — only NULL rows get backfilled."""
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, shot_idx=1, hoop_x=700, made=1,
             ball_landing_x=900, clean_make=1)  # caller asserted clean despite distance
    conn.close()
    # Reopen — _migrate runs again, must not flip clean_make to 0.
    conn = open_db(tmp_path / "shots.db")
    val = conn.execute("SELECT clean_make FROM shots").fetchone()[0]
    conn.close()
    assert val == 1


def test_log_shot_returns_rowid_and_set_make_corrects(tmp_path):
    """log_shot returns the inserted rowid; set_make retro-corrects the
    make verdict (hoop-respawn signal, #37) and stamps made_source."""
    from common.shot_log import open_db, log_shot, set_make
    conn = open_db(tmp_path / "shots.db")
    first = log_shot(conn, session_started="s", shot_idx=1, made=0, clean_make=0)
    second = log_shot(conn, session_started="s", shot_idx=2, made=0, clean_make=0)
    assert isinstance(first, int) and second == first + 1
    set_make(conn, first, 1, 0, "respawn")
    rows = conn.execute(
        "SELECT id, made, clean_make, made_source FROM shots ORDER BY id"
    ).fetchall()
    assert rows[0] == (first, 1, 0, "respawn")
    assert rows[1] == (second, 0, 0, None)  # untouched
    conn.close()
