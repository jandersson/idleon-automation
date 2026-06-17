"""Shared SQLite machinery for the per-bot action logs (#36).

Every bot keeps its own DB and schema (deliberately not shared — see
CLAUDE.md), but the IO plumbing around them was four identical copies:
connect with check_same_thread=False (worker threads log outcomes),
execute the CREATE TABLEs, ALTER TABLE in any late-added columns, insert
kwargs as a row, update outcome columns on an existing row. That
plumbing lives here; the schema modules next to each bot's main.py keep
owning the schemas, late-column lists, and query helpers.
"""
import sqlite3
from pathlib import Path
from typing import Iterable


def open_log_db(
    path: Path,
    schemas: Iterable[str],
    late_columns: dict[str, list[tuple[str, str]]],
) -> sqlite3.Connection:
    """Open (creating/migrating as needed) a bot's log DB.

    `schemas`: the CREATE TABLE IF NOT EXISTS statements.
    `late_columns`: per-table append-only lists of columns added after
    the original schema; each is ALTER TABLEd in, ignoring
    duplicate-column errors on DBs that already have them.

    check_same_thread=False so outcome callbacks fired from worker
    threads (threading.Timer etc.) can reuse the connection — SQLite
    serialises writes internally so that's safe.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    for schema in schemas:
        conn.execute(schema)
    for table, cols in late_columns.items():
        for name, decl in cols:
            try:
                # Quote the column name so SQL keywords work as column names
                # (insert_row quotes them too — e.g. "offset", "where").
                # Without the quotes the ALTER raises a *syntax* error that the
                # except below swallows as if the column already existed, so
                # the migration silently never runs.
                conn.execute(f'ALTER TABLE {table} ADD COLUMN "{name}" {decl}')
            except sqlite3.OperationalError:
                pass  # already exists — duplicate column error
    conn.commit()
    return conn


def insert_row(conn: sqlite3.Connection, table: str, fields: dict) -> int:
    """Insert `fields` as a row and return its rowid (so callers can
    update outcome columns later). Column names are quoted, so SQL
    keywords like "offset" are fine. Commits per row."""
    cols = ", ".join(f'"{k}"' for k in fields)
    placeholders = ", ".join("?" * len(fields))
    cur = conn.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
        tuple(fields.values()),
    )
    conn.commit()
    return cur.lastrowid


def update_row(conn: sqlite3.Connection, table: str, row_id: int, fields: dict) -> None:
    """Set `fields` on an existing row (outcome backfill helpers)."""
    assignments = ", ".join(f'"{k}" = ?' for k in fields)
    conn.execute(
        f"UPDATE {table} SET {assignments} WHERE id = ?",
        (*fields.values(), row_id),
    )
    conn.commit()
