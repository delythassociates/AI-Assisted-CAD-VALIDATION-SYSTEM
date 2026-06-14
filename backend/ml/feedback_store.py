"""
feedback_store.py — SQLite-backed persistent feedback store for the Active Learning Flywheel.

Schema:
  feedback       — every engineer correction logged here
  model_registry — history of fine-tuned models deployed to production
  inference_log  — lightweight counter for total parts analyzed (append-only)
"""

import os
import sqlite3
from contextlib import contextmanager

# Resolved at import time so it works whether the server is launched from
# the repo root or from backend/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DB_PATH = os.path.join(_REPO_ROOT, "data", "feedback", "feedback.db")


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    part_id         TEXT NOT NULL,
    yml_path        TEXT,
    predicted_label INTEGER NOT NULL,
    predicted_score REAL    NOT NULL,
    engineer_label  INTEGER NOT NULL,
    -- SQLite doesn't support generated columns in older versions;
    -- we compute these as plain columns filled on INSERT via a trigger.
    is_correction   INTEGER NOT NULL DEFAULT 0,
    false_positive  INTEGER NOT NULL DEFAULT 0,
    false_negative  INTEGER NOT NULL DEFAULT 0,
    model_version   TEXT    NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_registry (
    version         TEXT PRIMARY KEY,
    trained_on_n    INTEGER,
    auc_roc         REAL,
    f1_score        REAL,
    last_feedback_id INTEGER DEFAULT 0,
    deployed        INTEGER DEFAULT 0,
    deployed_at     TIMESTAMP,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS inference_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    part_id    TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trigger to fill computed columns on insert
CREATE TRIGGER IF NOT EXISTS trg_feedback_insert
AFTER INSERT ON feedback
BEGIN
    UPDATE feedback SET
        is_correction  = CASE WHEN NEW.predicted_label != NEW.engineer_label THEN 1 ELSE 0 END,
        false_positive = CASE WHEN NEW.predicted_label = 1 AND NEW.engineer_label = 0 THEN 1 ELSE 0 END,
        false_negative = CASE WHEN NEW.predicted_label = 0 AND NEW.engineer_label = 1 THEN 1 ELSE 0 END
    WHERE id = NEW.id;
END;
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _ensure_db():
    """Create DB directory and initialise schema on first run."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    conn.close()


@contextmanager
def get_db():
    """Yield an open, row-factory-enabled connection and auto-commit."""
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_feedback(
    part_id: str,
    yml_path: str | None,
    predicted_label: int,
    predicted_score: float,
    engineer_label: int,
    model_version: str,
) -> int:
    """Insert one feedback row.  Returns the new row id."""
    with get_db() as db:
        cur = db.execute(
            """
            INSERT INTO feedback
                (part_id, yml_path, predicted_label, predicted_score,
                 engineer_label, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (part_id, yml_path, predicted_label,
             predicted_score, engineer_label, model_version),
        )
        return cur.lastrowid


def log_inference(part_id: str):
    """Increment the inference counter (called on every /validate hit)."""
    with get_db() as db:
        db.execute("INSERT INTO inference_log (part_id) VALUES (?)", (part_id,))


def get_stats() -> dict:
    """Return aggregate feedback statistics from the live DB."""
    with get_db() as db:
        row = db.execute(
            """
            SELECT
                COUNT(*)                                             AS total_feedback,
                COALESCE(SUM(is_correction),  0)                    AS total_corrections,
                COALESCE(SUM(false_positive), 0)                    AS false_positives,
                COALESCE(SUM(false_negative), 0)                    AS false_negatives,
                COUNT(DISTINCT model_version)                        AS model_versions,
                COALESCE(SUM(CASE
                    WHEN created_at >= datetime('now', '-7 days')
                    THEN is_correction ELSE 0 END), 0)              AS corrections_this_week
            FROM feedback
            """
        ).fetchone()
        return dict(row)


def get_total_inference_count() -> int:
    """Total number of parts ever validated (from inference_log)."""
    with get_db() as db:
        row = db.execute("SELECT COUNT(*) AS n FROM inference_log").fetchone()
        return row["n"] if row else 0


def get_uncollected_corrections(min_count: int = 30) -> list[dict]:
    """
    Return correction rows not yet used in a fine-tune run.

    'Not yet used' = id is greater than last_feedback_id of the currently
    deployed model in model_registry (i.e., corrections logged after the
    last deployment).
    """
    with get_db() as db:
        rows = db.execute(
            """
            SELECT * FROM feedback
            WHERE is_correction = 1
              AND id > (
                  SELECT COALESCE(MAX(last_feedback_id), 0)
                  FROM model_registry
                  WHERE deployed = 1
              )
            ORDER BY id ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_current_registry_entry() -> dict:
    """Return the currently deployed model's registry row (or safe defaults)."""
    with get_db() as db:
        row = db.execute(
            """
            SELECT * FROM model_registry
            WHERE deployed = 1
            ORDER BY deployed_at DESC
            LIMIT 1
            """
        ).fetchone()
        if row:
            return dict(row)
        return {
            "version": "v0.0",
            "trained_on_n": 0,
            "auc_roc": 0.0,
            "f1_score": 0.0,
            "last_feedback_id": 0,
            "deployed": 1,
            "deployed_at": None,
        }


def get_all_registry_entries() -> list[dict]:
    """Return full model version history, newest first."""
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM model_registry ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def register_model(
    version: str,
    trained_on_n: int,
    auc_roc: float,
    f1_score: float,
    last_feedback_id: int,
):
    """Add a model version to the registry (not yet deployed)."""
    with get_db() as db:
        db.execute(
            """
            INSERT OR REPLACE INTO model_registry
                (version, trained_on_n, auc_roc, f1_score, last_feedback_id, deployed)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (version, trained_on_n, auc_roc, f1_score, last_feedback_id),
        )


def mark_deployed(version: str):
    """Flip the deployed flag for a version; unset all others."""
    with get_db() as db:
        db.execute("UPDATE model_registry SET deployed = 0")
        db.execute(
            """
            UPDATE model_registry
            SET deployed = 1, deployed_at = CURRENT_TIMESTAMP
            WHERE version = ?
            """,
            (version,),
        )


def seed_initial_registry(version: str, auc_roc: float, f1_score: float):
    """
    Called once at server startup if no registry entry exists.
    Ensures the /flywheel endpoint always has real baseline numbers to show.
    """
    with get_db() as db:
        exists = db.execute(
            "SELECT 1 FROM model_registry WHERE version = ?", (version,)
        ).fetchone()
        if not exists:
            db.execute(
                """
                INSERT INTO model_registry
                    (version, trained_on_n, auc_roc, f1_score,
                     last_feedback_id, deployed, deployed_at)
                VALUES (?, 0, ?, ?, 0, 1, CURRENT_TIMESTAMP)
                """,
                (version, auc_roc, f1_score),
            )
