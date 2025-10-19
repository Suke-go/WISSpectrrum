"""SQLite-backed job state management for the preprocessing pipeline."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

VALID_STATUSES = {STATUS_QUEUED, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    job_id: int
    pdf_path: Path
    summary_path: Path
    base_root: Optional[Path]
    relative_pdf: Optional[Path]
    status: str
    attempts: int
    last_error: Optional[str]
    metadata: Dict[str, object]


class PipelineState:
    """Lightweight job store with basic enqueue/run/fail bookkeeping."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._initialise()

    def close(self) -> None:
        self._conn.close()

    def _initialise(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_path TEXT UNIQUE NOT NULL,
                    summary_path TEXT NOT NULL,
                    base_root TEXT,
                    relative_pdf TEXT,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        self._ensure_columns()

    def _ensure_columns(self) -> None:
        """Backfill optional columns for legacy databases."""
        with self._conn:
            existing = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "base_root" not in existing:
                self._conn.execute("ALTER TABLE jobs ADD COLUMN base_root TEXT")
            if "relative_pdf" not in existing:
                self._conn.execute("ALTER TABLE jobs ADD COLUMN relative_pdf TEXT")

    def enqueue(
        self,
        pdf_path: Path,
        summary_path: Path,
        *,
        base_root: Optional[Path] = None,
        relative_pdf: Optional[Path] = None,
        metadata: Optional[Dict[str, object]] = None,
        force: bool = False,
    ) -> int:
        """Insert or requeue a job. Returns the job id."""
        pdf = str(pdf_path.expanduser().resolve())
        summary = str(summary_path.expanduser().resolve())
        base_root_str = str(base_root.expanduser().resolve()) if base_root else None
        relative_pdf_str = str(relative_pdf).replace("\\", "/") if relative_pdf else None
        payload = json.dumps(metadata or {}, ensure_ascii=False)
        now = utc_now()

        with self._conn:
            existing = self._conn.execute(
                "SELECT id, status FROM jobs WHERE pdf_path = ?", (pdf,)
            ).fetchone()
            if existing:
                job_id = int(existing["id"])
                if force or existing["status"] in (STATUS_FAILED, STATUS_DONE):
                    self._conn.execute(
                        """
                        UPDATE jobs
                           SET status = ?,
                               summary_path = ?,
                               metadata = ?,
                               attempts = CASE WHEN ? THEN 0 ELSE attempts END,
                               last_error = NULL,
                               base_root = COALESCE(?, base_root),
                               relative_pdf = COALESCE(?, relative_pdf),
                               updated_at = ?
                         WHERE id = ?
                        """,
                        (
                            STATUS_QUEUED,
                            summary,
                            payload,
                            int(force),
                            base_root_str,
                            relative_pdf_str,
                            now,
                            job_id,
                        ),
                    )
                return job_id

            cursor = self._conn.execute(
                """
                INSERT INTO jobs (pdf_path, summary_path, base_root, relative_pdf, status, metadata, attempts, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (pdf, summary, base_root_str, relative_pdf_str, STATUS_QUEUED, payload, now, now),
            )
            return int(cursor.lastrowid)

    def fetch(self, job_id: int) -> Optional[Job]:
        row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def next_queued(self) -> Optional[Job]:
        row = self._conn.execute(
            """
            SELECT * FROM jobs
             WHERE status = ?
             ORDER BY created_at ASC
             LIMIT 1
            """,
            (STATUS_QUEUED,),
        ).fetchone()
        return self._row_to_job(row) if row else None

    def mark_running(self, job_id: int) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE jobs
                   SET status = ?,
                       attempts = attempts + 1,
                       updated_at = ?
                 WHERE id = ?
                """,
                (STATUS_RUNNING, utc_now(), job_id),
            )

    def mark_done(self, job_id: int) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE jobs
                   SET status = ?,
                       last_error = NULL,
                       updated_at = ?
                 WHERE id = ?
                """,
                (STATUS_DONE, utc_now(), job_id),
            )

    def mark_failed(self, job_id: int, error: str) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE jobs
                   SET status = ?,
                       last_error = ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (STATUS_FAILED, error, utc_now(), job_id),
            )

    def list_jobs(self, statuses: Optional[Iterable[str]] = None) -> Iterator[Job]:
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query = f"SELECT * FROM jobs WHERE status IN ({placeholders}) ORDER BY created_at ASC"
            params = tuple(statuses)
        else:
            query = "SELECT * FROM jobs ORDER BY created_at ASC"
            params = ()
        cursor = self._conn.execute(query, params)
        for row in cursor.fetchall():
            job = self._row_to_job(row)
            if job:
                yield job

    @staticmethod
    def _row_to_job(row) -> Optional[Job]:
        if not row:
            return None
        metadata_raw = row["metadata"] or "{}"
        metadata = json.loads(metadata_raw)
        return Job(
            job_id=int(row["id"]),
            pdf_path=Path(row["pdf_path"]),
            summary_path=Path(row["summary_path"]),
            base_root=Path(row["base_root"]) if row["base_root"] else None,
            relative_pdf=Path(row["relative_pdf"]) if row["relative_pdf"] else None,
            status=row["status"],
            attempts=int(row["attempts"]),
            last_error=row["last_error"],
            metadata=metadata,
        )

    def purge_missing_jobs(self) -> int:
        """
        Remove jobs whose PDF no longer exists on disk.

        Returns number of deleted rows.
        """
        with self._conn:
            cursor = self._conn.execute(
                "SELECT id, pdf_path FROM jobs"
            )
            victims = [
                int(row["id"])
                for row in cursor.fetchall()
                if row["pdf_path"] and not Path(row["pdf_path"]).exists()
            ]
            if not victims:
                return 0
            self._conn.executemany("DELETE FROM jobs WHERE id = ?", [(vid,) for vid in victims])
            self._conn.execute("VACUUM")
            return len(victims)


__all__ = [
    "PipelineState",
    "Job",
    "STATUS_QUEUED",
    "STATUS_RUNNING",
    "STATUS_DONE",
    "STATUS_FAILED",
]

