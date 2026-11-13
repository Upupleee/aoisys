from __future__ import annotations

import sqlite3
from datetime import datetime
from os import PathLike
from threading import RLock
from typing import Optional, Tuple, Union

from .core import Defect, Result, ResultRepository, ResultStatus, Task


class SQLiteResultRepository(ResultRepository):
    def __init__(self, database: Union[str, PathLike], timeout: float = 5.0) -> None:
        self._lock = RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            database, timeout=timeout, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            with self._connection:
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS aoi_runs ("
                    "run_id TEXT PRIMARY KEY NOT NULL, task_id TEXT NOT NULL, "
                    "sample_path TEXT NOT NULL, template_path TEXT NOT NULL, "
                    "status TEXT NOT NULL CHECK(status IN ('succeeded', 'failed')), "
                    "started_at TEXT NOT NULL, finished_at TEXT NOT NULL, "
                    "error_type TEXT, error_message TEXT, "
                    "CHECK((status = 'succeeded' AND error_type IS NULL "
                    "AND error_message IS NULL) OR (status = 'failed' "
                    "AND error_type IS NOT NULL AND error_message IS NOT NULL)))"
                )
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS aoi_parameters ("
                    "run_id TEXT NOT NULL REFERENCES aoi_runs(run_id), "
                    "position INTEGER NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL, "
                    "PRIMARY KEY(run_id, position), UNIQUE(run_id, name))"
                )
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS aoi_defects ("
                    "run_id TEXT NOT NULL REFERENCES aoi_runs(run_id), "
                    "position INTEGER NOT NULL, label TEXT NOT NULL, "
                    "score REAL NOT NULL CHECK(score >= 0 AND score <= 1), "
                    "x_min REAL NOT NULL CHECK(x_min >= 0), "
                    "y_min REAL NOT NULL CHECK(y_min >= 0), "
                    "x_max REAL NOT NULL CHECK(x_max > x_min), "
                    "y_max REAL NOT NULL CHECK(y_max > y_min), "
                    "PRIMARY KEY(run_id, position))"
                )
                self._connection.execute(
                    "CREATE INDEX IF NOT EXISTS aoi_runs_task "
                    "ON aoi_runs(task_id, started_at, run_id)"
                )
        except Exception:
            self._connection.close()
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Repository is closed")

    def save(self, result: Result) -> None:
        if not isinstance(result, Result):
            raise TypeError("result must be a Result")
        with self._lock:
            self._ensure_open()
            with self._connection:
                self._connection.execute(
                    "INSERT INTO aoi_runs "
                    "(run_id, task_id, sample_path, template_path, status, "
                    "started_at, finished_at, error_type, error_message) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        result.run_id, result.task.task_id, result.task.sample_path,
                        result.task.template_path, result.status.value,
                        result.started_at.isoformat(), result.finished_at.isoformat(),
                        result.error_type, result.error_message,
                    ),
                )
                self._connection.executemany(
                    "INSERT INTO aoi_parameters (run_id, position, name, value) "
                    "VALUES (?, ?, ?, ?)",
                    ((result.run_id, i, key, value)
                     for i, (key, value) in enumerate(result.task.parameters)),
                )
                self._connection.executemany(
                    "INSERT INTO aoi_defects "
                    "(run_id, position, label, score, x_min, y_min, x_max, y_max) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ((result.run_id, i, defect.label, defect.score, *defect.bbox)
                     for i, defect in enumerate(result.defects)),
                )

    def _decode(self, row: sqlite3.Row) -> Result:
        parameters = self._connection.execute(
            "SELECT name, value FROM aoi_parameters WHERE run_id = ? ORDER BY position",
            (row["run_id"],),
        ).fetchall()
        defects = self._connection.execute(
            "SELECT label, score, x_min, y_min, x_max, y_max FROM aoi_defects "
            "WHERE run_id = ? ORDER BY position", (row["run_id"],),
        ).fetchall()
        return Result(
            run_id=row["run_id"],
            task=Task(
                row["task_id"], row["sample_path"], row["template_path"],
                tuple((item["name"], item["value"]) for item in parameters),
            ),
            status=ResultStatus(row["status"]),
            started_at=datetime.fromisoformat(row["started_at"]),
            finished_at=datetime.fromisoformat(row["finished_at"]),
            defects=tuple(
                Defect(item["label"], item["score"], tuple(item)[2:])
                for item in defects
            ),
            error_type=row["error_type"],
            error_message=row["error_message"],
        )

    def get(self, run_id: str) -> Optional[Result]:
        with self._lock:
            self._ensure_open()
            with self._connection:
                self._connection.execute("BEGIN")
                row = self._connection.execute(
                    "SELECT * FROM aoi_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                return None if row is None else self._decode(row)

    def list_results(
        self, task_id: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> Tuple[Result, ...]:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        query = "SELECT * FROM aoi_runs"
        values = ()
        if task_id is not None:
            query += " WHERE task_id = ?"
            values = (task_id,)
        query += " ORDER BY started_at, run_id LIMIT ? OFFSET ?"
        with self._lock:
            self._ensure_open()
            with self._connection:
                self._connection.execute("BEGIN")
                rows = self._connection.execute(query, (*values, limit, offset)).fetchall()
                return tuple(self._decode(row) for row in rows)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> SQLiteResultRepository:
        with self._lock:
            self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
