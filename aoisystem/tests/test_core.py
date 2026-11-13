import ast
import io
import sqlite3
import tokenize
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from aoi_next.core import (
    Defect, Detector, InspectionError, OfflineCore, PersistenceError,
    Result, ResultRepository, ResultStatus, Task,
)
from aoi_next.repository import SQLiteResultRepository


class StubDetector(Detector):
    def __init__(self, defects=(), error=None):
        self.defects = defects
        self.error = error
        self.tasks = []

    def detect(self, task):
        self.tasks.append(task)
        if self.error is not None:
            raise self.error
        return self.defects


class BrokenRepository(ResultRepository):
    def save(self, result):
        raise OSError("storage unavailable")

    def get(self, run_id):
        raise NotImplementedError

    def list_results(self, task_id=None, limit=100, offset=0):
        raise NotImplementedError

    def close(self):
        pass


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.repository = SQLiteResultRepository(":memory:")
        self.addCleanup(self.repository.close)
        self.task = Task("board-1", "样本.tif", "模板.tif", (("threshold", "0.5"),))
        self.defect = Defect("短路", 0.95, (1, 2, 10, 20))

    def run_core(self, detector, repository=None):
        return OfflineCore(
            detector, self.repository if repository is None else repository
        ).run(self.task)

    def test_success_persists_and_preserves_order(self):
        defects = (self.defect, Defect("断路", 0.8, (0, 0, 5, 5)))
        detector = StubDetector(iter(defects))
        result = self.run_core(detector)
        self.assertEqual(detector.tasks, [self.task])
        self.assertEqual(result.status, ResultStatus.SUCCEEDED)
        self.assertEqual(result.defects, defects)
        self.assertEqual(self.repository.get(result.run_id), result)
        self.assertEqual(self.repository.list_results(), (result,))
        self.assertIsNotNone(result.started_at.utcoffset())

    def test_no_defects_is_success(self):
        result = self.run_core(StubDetector())
        self.assertEqual(result.status, ResultStatus.SUCCEEDED)
        self.assertEqual(result.defects, ())

    def test_failure_is_persisted_before_exception(self):
        error = FileNotFoundError("missing sample")
        with self.assertLogs("aoi_next.core", level="ERROR"):
            with self.assertRaises(InspectionError) as caught:
                self.run_core(StubDetector(error=error))
        failure = caught.exception
        self.assertIs(failure.__cause__, error)
        self.assertIs(failure.cause, error)
        self.assertEqual(failure.result.status, ResultStatus.FAILED)
        self.assertEqual(failure.result.error_type, "builtins.FileNotFoundError")
        self.assertEqual(failure.result.error_message, "missing sample")
        self.assertEqual(self.repository.get(failure.result.run_id), failure.result)

    def test_generator_failure_discards_partial_defects(self):
        def defects():
            yield self.defect
            raise RuntimeError("incomplete inference")

        with self.assertLogs("aoi_next.core", level="ERROR"):
            with self.assertRaises(InspectionError) as caught:
                self.run_core(StubDetector(defects()))
        result = self.repository.get(caught.exception.result.run_id)
        self.assertEqual(result.defects, ())
        self.assertEqual(result.status, ResultStatus.FAILED)

    def test_invalid_adapter_output_is_recorded(self):
        for output in (None, (object(),), "invalid"):
            with self.subTest(output=output):
                with self.assertLogs("aoi_next.core", level="ERROR"):
                    with self.assertRaises(InspectionError) as caught:
                        self.run_core(StubDetector(output))
                self.assertEqual(
                    self.repository.get(caught.exception.result.run_id).status,
                    ResultStatus.FAILED,
                )

    def test_storage_failure_never_returns_success(self):
        with self.assertLogs("aoi_next.core", level="ERROR"):
            with self.assertRaises(PersistenceError) as caught:
                self.run_core(StubDetector(), BrokenRepository())
        self.assertIsInstance(caught.exception.storage_error, OSError)
        self.assertIs(caught.exception.__cause__, caught.exception.storage_error)
        self.assertIsNone(caught.exception.inspection_error)
        self.assertEqual(caught.exception.result.status, ResultStatus.SUCCEEDED)

    def test_dual_failure_retains_both_errors(self):
        error = RuntimeError("inference failed")
        with self.assertLogs("aoi_next.core", level="ERROR"):
            with self.assertRaises(PersistenceError) as caught:
                self.run_core(StubDetector(error=error), BrokenRepository())
        self.assertIs(caught.exception.inspection_error, error)
        self.assertIsInstance(caught.exception.storage_error, OSError)
        self.assertEqual(caught.exception.result.status, ResultStatus.FAILED)

    def test_retry_creates_independent_history(self):
        first = self.run_core(StubDetector())
        second = self.run_core(StubDetector((self.defect,)))
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(len(self.repository.list_results(self.task.task_id)), 2)

    def test_models_copy_mutable_inputs(self):
        parameters = [["mode", "offline"]]
        bbox = [1, 2, 3, 4]
        task = Task("id", "sample", "template", parameters)
        defect = Defect("short", 1, bbox)
        defects = [defect]
        now = datetime.now(timezone.utc)
        result = Result("run", task, ResultStatus.SUCCEEDED, now, now, defects)
        parameters[0][1] = "changed"
        bbox[0] = 999
        defects.clear()
        self.assertEqual(task.parameters, (("mode", "offline"),))
        self.assertEqual(defect.bbox, (1, 2, 3, 4))
        self.assertEqual(result.defects, (defect,))
        for model, field, value in (
            (task, "task_id", "changed"), (defect, "score", 0),
            (result, "defects", ()),
        ):
            with self.assertRaises(FrozenInstanceError):
                setattr(model, field, value)
            hash(model)

    def test_invalid_tasks(self):
        for values in (
            ("", "sample", "template"), ("id", " ", "template"),
            ("id", "sample", ""), ("id", "sample", "template", (("x", []),)),
            ("id", "sample", "template", (("x", "1"), ("x", "2"))),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Task(*values)

    def test_invalid_defects(self):
        for score in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(score=score), self.assertRaises(ValueError):
                Defect("short", score, (0, 0, 1, 1))
        for bbox in ((0, 0, 0, 1), (-1, 0, 1, 1), (0, 1, 1, 0),
                     (0, 0, 1), (0, 0, float("inf"), 1)):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                Defect("short", 0.5, bbox)

    def test_result_invariants(self):
        result = self.run_core(StubDetector())
        for changes in (
            {"error_message": "unexpected"}, {"status": ResultStatus.FAILED},
            {"finished_at": result.started_at - timedelta(seconds=1)},
            {"started_at": datetime.now()},
            {"status": ResultStatus.FAILED, "error_type": "RuntimeError",
             "error_message": "", "defects": (self.defect,)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(result, **changes)

    def test_empty_exception_message_is_valid(self):
        with self.assertLogs("aoi_next.core", level="ERROR"):
            with self.assertRaises(InspectionError) as caught:
                self.run_core(StubDetector(error=RuntimeError()))
        self.assertEqual(self.repository.get(caught.exception.result.run_id).error_message, "")

    def test_abc_contracts(self):
        for cls in (Detector, ResultRepository):
            with self.assertRaises(TypeError):
                cls()

    def test_invalid_task_never_calls_adapter(self):
        detector = StubDetector()
        with self.assertRaises(TypeError):
            OfflineCore(detector, self.repository).run(None)
        self.assertEqual(detector.tasks, [])
        self.assertEqual(self.repository.list_results(), ())

    def test_repository_reopens_with_unicode_and_sql_characters(self):
        task = Task("'; DROP TABLE aoi_runs; --", "样本.tif", "模板.tif", (("x", "'"),))
        with TemporaryDirectory() as directory:
            database = Path(directory) / "results.sqlite3"
            with SQLiteResultRepository(database) as repository:
                result = OfflineCore(StubDetector((self.defect,)), repository).run(task)
            with SQLiteResultRepository(database) as repository:
                self.assertEqual(repository.get(result.run_id), result)
                self.assertEqual(repository.list_results(task.task_id), (result,))
                self.assertIsNone(repository.get("missing"))

    def test_duplicate_run_does_not_overwrite(self):
        result = self.run_core(StubDetector((self.defect,)))
        with self.assertRaises(sqlite3.IntegrityError):
            self.repository.save(replace(result, defects=()))
        self.assertEqual(self.repository.get(result.run_id), result)

    def test_transaction_rolls_back_all_rows(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "results.sqlite3"
            with SQLiteResultRepository(database) as repository:
                connection = sqlite3.connect(database)
                try:
                    with connection:
                        connection.execute(
                            "CREATE TRIGGER reject_defect BEFORE INSERT ON aoi_defects "
                            "BEGIN SELECT RAISE(ABORT, 'rejected'); END"
                        )
                    with self.assertLogs("aoi_next.core", level="ERROR"):
                        with self.assertRaises(PersistenceError):
                            self.run_core(StubDetector((self.defect,)), repository)
                    for table in ("aoi_runs", "aoi_parameters", "aoi_defects"):
                        self.assertEqual(
                            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0
                        )
                    self.assertEqual(repository.list_results(), ())
                finally:
                    connection.close()

    def test_filter_and_pagination(self):
        results = [self.run_core(StubDetector()) for _ in range(3)]
        results.sort(key=lambda result: (result.started_at, result.run_id))
        self.assertEqual(self.repository.list_results(limit=1, offset=1), (results[1],))
        self.assertEqual(self.repository.list_results("unknown"), ())
        for arguments in ({"limit": 0}, {"offset": -1}, {"limit": True}, {"offset": 1.5}):
            with self.assertRaises(ValueError):
                self.repository.list_results(**arguments)

    def test_equal_timestamps_have_stable_order(self):
        now = datetime.now(timezone.utc)
        for run_id in ("c", "a", "b"):
            self.repository.save(Result(
                run_id, self.task, ResultStatus.SUCCEEDED, now, now
            ))
        self.assertEqual(
            tuple(result.run_id for result in self.repository.list_results()),
            ("a", "b", "c"),
        )
        self.assertEqual(self.repository.list_results(limit=1, offset=1)[0].run_id, "b")

    def test_shared_repository_supports_worker_threads(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = tuple(pool.map(lambda _: self.run_core(StubDetector()), range(12)))
        self.assertEqual(len({result.run_id for result in results}), 12)
        self.assertEqual(set(self.repository.list_results()), set(results))

    def test_close_is_idempotent_and_operations_fail_explicitly(self):
        result = self.run_core(StubDetector())
        self.repository.close()
        self.repository.close()
        for operation in (
            lambda: self.repository.save(result), lambda: self.repository.get(result.run_id),
            self.repository.list_results, self.repository.__enter__,
        ):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                operation()

    def test_source_has_only_standard_library_imports_and_no_comments(self):
        root = Path(__file__).resolve().parents[1]
        allowed = {"__future__", "logging", "math", "abc", "dataclasses", "datetime",
                   "enum", "typing", "uuid", "sqlite3", "os", "threading"}
        for path in (root / "aoi_next/core.py", root / "aoi_next/repository.py"):
            source = path.read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        self.assertIn(name.name.split(".")[0], allowed)
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    self.assertIn(node.module.split(".")[0], allowed)
            self.assertFalse(any(
                token.type == tokenize.COMMENT
                for token in tokenize.generate_tokens(io.StringIO(source).readline)
            ))


if __name__ == "__main__":
    unittest.main()
