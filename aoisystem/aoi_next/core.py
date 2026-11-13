from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Optional, Tuple
from uuid import uuid4


logger = logging.getLogger(__name__)


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


@dataclass(frozen=True)
class Task:
    task_id: str
    sample_path: str
    template_path: str
    parameters: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in ("task_id", "sample_path", "template_path"):
            _text(getattr(self, name), name)
        parameters = []
        keys = set()
        for pair in self.parameters:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("parameters must contain key/value pairs")
            key, value = pair
            _text(key, "parameter key")
            if not isinstance(value, str):
                raise ValueError("parameter values must be strings")
            if key in keys:
                raise ValueError(f"Duplicate parameter: {key}")
            keys.add(key)
            parameters.append((key, value))
        object.__setattr__(self, "parameters", tuple(parameters))


@dataclass(frozen=True)
class Defect:
    label: str
    score: float
    bbox: Tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _text(self.label, "label")
        score = _number(self.score, "score")
        if not 0 <= score <= 1:
            raise ValueError("score must be between 0 and 1")
        bbox = tuple(_number(value, "bbox coordinate") for value in self.bbox)
        if len(bbox) != 4:
            raise ValueError("bbox must contain x_min, y_min, x_max, y_max")
        x_min, y_min, x_max, y_max = bbox
        if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min:
            raise ValueError("bbox must have non-negative origin and positive area")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "bbox", bbox)


class ResultStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class Result:
    run_id: str
    task: Task
    status: ResultStatus
    started_at: datetime
    finished_at: datetime
    defects: Tuple[Defect, ...] = ()
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.run_id, "run_id")
        if not isinstance(self.task, Task):
            raise TypeError("task must be a Task")
        if not isinstance(self.status, ResultStatus):
            raise TypeError("status must be a ResultStatus")
        for name in ("started_at", "finished_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ValueError(f"{name} must be a timezone-aware datetime")
            object.__setattr__(self, name, value.astimezone(timezone.utc))
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        defects = tuple(self.defects)
        if any(not isinstance(defect, Defect) for defect in defects):
            raise TypeError("defects must contain only Defect instances")
        object.__setattr__(self, "defects", defects)
        if self.status is ResultStatus.SUCCEEDED:
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("Successful results must not contain errors")
        else:
            _text(self.error_type, "error_type")
            if not isinstance(self.error_message, str):
                raise ValueError("Failed results require an error_message")
            if defects:
                raise ValueError("Failed results must not contain partial defects")


class Detector(ABC):
    @abstractmethod
    def detect(self, task: Task) -> Iterable[Defect]:
        raise NotImplementedError


class ResultRepository(ABC):
    @abstractmethod
    def save(self, result: Result) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, run_id: str) -> Optional[Result]:
        raise NotImplementedError

    @abstractmethod
    def list_results(
        self, task_id: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> Tuple[Result, ...]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class InspectionError(RuntimeError):
    def __init__(self, result: Result, cause: Exception) -> None:
        self.result = result
        self.cause = cause
        super().__init__(f"Inspection failed for run {result.run_id}: {result.error_type}")


class PersistenceError(RuntimeError):
    def __init__(
        self,
        result: Result,
        storage_error: Exception,
        inspection_error: Optional[Exception] = None,
    ) -> None:
        self.result = result
        self.storage_error = storage_error
        self.inspection_error = inspection_error
        super().__init__(f"Could not persist run {result.run_id}")


class OfflineCore:
    def __init__(self, detector: Detector, repository: ResultRepository) -> None:
        self._detector = detector
        self._repository = repository

    def run(self, task: Task) -> Result:
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        run_id = uuid4().hex
        started_at = datetime.now(timezone.utc)
        try:
            defects = tuple(self._detector.detect(task))
            result = Result(
                run_id=run_id,
                task=task,
                status=ResultStatus.SUCCEEDED,
                started_at=started_at,
                finished_at=max(started_at, datetime.now(timezone.utc)),
                defects=defects,
            )
        except Exception as error:
            result = Result(
                run_id=run_id,
                task=task,
                status=ResultStatus.FAILED,
                started_at=started_at,
                finished_at=max(started_at, datetime.now(timezone.utc)),
                error_type=f"{type(error).__module__}.{type(error).__qualname__}",
                error_message=str(error),
            )
            logger.error("Inspection failed for run %s (%s)", run_id, result.error_type)
            self._save(result, error)
            raise InspectionError(result, error) from error
        self._save(result)
        return result

    def _save(self, result: Result, error: Optional[Exception] = None) -> None:
        try:
            self._repository.save(result)
        except Exception as storage_error:
            logger.error("Persistence failed for run %s", result.run_id)
            raise PersistenceError(result, storage_error, error) from storage_error
