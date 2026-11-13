from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable

from .core import Defect, Detector, Task


@dataclass(frozen=True)
class RegisteredImage:
    pixels: Any
    width: int
    height: int
    coordinate_space: str = "registered_overlay_pixels"

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Registered image must not be empty")


class Registrar(ABC):
    @abstractmethod
    def register(self, task: Task) -> RegisteredImage:
        raise NotImplementedError


class ImageDetector(ABC):
    @abstractmethod
    def predict(self, image: RegisteredImage) -> Iterable[Defect]:
        raise NotImplementedError


class DetectionPipeline(Detector):
    def __init__(self, registrar: Registrar, detector: ImageDetector):
        self.registrar = registrar
        self.detector = detector

    def detect(self, task: Task) -> Iterable[Defect]:
        image = self.registrar.register(task)
        defects = tuple(self.detector.predict(image))
        for defect in defects:
            if defect.bbox[2] > image.width or defect.bbox[3] > image.height:
                raise ValueError("Detection exceeds registered overlay bounds")
        return defects


def coco_to_defects(predictions, y_offset=0):
    if y_offset < 0:
        raise ValueError("y_offset must be non-negative")
    result = []
    for prediction in predictions:
        x, y, width, height = prediction["bbox"]
        result.append(Defect(
            label=str(prediction["category_id"]),
            score=prediction["score"],
            bbox=(x, y + y_offset, x + width, y + y_offset + height),
        ))
    return tuple(result)
