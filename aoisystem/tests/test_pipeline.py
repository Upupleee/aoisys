import copy
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aoi_next.adapters import AlgorithmConfig, LegacyRegistrar, SahiDetector
from aoi_next.core import Defect, InspectionError, OfflineCore, ResultStatus, Task
from aoi_next.pipeline import DetectionPipeline, ImageDetector, Registrar, RegisteredImage, coco_to_defects
from aoi_next.repository import SQLiteResultRepository


class Pixels:
    def __init__(self, height=5, width=10, start=0):
        self.height = height
        self.width = width
        self.start = start

    def __getitem__(self, slices):
        rows, columns, channels = slices
        if channels.step != -1:
            raise AssertionError("BGR must be converted to RGB")
        return Pixels(rows.stop - rows.start, self.width, rows.start)

    def copy(self):
        return Pixels(self.height, self.width, self.start)


class Prediction:
    def __init__(self, data):
        self.data = data

    def to_coco_predictions(self, image_id):
        return self.data


class FixedRegistrar(Registrar):
    def register(self, task):
        return RegisteredImage(Pixels(), 10, 5)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.task = Task("test", "sample.tif", "template.tif")

    def test_coco_offset_preserves_height_and_input(self):
        predictions = [{"bbox": [1, 2, 3, 4], "category_id": 7, "score": 0.9}]
        original = copy.deepcopy(predictions)
        defect, = coco_to_defects(predictions, 9)
        self.assertEqual(defect.bbox, (1, 11, 4, 15))
        self.assertEqual(predictions, original)

    def test_empty_predictions(self):
        self.assertEqual(coco_to_defects([], 10), ())

    def test_invalid_offset(self):
        with self.assertRaises(ValueError):
            coco_to_defects([], -1)

    def test_invalid_box(self):
        with self.assertRaises(ValueError):
            coco_to_defects([{"bbox": [1, 2, -1, 4], "category_id": 7, "score": 0.9}])

    def test_halves_and_parameters(self):
        calls = []
        model = object()

        def predict(pixels, detector, **kwargs):
            calls.append((pixels.height, pixels.start, detector, kwargs))
            return Prediction([{"bbox": [1, 0, 2, 1], "category_id": 1, "score": 0.8}])

        config = AlgorithmConfig("unused.pt", slice_size=640, overlap=0.1)
        detector = SahiDetector(config, model=model, predict=predict)
        pipeline = DetectionPipeline(FixedRegistrar(), detector)
        with SQLiteResultRepository(":memory:") as repository:
            result = OfflineCore(pipeline, repository).run(self.task)
            self.assertEqual(result.status, ResultStatus.SUCCEEDED)
            self.assertEqual([d.bbox for d in result.defects], [(1, 0, 3, 1), (1, 2, 3, 3)])
            self.assertEqual(repository.get(result.run_id), result)
        self.assertEqual([call[:2] for call in calls], [(2, 0), (3, 2)])
        self.assertTrue(all(call[2] is model for call in calls))
        for call in calls:
            self.assertEqual(call[3]["slice_height"], 640)
            self.assertEqual(call[3]["overlap_width_ratio"], 0.1)
            self.assertFalse(call[3]["visualize_slices"])

    def test_one_row_is_not_split_into_empty_region(self):
        heights = []

        def predict(pixels, detector, **kwargs):
            heights.append(pixels.height)
            return Prediction([])

        detector = SahiDetector(AlgorithmConfig("unused.pt"), object(), predict)
        self.assertEqual(detector.predict(RegisteredImage(Pixels(1), 10, 1)), ())
        self.assertEqual(heights, [1])

    def test_missing_image_is_recorded_without_loading_algorithms(self):
        pipeline = DetectionPipeline(LegacyRegistrar(AlgorithmConfig("unused.pt")), None)
        with tempfile.TemporaryDirectory() as directory:
            task = Task("missing", str(Path(directory) / "missing"), str(Path(directory) / "template"))
            with SQLiteResultRepository(":memory:") as repository:
                with self.assertRaises(InspectionError):
                    OfflineCore(pipeline, repository).run(task)
                self.assertEqual(repository.list_results()[0].status, ResultStatus.FAILED)

    def test_out_of_bounds_is_failure(self):
        class BadDetector(ImageDetector):
            def predict(self, image):
                return [Defect("1", 0.5, (0, 0, 20, 20))]

        with self.assertRaises(ValueError):
            DetectionPipeline(FixedRegistrar(), BadDetector()).detect(self.task)

    def test_config_rejects_invalid_values(self):
        for kwargs in ({"crop_top": -1}, {"slice_size": 0}, {"confidence": float("nan")},
                       {"overlap": 1}, {"sample_width_ratio": 0}, {"grid_size": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AlgorithmConfig("model.pt", **kwargs)

    def test_cli_help_has_no_heavy_imports(self):
        module = importlib.import_module("aoi_next.__main__")
        with self.assertRaises(SystemExit) as context:
            module.main(["--help"])
        self.assertEqual(context.exception.code, 0)
        for name in ("torch", "cv2", "PyQt5", "numpy"):
            self.assertNotIn(name, sys.modules)

    def test_batch_confidence_reaches_model(self):
        calls = []

        class Model:
            def predict(self, **kwargs):
                calls.append(kwargs)
                return [SimpleNamespace(boxes=SimpleNamespace(data="boxes"))]

        class BaseModel:
            def __init__(self, model_path, confidence_threshold, device):
                self.confidence_threshold = confidence_threshold
                self.device = device
                self.model = Model()
                self.has_mask = False
                self.is_obb = False

        class Slice:
            def __getitem__(self, value):
                return "bgr"

        modules = {
            "sahi.models.ultralytics": SimpleNamespace(UltralyticsDetectionModel=BaseModel),
            "sahi.predict": SimpleNamespace(get_batched_sliced_prediction=object()),
        }
        detector = SahiDetector(AlgorithmConfig("weights.pt", confidence=0.12, device="cuda:0"))
        with patch.dict(sys.modules, modules), patch("aoi_next.adapters.local_sahi"), patch.object(Path, "is_file", return_value=True):
            detector._load()
            detector.model.perform_batched_inference([Slice()])
        self.assertEqual(calls[0]["conf"], 0.12)
        self.assertEqual(calls[0]["device"], "cuda:0")
        self.assertEqual(calls[0]["source"], ["bgr"])
        self.assertEqual(detector.model._original_predictions, ["boxes"])

    def test_cli_records_failure_and_returns_nonzero(self):
        from aoi_next.__main__ import main

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "runs.sqlite3"
            code = main([
                "--sample", str(Path(directory) / "missing.tif"),
                "--template", str(Path(directory) / "template.tif"),
                "--weights", "missing.pt", "--database", str(database), "--task-id", "cli-test",
            ])
            self.assertEqual(code, 1)
            with SQLiteResultRepository(database) as repository:
                record, = repository.list_results()
                self.assertEqual(record.status, ResultStatus.FAILED)
                self.assertEqual(dict(record.task.parameters)["bbox_format"], "xyxy")


if __name__ == "__main__":
    unittest.main()
