import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from .pipeline import ImageDetector, Registrar, RegisteredImage, coco_to_defects


@dataclass(frozen=True)
class AlgorithmConfig:
    weights: str
    device: str = "cpu"
    crop_top: int = 9070
    sample_width_ratio: float = 1.0
    slice_size: int = 1200
    overlap: float = 0.05
    confidence: float = 0.2
    grid_size: int = 7
    max_sampling_per_grid: int = 100

    def __post_init__(self):
        if not self.weights or not self.device:
            raise ValueError("weights and device are required")
        for name in ("crop_top", "slice_size", "grid_size", "max_sampling_per_grid"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.crop_top < 0 or min(self.slice_size, self.grid_size, self.max_sampling_per_grid) <= 0:
            raise ValueError("Invalid image or grid dimensions")
        for name in ("sample_width_ratio", "overlap", "confidence"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.sample_width_ratio <= 1:
            raise ValueError("sample_width_ratio must be in (0, 1]")
        if not 0 <= self.overlap < 1 or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid overlap or confidence")


def local_sahi():
    package_root = Path(__file__).resolve().parents[1] / "DefectDetection" / "sahi"
    expected = (package_root / "sahi" / "__init__.py").resolve()
    if not expected.is_file():
        raise RuntimeError("Repository SAHI implementation is missing")
    loaded = sys.modules.get("sahi")
    if loaded is not None and Path(loaded.__file__).resolve() != expected:
        raise RuntimeError("Another SAHI package is loaded; use a fresh aoi_next process")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    module = importlib.import_module("sahi")
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError("SAHI package resolution mismatch")
    return module


class MatchingModels:
    MODEL_TYPE_ALIKED = "aliked"
    MODEL_TYPE_LIGHTGLUE = "lightglue"

    def __init__(self, device):
        self.device = device
        self.cache = {}

    def get_model(self, model_type, **kwargs):
        key = (model_type, tuple(sorted(kwargs.items())))
        if key not in self.cache:
            import torch
            from TemplateMatching.EfficientLoFTR.lightglue import ALIKED, LightGlue

            device = torch.device(self.device)
            if model_type == self.MODEL_TYPE_ALIKED:
                model = ALIKED(max_num_keypoints=kwargs.get("max_num_keypoints", 6144))
            elif model_type == self.MODEL_TYPE_LIGHTGLUE:
                model = LightGlue(
                    features=kwargs.get("features", "aliked"),
                    depth_confidence=kwargs.get("depth_confidence", -1),
                    width_confidence=kwargs.get("width_confidence", -1),
                )
            else:
                raise ValueError(f"Unsupported matching model: {model_type}")
            self.cache[key] = (model.eval().to(device), device)
        return self.cache[key]


class LegacyRegistrar(Registrar):
    def __init__(self, config, matching_function=None, models=None):
        self.config = config
        self.matching_function = matching_function
        self.models = models if models is not None else MatchingModels(config.device)

    def register(self, task):
        for filename in (task.template_path, task.sample_path):
            if not Path(filename).is_file():
                raise FileNotFoundError(filename)
        import numpy as np
        from PIL import Image

        with Image.open(task.template_path) as source:
            template = np.asarray(source.convert("L"), dtype=np.float32)
        with Image.open(task.sample_path) as source:
            width, height = source.size
            crop_width = int(width * self.config.sample_width_ratio)
            if height <= self.config.crop_top or crop_width < 1:
                raise ValueError("Sample is empty after configured cropping")
            sample = np.asarray(source.convert("L").crop((
                width - crop_width, self.config.crop_top, width, height
            )), dtype=np.float32)
        std = float(template.std())
        if std <= 0 or not math.isfinite(std):
            raise ValueError("Template has no usable intensity variation")
        template = np.clip(
            (template - template.mean()) / std * sample.std() + sample.mean(), 0, 255
        ).astype(np.uint8)
        if self.matching_function is None:
            local_sahi()
            from TemplateMatching.EfficientLoFTR.fine_matching_loftr_new_drew_test import fine_matching_pillow_on_template_drew_return_bgr

            self.matching_function = fine_matching_pillow_on_template_drew_return_bgr
        overlay = self.matching_function(
            sample.astype(np.uint8), template,
            grid_size=self.config.grid_size,
            max_sampling_per_grid=self.config.max_sampling_per_grid,
            enhance_algorithms="clahe",
            enhance_param=[50, 100, "None"],
            extractor_name="aliked", model=self.models,
        )
        if not isinstance(overlay, np.ndarray) or overlay.ndim != 3 or overlay.shape[2] != 3:
            raise ValueError("Registration must return a BGR image")
        if overlay.dtype != np.uint8 or min(overlay.shape[:2]) < 1:
            raise ValueError("Registration returned invalid pixels")
        return RegisteredImage(overlay, overlay.shape[1], overlay.shape[0])


class SahiDetector(ImageDetector):
    def __init__(self, config, model=None, predict=None):
        self.config = config
        self.model = model
        self.predict_function = predict

    def _load(self):
        if self.model is not None and self.predict_function is not None:
            return
        if not Path(self.config.weights).is_file():
            raise FileNotFoundError(self.config.weights)
        local_sahi()
        from sahi.models.ultralytics import UltralyticsDetectionModel
        from sahi.predict import get_batched_sliced_prediction

        class ThresholdDetectionModel(UltralyticsDetectionModel):
            def perform_batched_inference(self, images, images_path=None):
                if not images:
                    raise ValueError("Expected non-empty RGB slices")
                results = self.model.predict(
                    source=[image[:, :, ::-1] for image in images],
                    stream=True, verbose=False, conf=self.confidence_threshold,
                    device=self.device,
                )
                self._original_predictions = [result.boxes.data for result in results]

        self.model = ThresholdDetectionModel(
            model_path=self.config.weights,
            confidence_threshold=self.config.confidence,
            device=self.config.device,
        )
        if self.model.has_mask or self.model.is_obb:
            raise ValueError("Only axis-aligned detection weights are supported")
        self.predict_function = get_batched_sliced_prediction

    def predict(self, image):
        self._load()
        split = image.height // 2
        regions = [(0, image.height)] if split == 0 else [(0, split), (split, image.height)]
        defects = []
        for start, end in regions:
            rgb = image.pixels[start:end, :, ::-1].copy()
            prediction = self.predict_function(
                rgb, self.model,
                slice_height=self.config.slice_size, slice_width=self.config.slice_size,
                overlap_height_ratio=self.config.overlap,
                overlap_width_ratio=self.config.overlap,
                visualize_slices=False, verbose=0,
            )
            defects.extend(coco_to_defects(prediction.to_coco_predictions(image_id=1), start))
        return tuple(defects)
