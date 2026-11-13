import argparse
import logging
from dataclasses import asdict
from pathlib import Path

from .adapters import AlgorithmConfig, LegacyRegistrar, SahiDetector
from .core import InspectionError, OfflineCore, PersistenceError, Task
from .pipeline import DetectionPipeline
from .repository import SQLiteResultRepository


def main(argv=None):
    parser = argparse.ArgumentParser(description="AOI next: independent offline inspection")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crop-top", type=int, default=9070)
    parser.add_argument("--sample-width-ratio", type=float, default=1.0)
    parser.add_argument("--slice-size", type=int, default=1200)
    parser.add_argument("--overlap", type=float, default=0.05)
    parser.add_argument("--confidence", type=float, default=0.2)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--max-sampling-per-grid", type=int, default=100)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        config = AlgorithmConfig(
            weights=str(Path(args.weights).resolve()), device=args.device,
            crop_top=args.crop_top, sample_width_ratio=args.sample_width_ratio,
            slice_size=args.slice_size, overlap=args.overlap, confidence=args.confidence,
            grid_size=args.grid_size, max_sampling_per_grid=args.max_sampling_per_grid,
        )
        parameters = {key: str(value) for key, value in asdict(config).items()}
        parameters["coordinate_space"] = "registered_overlay_pixels"
        parameters["bbox_format"] = "xyxy"
        parameters["algorithm"] = "legacy_drew_aliked_lightglue_sahi_v1"
        task = Task(
            args.task_id, str(Path(args.sample).resolve()),
            str(Path(args.template).resolve()), tuple(parameters.items()),
        )
        pipeline = DetectionPipeline(LegacyRegistrar(config), SahiDetector(config))
        with SQLiteResultRepository(args.database) as repository:
            result = OfflineCore(pipeline, repository).run(task)
        logging.info("run_id=%s status=%s defects=%d", result.run_id, result.status.value, len(result.defects))
        return 0
    except InspectionError as error:
        logging.error("run_id=%s status=failed: %s", error.result.run_id, error.result.error_message)
        return 1
    except PersistenceError as error:
        logging.error("Result was not saved: %s", error.storage_error)
        return 2
    except Exception as error:
        logging.error("Cannot start offline inspection: %s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
