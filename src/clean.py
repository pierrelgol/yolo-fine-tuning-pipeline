from __future__ import annotations

from pathlib import Path

from src.common import remove_path
from src.config import AppConfig


def clean_artifacts(config: AppConfig) -> None:
    remove_path(config.paths.dataset_dir)
    remove_local_model_file(config)
    print(f"Removed artifacts under {config.paths.dataset_dir}")


def prune_artifacts(config: AppConfig) -> None:
    remove_path(config.paths.train_dir)
    remove_path(config.paths.eval_dir)
    remove_path(config.paths.infer_dir)
    remove_path(config.paths.run_versions_path)

    for dataset_dir in [config.paths.coco128_dir, config.paths.annotation_dir, config.paths.augmented_dir]:
        remove_path(dataset_dir / "predictions")
        remove_path(dataset_dir / "predictions_manifest.json")

    remove_local_model_file(config)
    print(f"Pruned post-training artifacts under {config.paths.dataset_dir}")


def remove_local_model_file(config: AppConfig) -> None:
    model_path = Path(config.train.model_name)
    if not model_path.is_absolute():
        model_path = (config.paths.project_root / model_path).resolve()
    remove_path(model_path)
