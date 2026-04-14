from __future__ import annotations

from src.common import remove_path, resolve_path
from src.config import AppConfig
from src.tracking import delete_tracking_project


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
    delete_tracking_project(config.tracking.project_name)
    print(f"Pruned post-training artifacts under {config.paths.dataset_dir}")


def remove_local_model_file(config: AppConfig) -> None:
    model_path = resolve_path(config.train.model_name, base_dir=config.paths.project_root)
    remove_path(model_path)
