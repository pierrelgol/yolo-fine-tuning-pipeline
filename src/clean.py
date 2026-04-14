from __future__ import annotations

from pathlib import Path
import shutil

from src.config import AppConfig


def clean_artifacts(config: AppConfig) -> None:
    remove_directory_if_present(config.paths.dataset_dir)
    remove_local_model_file_if_present(config)
    print(f"Removed artifacts under {config.paths.dataset_dir}")


def prune_artifacts(config: AppConfig) -> None:
    remove_directory_if_present(config.paths.train_dir)
    remove_directory_if_present(config.paths.eval_dir)
    remove_directory_if_present(config.paths.infer_dir)

    clear_prediction_outputs(config.paths.coco128_dir)
    clear_prediction_outputs(config.paths.annotation_dir)
    clear_prediction_outputs(config.paths.augmented_dir)

    remove_local_model_file_if_present(config)
    print(f"Pruned post-training artifacts under {config.paths.dataset_dir}")


def remove_directory_if_present(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)


def remove_local_model_file_if_present(config: AppConfig) -> None:
    model_path = Path(config.train.model_name)
    if not model_path.is_absolute():
        model_path = (config.paths.project_root / model_path).resolve()

    if model_path.exists() and model_path.is_file():
        model_path.unlink()


def clear_prediction_outputs(dataset_dir: Path) -> None:
    predictions_dir = dataset_dir / "predictions"
    predictions_manifest_path = dataset_dir / "predictions_manifest.json"

    if predictions_dir.exists():
        shutil.rmtree(predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)

    if predictions_manifest_path.exists():
        predictions_manifest_path.unlink()
