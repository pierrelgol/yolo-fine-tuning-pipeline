from __future__ import annotations

import json
from pathlib import Path
import shutil
import zipfile

from src.common import (
    dataset_images_dir,
    dataset_labels_dir,
    dataset_manifest_path,
    dataset_predictions_dir,
    dataset_yaml_path,
    discover_images,
    ensure_dir,
    write_json,
)
from src.config import AppConfig


def prepare_dataset(config: AppConfig, archive_path: Path | None = None, force: bool = False) -> dict:
    source_archive_path = archive_path or config.paths.raw_dir / config.fetch.archive_name
    dataset_dir = config.paths.coco128_dir
    manifest_path = dataset_manifest_path(dataset_dir)

    if not source_archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {source_archive_path}")

    if dataset_dir.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(json.dumps(manifest, indent=2))
        return manifest

    unpack_archive(source_archive_path, dataset_dir)
    manifest = build_manifest(dataset_dir)
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def unpack_archive(archive_path: Path, dataset_dir: Path) -> None:
    temporary_dir = dataset_dir.parent / f"{dataset_dir.name}_tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)

    ensure_dir(temporary_dir)
    with zipfile.ZipFile(archive_path) as archive_file:
        archive_file.extractall(temporary_dir)

    extracted_children = list(temporary_dir.iterdir())
    extracted_root = extracted_children[0] if len(extracted_children) == 1 and extracted_children[0].is_dir() else temporary_dir

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    shutil.move(str(extracted_root), str(dataset_dir))

    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)


def build_manifest(dataset_dir: Path) -> dict:
    image_dir = dataset_images_dir(dataset_dir)
    label_dir = dataset_labels_dir(dataset_dir)
    prediction_dir = dataset_predictions_dir(dataset_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Expected image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Expected label directory not found: {label_dir}")

    ensure_dir(prediction_dir)

    if not dataset_yaml_path(dataset_dir).exists():
        dataset_yaml_path(dataset_dir).write_text("", encoding="utf-8")

    return {
        "dataset_dir": str(dataset_dir),
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "prediction_dir": str(prediction_dir),
        "num_images": len(discover_images(image_dir)),
        "dataset_yaml": str(dataset_yaml_path(dataset_dir)),
    }
