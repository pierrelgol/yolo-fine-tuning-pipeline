from __future__ import annotations

import json
from pathlib import Path
import shutil
import zipfile

import yaml

from src.common import discover_images, ensure_dir, write_json
from src.config import AppConfig


def prepare_dataset(
    config: AppConfig,
    archive_path: Path | None = None,
    force: bool = False,
) -> dict:
    selected_archive_path = archive_path or config.paths.raw_dir / config.fetch.archive_name
    output_dir = config.paths.coco128_dir

    if not selected_archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {selected_archive_path}")

    if force and output_dir.exists():
        shutil.rmtree(output_dir)

    if output_dir.exists() and not force:
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            print(json.dumps(manifest, indent=2))
            return manifest

    extract_archive_to_directory(selected_archive_path, output_dir)
    manifest = inspect_unpacked_dataset(output_dir)
    write_dataset_yaml(config)
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def extract_archive_to_directory(archive_path: Path, output_dir: Path) -> None:
    temporary_extract_dir = output_dir.parent / f"{output_dir.name}_tmp_extract"
    if temporary_extract_dir.exists():
        shutil.rmtree(temporary_extract_dir)

    ensure_dir(temporary_extract_dir)
    with zipfile.ZipFile(archive_path) as archive_file:
        archive_file.extractall(temporary_extract_dir)

    extracted_children = list(temporary_extract_dir.iterdir())
    if len(extracted_children) == 1 and extracted_children[0].is_dir():
        extracted_root = extracted_children[0]
    else:
        extracted_root = temporary_extract_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.move(str(extracted_root), str(output_dir))

    if temporary_extract_dir.exists():
        shutil.rmtree(temporary_extract_dir)


def inspect_unpacked_dataset(output_dir: Path) -> dict:
    image_dir = output_dir / "images" / "train2017"
    label_dir = output_dir / "labels" / "train2017"
    prediction_dir = output_dir / "predictions" / "train2017"

    if not image_dir.exists():
        raise FileNotFoundError(f"Expected image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Expected label directory not found: {label_dir}")
    ensure_dir(prediction_dir)

    image_paths = discover_images(image_dir)
    manifest = {
        "dataset_dir": str(output_dir),
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "prediction_dir": str(prediction_dir),
        "num_images": len(image_paths),
        "dataset_yaml": str(output_dir / "dataset.yaml"),
    }
    return manifest


def write_dataset_yaml(config: AppConfig) -> None:
    dataset_yaml_path = config.paths.coco128_dataset_yaml_path
    dataset_payload = {
        "path": str(config.paths.coco128_dir.resolve()),
        "train": "images/train2017",
        "val": "images/train2017",
        "names": {index: name for index, name in enumerate(bundled_class_names())},
    }
    dataset_yaml_path.write_text(yaml.safe_dump(dataset_payload, sort_keys=False), encoding="utf-8")


def bundled_class_names() -> list[str]:
    try:
        import ultralytics
    except Exception:
        return []

    dataset_yaml_path = Path(ultralytics.__file__).resolve().parent / "cfg" / "datasets" / "coco128.yaml"
    if not dataset_yaml_path.exists():
        return []

    payload = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8")) or {}
    raw_names = payload.get("names", {})

    if isinstance(raw_names, dict):
        ordered_items = sorted(raw_names.items(), key=lambda item: int(item[0]))
        return [str(name) for _, name in ordered_items]

    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]

    return []
