from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from src.common import (
    dataset_classes_path,
    dataset_images_dir,
    dataset_labels_dir,
    dataset_manifest_path,
    dataset_predictions_dir,
    dataset_yaml_path,
    discover_images,
    ensure_dir,
    ordered_class_names,
    portable_path,
    resolve_path,
    save_class_map,
    write_json,
)
from src.config import AppConfig


def prepare_dataset(
    config: AppConfig, archive_path: Path | None = None, force: bool = False
) -> dict:
    source_archive_path = (
        resolve_path(archive_path, base_dir=config.paths.project_root)
        if archive_path is not None
        else config.paths.raw_dir / config.fetch.archive_name
    )
    dataset_dir = config.paths.coco128_dir
    manifest_path = dataset_manifest_path(dataset_dir)

    if not source_archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {source_archive_path}")

    class_map, source_manifest = validate_source_images(
        config.paths.augment_source_dir, config.paths.project_root
    )

    if dataset_dir.exists() and manifest_path.exists() and not force:
        save_class_map(dataset_classes_path(dataset_dir), class_map)
        write_json(dataset_dir / "source_manifest.json", source_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_manifest"] = portable_path(
            dataset_dir / "source_manifest.json",
            base_dir=config.paths.project_root,
        )
        manifest["classes"] = source_manifest.get("classes", [])
        manifest["num_classes"] = source_manifest.get("num_classes", 0)
        write_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2))
        return manifest

    unpack_archive(source_archive_path, dataset_dir)
    save_class_map(dataset_classes_path(dataset_dir), class_map)
    write_json(dataset_dir / "source_manifest.json", source_manifest)

    manifest = build_manifest(
        dataset_dir,
        config.paths.project_root,
        source_manifest=source_manifest,
    )
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
    extracted_root = (
        extracted_children[0]
        if len(extracted_children) == 1 and extracted_children[0].is_dir()
        else temporary_dir
    )

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    shutil.move(str(extracted_root), str(dataset_dir))

    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)


def build_manifest(
    dataset_dir: Path, project_root: Path, source_manifest: dict | None = None
) -> dict:
    image_dir = dataset_images_dir(dataset_dir)
    label_dir = dataset_labels_dir(dataset_dir)
    prediction_dir = dataset_predictions_dir(dataset_dir)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Expected image directory not found: {image_dir}"
        )
    if not label_dir.exists():
        raise FileNotFoundError(
            f"Expected label directory not found: {label_dir}"
        )

    ensure_dir(prediction_dir)

    if not dataset_yaml_path(dataset_dir).exists():
        dataset_yaml_path(dataset_dir).write_text("", encoding="utf-8")

    manifest = {
        "dataset_dir": portable_path(dataset_dir, base_dir=project_root),
        "image_dir": portable_path(image_dir, base_dir=project_root),
        "label_dir": portable_path(label_dir, base_dir=project_root),
        "prediction_dir": portable_path(prediction_dir, base_dir=project_root),
        "num_images": len(discover_images(image_dir)),
        "dataset_yaml": portable_path(
            dataset_yaml_path(dataset_dir), base_dir=project_root
        ),
    }
    if source_manifest is not None:
        manifest["source_manifest"] = portable_path(
            dataset_dir / "source_manifest.json", base_dir=project_root
        )
        manifest["classes"] = source_manifest.get("classes", [])
        manifest["num_classes"] = source_manifest.get("num_classes", 0)
    return manifest


def validate_source_images(
    source_dir: Path, project_root: Path
) -> tuple[dict[str, int], dict]:
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source image directory not found: {source_dir}. "
            "Create images/<class_name>/ folders before setup."
        )

    class_map: dict[str, int] = {}
    class_entries: list[dict] = []
    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        image_paths = discover_images(class_dir)
        if not image_paths:
            continue
        class_map[class_dir.name] = len(class_map)
        class_entries.append(
            {
                "class_name": class_dir.name,
                "image_dir": portable_path(class_dir, base_dir=project_root),
                "num_images": len(image_paths),
                "images": [
                    portable_path(image_path, base_dir=project_root)
                    for image_path in image_paths
                ],
            }
        )

    if not class_map:
        raise FileNotFoundError(
            f"No class image folders found under {source_dir}. "
            "Expected folders like images/<class_name>/ containing images."
        )

    return class_map, {
        "source_dir": portable_path(source_dir, base_dir=project_root),
        "classes": ordered_class_names(class_map),
        "num_classes": len(class_map),
        "class_entries": class_entries,
    }
