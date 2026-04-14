from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_SPLIT = "train2017"
VAL_SPLIT = "val2017"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    return ensure_dir(path)


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def non_empty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def discover_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    image_paths: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)

    image_paths.sort()
    return image_paths


def copy_file(source_path: Path, destination_path: Path) -> None:
    ensure_dir(destination_path.parent)
    shutil.copy2(source_path, destination_path)


def stable_name(*parts: str, suffix: str) -> str:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{digest}{suffix}"


def split_items(items: Sequence, ratio: float, seed: int) -> tuple[list, list]:
    ordered_items = list(items)
    if len(ordered_items) <= 1:
        return ordered_items, []

    random.Random(seed).shuffle(ordered_items)
    split_index = int(len(ordered_items) * ratio)
    split_index = max(1, min(len(ordered_items) - 1, split_index))
    return ordered_items[:split_index], ordered_items[split_index:]


def clamp_bbox(values: Sequence[float]) -> tuple[float, float, float, float]:
    return tuple(min(1.0, max(0.0, value)) for value in values)  # type: ignore[return-value]


def yolo_label_line(class_id: int, bbox: Sequence[float]) -> str:
    x_center, y_center, width, height = clamp_bbox(bbox)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def save_yolo_labels(path: Path, lines: Sequence[str]) -> None:
    ensure_dir(path.parent)
    payload = "\n".join(lines)
    if payload:
        payload = f"{payload}\n"
    path.write_text(payload, encoding="utf-8")


def parse_yolo_labels(
    path: Path,
) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not path.exists():
        return []

    annotations: list[tuple[int, tuple[float, float, float, float]]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label line in {path}: {line}")

        class_id = int(parts[0])
        bbox = clamp_bbox([float(value) for value in parts[1:5]])
        annotations.append((class_id, bbox))

    return annotations


def load_class_map(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}

    payload = read_json(path)
    raw_mapping = payload.get("name_to_id", payload)
    return {str(name): int(class_id) for name, class_id in raw_mapping.items()}


def save_class_map(path: Path, mapping: dict[str, int]) -> None:
    ordered_items = sorted(mapping.items(), key=lambda item: item[1])
    write_json(
        path,
        {"name_to_id": {name: class_id for name, class_id in ordered_items}},
    )


def ordered_class_names(class_map: dict[str, int]) -> list[str]:
    ordered_items = sorted(class_map.items(), key=lambda item: item[1])
    return [class_name for class_name, _ in ordered_items]


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def image_label_path(
    image_path: Path, image_root: Path, label_root: Path
) -> Path:
    return label_root / image_path.relative_to(image_root).with_suffix(".txt")


def dataset_images_dir(dataset_dir: Path, split: str = TRAIN_SPLIT) -> Path:
    return dataset_dir / "images" / split


def dataset_labels_dir(dataset_dir: Path, split: str = TRAIN_SPLIT) -> Path:
    return dataset_dir / "labels" / split


def dataset_predictions_dir(
    dataset_dir: Path, split: str = TRAIN_SPLIT
) -> Path:
    return dataset_dir / "predictions" / split


def dataset_yaml_path(dataset_dir: Path) -> Path:
    return dataset_dir / "dataset.yaml"


def dataset_classes_path(dataset_dir: Path) -> Path:
    return dataset_dir / "classes.json"


def dataset_manifest_path(dataset_dir: Path) -> Path:
    return dataset_dir / "manifest.json"


def resolve_path(path: Path | str, base_dir: Path | None = None) -> Path:
    raw_path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if raw_path.is_absolute():
        return raw_path.resolve()
    anchor_dir = (
        base_dir.resolve() if base_dir is not None else Path.cwd().resolve()
    )
    return (anchor_dir / raw_path).resolve()


def portable_path(path: Path | str, base_dir: Path | None = None) -> str:
    resolved_path = resolve_path(path, base_dir=base_dir)
    if base_dir is None:
        return resolved_path.as_posix()

    try:
        return resolved_path.relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def resolve_portable_path(
    value: str,
    *,
    project_root: Path,
    dataset_root: Path | None = None,
) -> Path:
    candidate_path = Path(os.path.expandvars(os.path.expanduser(value)))
    if candidate_path.is_absolute():
        return candidate_path.resolve()

    base_dirs: list[Path] = []
    if dataset_root is not None:
        base_dirs.append(dataset_root.resolve())
    base_dirs.extend([project_root.resolve(), Path.cwd().resolve()])

    seen_base_dirs: set[Path] = set()
    for base_dir in base_dirs:
        if base_dir in seen_base_dirs:
            continue
        seen_base_dirs.add(base_dir)

        resolved_candidate = (base_dir / candidate_path).resolve()
        if resolved_candidate.exists():
            return resolved_candidate

    return (project_root.resolve() / candidate_path).resolve()


def write_dataset_yaml(
    dataset_dir: Path,
    class_names: Sequence[str],
    train_split: str = TRAIN_SPLIT,
    val_split: str | None = None,
) -> Path:
    import yaml

    if val_split is None:
        val_split = (
            VAL_SPLIT
            if dataset_images_dir(dataset_dir, VAL_SPLIT).exists()
            else train_split
        )

    payload = {
        "train": f"images/{train_split}",
        "val": f"images/{val_split}",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    path = dataset_yaml_path(dataset_dir)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def preferred_image_split(dataset_dir: Path) -> str:
    if discover_images(dataset_images_dir(dataset_dir, VAL_SPLIT)):
        return VAL_SPLIT
    if discover_images(dataset_images_dir(dataset_dir, TRAIN_SPLIT)):
        return TRAIN_SPLIT
    return ""


def resolve_dataset_directory(
    project_root: Path, dataset_root: Path, dataset_path: Path
) -> Path:
    candidate_paths: list[Path] = []
    if dataset_path.is_absolute():
        candidate_paths.append(dataset_path.resolve())
    else:
        candidate_paths.append((dataset_root / dataset_path).resolve())
        candidate_paths.append((project_root / dataset_path).resolve())
        candidate_paths.append((Path.cwd() / dataset_path).resolve())

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        "Dataset directory not found. Checked: "
        + ", ".join(str(path) for path in candidate_paths)
    )


def sanitized_name(value: str) -> str:
    normalized_value = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip(
        "-"
    )
    return normalized_value or "run"


def next_run_name(versions_path: Path, model_name: str) -> str:
    payload = read_json(versions_path) if versions_path.exists() else {}
    next_version = int(payload.get("next_version", 1))
    write_json(versions_path, {"next_version": next_version + 1})
    return f"{sanitized_name(Path(model_name).stem)}-v{next_version:03d}-{date.today().isoformat()}"


def child_run_name(parent_run_name: str, task_name: str) -> str:
    return f"{parent_run_name}-{sanitized_name(task_name)}"


def latest_train_run_name(
    train_runs_dir: Path, latest_run_path: Path, fallback_name: str
) -> str:
    if latest_run_path.exists():
        run_name = read_json(latest_run_path).get("run_name")
        if isinstance(run_name, str) and run_name.strip():
            return run_name

    if train_runs_dir.exists():
        run_dirs = [path for path in train_runs_dir.iterdir() if path.is_dir()]
        if run_dirs:
            run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return run_dirs[0].name

    return fallback_name
