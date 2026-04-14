from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def non_empty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_name(*parts: str, suffix: str) -> str:
    joined_parts = "::".join(parts)
    digest = hashlib.sha256(joined_parts.encode("utf-8")).hexdigest()[:16]
    return f"{digest}{suffix}"


def discover_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    image_paths: list[Path] = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_paths.append(path)

    image_paths.sort()
    return image_paths


def copy_image(source_path: Path, destination_path: Path) -> None:
    ensure_dir(destination_path.parent)
    shutil.copy2(source_path, destination_path)


def split_items(items: Sequence[Path], ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    ordered_items = list(items)
    random_generator = random.Random(seed)
    random_generator.shuffle(ordered_items)

    if len(ordered_items) <= 1:
        return sorted(ordered_items), []

    split_index = int(len(ordered_items) * ratio)
    split_index = max(1, split_index)
    split_index = min(len(ordered_items) - 1, split_index)

    train_items = sorted(ordered_items[:split_index])
    validation_items = sorted(ordered_items[split_index:])
    return train_items, validation_items


def coco_to_yolo_bbox(coco_bbox: Sequence[float], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x_min, y_min, box_width, box_height = coco_bbox
    x_center = (x_min + box_width / 2) / image_width
    y_center = (y_min + box_height / 2) / image_height
    normalized_width = box_width / image_width
    normalized_height = box_height / image_height
    return x_center, y_center, normalized_width, normalized_height


def clamp_bbox(values: Sequence[float]) -> tuple[float, float, float, float]:
    clamped_values: list[float] = []
    for value in values:
        clamped_value = min(1.0, max(0.0, value))
        clamped_values.append(clamped_value)
    return tuple(clamped_values)  # type: ignore[return-value]


def dense_category_mapping(categories: Iterable[dict]) -> dict[int, int]:
    ordered_categories = sorted(categories, key=lambda item: int(item["id"]))
    mapping: dict[int, int] = {}
    for index, category in enumerate(ordered_categories):
        category_id = int(category["id"])
        mapping[category_id] = index
    return mapping


def yolo_label_line(class_id: int, bbox: Sequence[float]) -> str:
    x_center, y_center, width, height = clamp_bbox(bbox)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def save_yolo_labels(path: Path, lines: Sequence[str]) -> None:
    ensure_dir(path.parent)
    payload = "\n".join(lines)
    if payload:
        payload = f"{payload}\n"
    path.write_text(payload, encoding="utf-8")


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return width, height


def image_label_path(image_path: Path, image_root: Path, label_root: Path) -> Path:
    relative_image_path = image_path.relative_to(image_root)
    return label_root / relative_image_path.with_suffix(".txt")


def parse_yolo_labels(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not path.exists():
        return []

    annotations: list[tuple[int, tuple[float, float, float, float]]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label line in {path}: {line}")

        class_id = int(parts[0])
        bbox_values = [float(value) for value in parts[1:5]]
        bbox = clamp_bbox(bbox_values)
        annotations.append((class_id, bbox))

    return annotations


def load_class_map(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}

    payload = read_json(path)
    raw_mapping = payload.get("name_to_id", payload)

    class_map: dict[str, int] = {}
    for name, class_id in raw_mapping.items():
        class_map[str(name)] = int(class_id)

    return class_map


def save_class_map(path: Path, mapping: dict[str, int]) -> None:
    ordered_items = sorted(mapping.items(), key=lambda item: item[1])
    ordered_mapping = {name: class_id for name, class_id in ordered_items}
    write_json(path, {"name_to_id": ordered_mapping})


def resolve_dataset_directory(project_root: Path, dataset_root: Path, dataset_path: Path) -> Path:
    candidate_paths: list[Path] = []

    if dataset_path.is_absolute():
        candidate_paths.append(dataset_path.resolve())
    else:
        candidate_paths.append((Path.cwd() / dataset_path).resolve())
        candidate_paths.append((project_root / dataset_path).resolve())
        candidate_paths.append((dataset_root / dataset_path).resolve())

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    rendered_candidates = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"Dataset directory not found. Checked: {rendered_candidates}")
