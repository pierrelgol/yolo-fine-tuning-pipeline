from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import yaml
from PIL import Image, ImageTk

from src.common import (
    dataset_classes_path,
    discover_images,
    parse_yolo_labels,
    preferred_image_split,
    read_json,
    resolve_dataset_directory,
    resolve_portable_path,
)
from src.config import AppConfig


@dataclass(frozen=True)
class VisualizerTarget:
    dataset_dir: Path
    images_root: Path
    labels_root: Path
    predictions_root: Path


@dataclass(frozen=True)
class OverlayAnnotation:
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]


def launch_visualizer(
    config: AppConfig,
    dataset_subdir: Path | None = None,
    show_labels: bool = True,
    show_predictions: bool = True,
) -> None:
    target = resolve_visualizer_target(config, dataset_subdir)
    VisualizerApp(target, show_labels, show_predictions).run()


def resolve_visualizer_target(
    config: AppConfig, dataset_subdir: Path | None
) -> VisualizerTarget:
    if dataset_subdir is not None:
        dataset_dir = resolve_dataset_directory(
            config.paths.project_root, config.paths.dataset_dir, dataset_subdir
        )
        return target_from_dataset_dir(dataset_dir)

    if config.paths.infer_latest_manifest_path.exists():
        manifest = read_json(config.paths.infer_latest_manifest_path)
        dataset_dir_value = manifest.get("dataset_dir")
        images_root_value = manifest.get("images_root")
        if isinstance(dataset_dir_value, str) and isinstance(
            images_root_value, str
        ):
            dataset_dir = resolve_portable_path(
                dataset_dir_value, project_root=config.paths.project_root
            )
            images_root = resolve_portable_path(
                images_root_value, project_root=config.paths.project_root
            )
            if dataset_dir.exists() and images_root.exists():
                return target_from_paths(dataset_dir, images_root)

    dataset_dir = resolve_dataset_directory(
        config.paths.project_root,
        config.paths.dataset_dir,
        Path(config.infer.dataset_name),
    )
    return target_from_dataset_dir(dataset_dir)


def target_from_dataset_dir(dataset_dir: Path) -> VisualizerTarget:
    split = preferred_image_split(dataset_dir)
    if split:
        images_root = dataset_dir / "images" / split
    else:
        images_root = dataset_dir / "images"
    return target_from_paths(dataset_dir, images_root)


def target_from_paths(dataset_dir: Path, images_root: Path) -> VisualizerTarget:
    labels_root = dataset_dir / "labels"
    predictions_root = dataset_dir / "predictions"
    images_dir = dataset_dir / "images"

    if images_root != images_dir:
        split = images_root.relative_to(images_dir)
        labels_root = labels_root / split
        predictions_root = predictions_root / split

    return VisualizerTarget(
        dataset_dir=dataset_dir,
        images_root=images_root,
        labels_root=labels_root,
        predictions_root=predictions_root,
    )


class VisualizerApp:
    def __init__(
        self,
        target: VisualizerTarget,
        show_labels: bool,
        show_predictions: bool,
    ) -> None:
        self.target = target
        self.class_names = load_class_names(target.dataset_dir)
        self.image_paths = discover_images(target.images_root)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {target.images_root}")

        self.current_index = 0
        self.current_image: Image.Image | None = None
        self.current_photo: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0
        self.display_offset_x = 0.0
        self.display_offset_y = 0.0
        self.image_width = 1
        self.image_height = 1

        self.root = tk.Tk()
        self.root.title("Dataset Visualizer")
        self.root.geometry("1280x900")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.show_labels_var = tk.BooleanVar(value=show_labels)
        self.show_predictions_var = tk.BooleanVar(value=show_predictions)
        self.build_layout()
        self.load_image(0)

    def run(self) -> None:
        self.root.mainloop()

    def build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.grid(sticky="nsew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            container, background="#1f1f1f", highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        sidebar = ttk.Frame(container, padding=(12, 0, 0, 0))
        sidebar.grid(row=0, column=1, sticky="ns")
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(4, weight=1)

        self.status_label = ttk.Label(sidebar, text="", wraplength=280)
        self.status_label.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        ttk.Checkbutton(
            sidebar,
            text="Show labels",
            variable=self.show_labels_var,
            command=self.render,
        ).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(
            sidebar,
            text="Show predictions",
            variable=self.show_predictions_var,
            command=self.render,
        ).grid(row=2, column=0, sticky="w", pady=(4, 12))

        toggle_row = ttk.Frame(sidebar)
        toggle_row.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        toggle_row.columnconfigure(0, weight=1)
        toggle_row.columnconfigure(1, weight=1)

        ttk.Button(
            toggle_row, text="Show All", command=self.show_all_overlays
        ).grid(row=0, column=0, sticky="ew")
        ttk.Button(
            toggle_row, text="Hide All", command=self.hide_all_overlays
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.overlay_listbox = tk.Listbox(sidebar, height=20)
        self.overlay_listbox.grid(row=4, column=0, sticky="nsew")

        navigation_row = ttk.Frame(sidebar)
        navigation_row.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        for column_index in range(3):
            navigation_row.columnconfigure(column_index, weight=1)

        ttk.Button(
            navigation_row, text="Prev", command=self.show_previous_image
        ).grid(row=0, column=0, sticky="ew")
        ttk.Button(
            navigation_row, text="Refresh", command=self.reload_current_image
        ).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(
            navigation_row, text="Next", command=self.show_next_image
        ).grid(row=0, column=2, sticky="ew")

    def load_image(self, index: int) -> None:
        self.current_index = index
        image_path = self.image_paths[index]

        with Image.open(image_path) as opened_image:
            self.current_image = opened_image.convert("RGB")

        self.image_width, self.image_height = self.current_image.size
        relative_path = image_path.relative_to(self.target.dataset_dir)
        self.status_label.config(
            text="\n".join([
                f"{index + 1}/{len(self.image_paths)}",
                str(relative_path),
                self.describe_overlay_availability(),
            ])
        )
        self.refresh_overlay_list()
        self.render()

    def describe_overlay_availability(self) -> str:
        labels_text = (
            "labels: available"
            if self.target.labels_root.exists()
            else "labels: missing"
        )
        predictions_text = (
            "predictions: available"
            if self.target.predictions_root.exists()
            else "predictions: missing"
        )
        return f"{labels_text} | {predictions_text}"

    def reload_current_image(self) -> None:
        self.load_image(self.current_index)

    def on_canvas_resize(self, _event: tk.Event[tk.Canvas]) -> None:
        self.render()

    def render(self) -> None:
        if self.current_image is None:
            return

        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        self.display_scale = min(
            canvas_width / self.image_width, canvas_height / self.image_height
        )

        display_width = max(1, int(self.image_width * self.display_scale))
        display_height = max(1, int(self.image_height * self.display_scale))
        self.display_offset_x = (canvas_width - display_width) / 2
        self.display_offset_y = (canvas_height - display_height) / 2

        resized_image = self.current_image.resize((
            display_width,
            display_height,
        ))
        self.current_photo = ImageTk.PhotoImage(resized_image)

        self.canvas.delete("all")
        self.canvas.create_image(
            self.display_offset_x,
            self.display_offset_y,
            anchor="nw",
            image=self.current_photo,
        )

        if self.show_labels_var.get():
            for annotation in load_overlay_annotations(
                self.overlay_path(self.target.labels_root), self.class_names
            ):
                self.draw_annotation(annotation, "#00d084")

        if self.show_predictions_var.get():
            for annotation in load_overlay_annotations(
                self.overlay_path(self.target.predictions_root),
                self.class_names,
            ):
                self.draw_annotation(annotation, "#ff6b6b")

    def overlay_path(self, overlay_root: Path) -> Path:
        image_path = self.image_paths[self.current_index]
        relative_path = image_path.relative_to(
            self.target.images_root
        ).with_suffix(".txt")
        return overlay_root / relative_path

    def draw_annotation(
        self, annotation: OverlayAnnotation, outline_color: str
    ) -> None:
        x_center, y_center, width, height = annotation.bbox
        left = (x_center - width / 2) * self.image_width
        top = (y_center - height / 2) * self.image_height
        right = (x_center + width / 2) * self.image_width
        bottom = (y_center + height / 2) * self.image_height

        x1 = self.display_offset_x + left * self.display_scale
        y1 = self.display_offset_y + top * self.display_scale
        x2 = self.display_offset_x + right * self.display_scale
        y2 = self.display_offset_y + bottom * self.display_scale

        self.canvas.create_rectangle(
            x1, y1, x2, y2, outline=outline_color, width=2
        )
        self.canvas.create_text(
            x1 + 4,
            max(y1 - 12, 4),
            text=annotation.class_name,
            anchor="nw",
            fill=outline_color,
            font=("TkDefaultFont", 10, "bold"),
        )

    def refresh_overlay_list(self) -> None:
        self.overlay_listbox.delete(0, tk.END)

        label_annotations = load_overlay_annotations(
            self.overlay_path(self.target.labels_root), self.class_names
        )
        prediction_annotations = load_overlay_annotations(
            self.overlay_path(self.target.predictions_root), self.class_names
        )

        if not self.target.labels_root.exists():
            self.overlay_listbox.insert(tk.END, "label overlay unavailable")
        elif not label_annotations:
            self.overlay_listbox.insert(tk.END, "no labels for this image")
        else:
            for annotation in label_annotations:
                self.overlay_listbox.insert(
                    tk.END, f"label: {annotation.class_name}"
                )

        if not self.target.predictions_root.exists():
            self.overlay_listbox.insert(
                tk.END, "prediction overlay unavailable"
            )
        elif not prediction_annotations:
            self.overlay_listbox.insert(tk.END, "no predictions for this image")
        else:
            for annotation in prediction_annotations:
                self.overlay_listbox.insert(
                    tk.END, f"prediction: {annotation.class_name}"
                )

    def show_all_overlays(self) -> None:
        self.show_labels_var.set(True)
        self.show_predictions_var.set(True)
        self.render()

    def hide_all_overlays(self) -> None:
        self.show_labels_var.set(False)
        self.show_predictions_var.set(False)
        self.render()

    def show_previous_image(self) -> None:
        if self.current_index > 0:
            self.load_image(self.current_index - 1)

    def show_next_image(self) -> None:
        if self.current_index < len(self.image_paths) - 1:
            self.load_image(self.current_index + 1)


def load_overlay_annotations(
    label_path: Path, class_names: dict[int, str]
) -> list[OverlayAnnotation]:
    annotations: list[OverlayAnnotation] = []
    for class_id, bbox in parse_yolo_labels(label_path):
        annotations.append(
            OverlayAnnotation(
                class_id=class_id,
                class_name=class_names.get(class_id, f"class_{class_id}"),
                bbox=bbox,
            )
        )
    return annotations


def load_class_names(dataset_dir: Path) -> dict[int, str]:
    dataset_yaml = dataset_dir / "dataset.yaml"
    if dataset_yaml.exists():
        payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
        raw_names = payload.get("names", {})
        if isinstance(raw_names, dict):
            return {int(key): str(value) for key, value in raw_names.items()}
        if isinstance(raw_names, list):
            return {index: str(value) for index, value in enumerate(raw_names)}

    class_map = (
        read_json(dataset_classes_path(dataset_dir))
        if dataset_classes_path(dataset_dir).exists()
        else {}
    )
    raw_mapping = class_map.get("name_to_id", class_map)
    return {int(class_id): str(name) for name, class_id in raw_mapping.items()}
