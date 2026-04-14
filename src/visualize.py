from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk
import yaml

from src.common import clamp_bbox, discover_images, read_json, resolve_dataset_directory
from src.config import AppConfig


@dataclass
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
    dataset_dir = resolve_visualizer_dataset_dir(config, dataset_subdir)
    app = VisualizerApp(
        dataset_dir=dataset_dir,
        show_labels=show_labels,
        show_predictions=show_predictions,
    )
    app.run()


def resolve_visualizer_dataset_dir(config: AppConfig, dataset_subdir: Path | None) -> Path:
    if dataset_subdir is not None:
        return resolve_dataset_directory(
            project_root=config.paths.project_root,
            dataset_root=config.paths.dataset_dir,
            dataset_path=dataset_subdir,
        )

    latest_manifest_path = config.paths.infer_latest_manifest_path
    if latest_manifest_path.exists():
        manifest = read_json(latest_manifest_path)
        dataset_dir_value = manifest.get("dataset_dir")
        if isinstance(dataset_dir_value, str) and dataset_dir_value.strip():
            dataset_dir = Path(dataset_dir_value).resolve()
            if dataset_dir.exists():
                return dataset_dir

    return resolve_dataset_directory(
        project_root=config.paths.project_root,
        dataset_root=config.paths.dataset_dir,
        dataset_path=Path(config.infer.dataset_name),
    )


class VisualizerApp:
    def __init__(self, dataset_dir: Path, show_labels: bool, show_predictions: bool) -> None:
        self.dataset_dir = dataset_dir
        self.images_root = dataset_dir / "images"
        self.labels_root = dataset_dir / "labels"
        self.predictions_root = dataset_dir / "predictions"
        self.class_names = load_class_names(dataset_dir)
        self.image_paths = discover_images(self.images_root)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_root}")
        self.labels_available = self.labels_root.exists()
        self.predictions_available = self.predictions_root.exists()

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

    def build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.grid(sticky="nsew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(container, background="#1f1f1f", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        sidebar = ttk.Frame(container, padding=(12, 0, 0, 0))
        sidebar.grid(row=0, column=1, sticky="ns")
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(4, weight=1)

        self.status_label = ttk.Label(sidebar, text="", wraplength=280)
        self.status_label.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        ttk.Checkbutton(sidebar, text="Show labels", variable=self.show_labels_var, command=self.render).grid(
            row=1,
            column=0,
            sticky="w",
        )
        ttk.Checkbutton(
            sidebar,
            text="Show predictions",
            variable=self.show_predictions_var,
            command=self.render,
        ).grid(row=2, column=0, sticky="w", pady=(4, 12))

        overlay_toggle_row = ttk.Frame(sidebar)
        overlay_toggle_row.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        overlay_toggle_row.columnconfigure(0, weight=1)
        overlay_toggle_row.columnconfigure(1, weight=1)

        ttk.Button(overlay_toggle_row, text="Show All", command=self.show_all_overlays).grid(
            row=0,
            column=0,
            sticky="ew",
        )
        ttk.Button(overlay_toggle_row, text="Hide All", command=self.hide_all_overlays).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(8, 0),
        )

        self.overlay_listbox = tk.Listbox(sidebar, height=20)
        self.overlay_listbox.grid(row=4, column=0, sticky="nsew")

        navigation_row = ttk.Frame(sidebar)
        navigation_row.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        navigation_row.columnconfigure(0, weight=1)
        navigation_row.columnconfigure(1, weight=1)
        navigation_row.columnconfigure(2, weight=1)

        ttk.Button(navigation_row, text="Prev", command=self.show_previous_image).grid(row=0, column=0, sticky="ew")
        ttk.Button(navigation_row, text="Refresh", command=self.reload_current_image).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(navigation_row, text="Next", command=self.show_next_image).grid(row=0, column=2, sticky="ew")

    def run(self) -> None:
        self.root.mainloop()

    def load_image(self, index: int) -> None:
        self.current_index = index
        image_path = self.image_paths[index]
        with Image.open(image_path) as opened_image:
            self.current_image = opened_image.convert("RGB")

        self.image_width, self.image_height = self.current_image.size
        relative_path = image_path.relative_to(self.dataset_dir)
        overlay_status_lines = [f"{index + 1}/{len(self.image_paths)}", str(relative_path)]
        overlay_status_lines.append(self.describe_overlay_availability())
        self.status_label.config(text="\n".join(overlay_status_lines))
        self.refresh_overlay_list()
        self.render()

    def describe_overlay_availability(self) -> str:
        label_text = "labels: available" if self.labels_available else "labels: missing"
        prediction_text = "predictions: available" if self.predictions_available else "predictions: missing"
        return f"{label_text} | {prediction_text}"

    def reload_current_image(self) -> None:
        self.load_image(self.current_index)

    def on_canvas_resize(self, _event: tk.Event[tk.Canvas]) -> None:
        if self.current_image is not None:
            self.render()

    def render(self) -> None:
        if self.current_image is None:
            return

        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        self.display_scale = min(canvas_width / self.image_width, canvas_height / self.image_height)

        display_width = max(1, int(self.image_width * self.display_scale))
        display_height = max(1, int(self.image_height * self.display_scale))
        self.display_offset_x = (canvas_width - display_width) / 2
        self.display_offset_y = (canvas_height - display_height) / 2

        resized_image = self.current_image.resize((display_width, display_height))
        self.current_photo = ImageTk.PhotoImage(resized_image)
        self.canvas.delete("all")
        self.canvas.create_image(self.display_offset_x, self.display_offset_y, anchor="nw", image=self.current_photo)

        if self.show_labels_var.get():
            for annotation in self.load_label_annotations():
                self.draw_annotation(annotation, outline_color="#00d084")

        if self.show_predictions_var.get():
            for annotation in self.load_prediction_annotations():
                self.draw_annotation(annotation, outline_color="#ff6b6b")

    def show_all_overlays(self) -> None:
        self.show_labels_var.set(True)
        self.show_predictions_var.set(True)
        self.render()

    def hide_all_overlays(self) -> None:
        self.show_labels_var.set(False)
        self.show_predictions_var.set(False)
        self.render()

    def draw_annotation(self, annotation: OverlayAnnotation, outline_color: str) -> None:
        x1, y1, x2, y2 = self.canvas_coordinates_for_bbox(annotation.bbox)
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=outline_color, width=2)
        self.canvas.create_text(
            x1 + 4,
            max(y1 - 12, 4),
            text=annotation.class_name,
            anchor="nw",
            fill=outline_color,
            font=("TkDefaultFont", 10, "bold"),
        )

    def canvas_coordinates_for_bbox(self, bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x_center, y_center, width, height = bbox
        left = (x_center - width / 2) * self.image_width
        top = (y_center - height / 2) * self.image_height
        right = (x_center + width / 2) * self.image_width
        bottom = (y_center + height / 2) * self.image_height

        canvas_left = self.display_offset_x + left * self.display_scale
        canvas_top = self.display_offset_y + top * self.display_scale
        canvas_right = self.display_offset_x + right * self.display_scale
        canvas_bottom = self.display_offset_y + bottom * self.display_scale
        return canvas_left, canvas_top, canvas_right, canvas_bottom

    def refresh_overlay_list(self) -> None:
        self.overlay_listbox.delete(0, tk.END)
        label_annotations = self.load_label_annotations()
        prediction_annotations = self.load_prediction_annotations()

        if not self.labels_available:
            self.overlay_listbox.insert(tk.END, "label overlay unavailable")
        else:
            for annotation in label_annotations:
                self.overlay_listbox.insert(tk.END, f"label: {annotation.class_name}")

        if not self.predictions_available:
            self.overlay_listbox.insert(tk.END, "prediction overlay unavailable")
        else:
            for annotation in prediction_annotations:
                self.overlay_listbox.insert(tk.END, f"prediction: {annotation.class_name}")

        if self.overlay_listbox.size() == 0:
            self.overlay_listbox.insert(tk.END, "no overlays for this image")

    def load_label_annotations(self) -> list[OverlayAnnotation]:
        if not self.labels_available:
            return []
        image_path = self.image_paths[self.current_index]
        label_path = related_annotation_path(image_path, self.images_root, self.labels_root)
        return load_overlay_annotations(label_path, self.class_names)

    def load_prediction_annotations(self) -> list[OverlayAnnotation]:
        if not self.predictions_available:
            return []
        image_path = self.image_paths[self.current_index]
        prediction_path = related_annotation_path(image_path, self.images_root, self.predictions_root)
        return load_overlay_annotations(prediction_path, self.class_names)

    def show_previous_image(self) -> None:
        if self.current_index == 0:
            return
        self.load_image(self.current_index - 1)

    def show_next_image(self) -> None:
        if self.current_index >= len(self.image_paths) - 1:
            return
        self.load_image(self.current_index + 1)


def related_annotation_path(image_path: Path, images_root: Path, overlay_root: Path) -> Path:
    relative_image_path = image_path.relative_to(images_root)
    return overlay_root / relative_image_path.with_suffix(".txt")


def load_overlay_annotations(label_path: Path, class_names: dict[int, str]) -> list[OverlayAnnotation]:
    if not label_path.exists():
        return []

    annotations: list[OverlayAnnotation] = []
    for class_id, bbox in parse_overlay_file(label_path):
        class_name = class_names.get(class_id, f"class_{class_id}")
        annotations.append(OverlayAnnotation(class_id=class_id, class_name=class_name, bbox=bbox))
    return annotations


def parse_overlay_file(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    annotations: list[tuple[int, tuple[float, float, float, float]]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid overlay line in {path}: {line}")

        class_id = int(parts[0])
        bbox_values = [float(value) for value in parts[1:5]]
        bbox = clamp_bbox(bbox_values)
        annotations.append((class_id, bbox))

    return annotations


def load_class_names(dataset_dir: Path) -> dict[int, str]:
    dataset_yaml_path = dataset_dir / "dataset.yaml"
    if dataset_yaml_path.exists():
        payload = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8")) or {}
        raw_names = payload.get("names", {})
        if isinstance(raw_names, dict):
            return {int(key): str(value) for key, value in raw_names.items()}
        if isinstance(raw_names, list):
            return {index: str(value) for index, value in enumerate(raw_names)}

    classes_json_path = dataset_dir / "classes.json"
    if classes_json_path.exists():
        payload = read_json(classes_json_path)
        raw_mapping = payload.get("name_to_id", payload)
        return {int(class_id): str(name) for name, class_id in raw_mapping.items()}

    return {}
