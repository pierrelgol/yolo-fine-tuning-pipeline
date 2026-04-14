from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from src.common import (
    copy_file,
    dataset_classes_path,
    dataset_images_dir,
    dataset_labels_dir,
    dataset_manifest_path,
    dataset_predictions_dir,
    discover_images,
    ensure_dir,
    image_label_path,
    load_class_map,
    ordered_class_names,
    parse_yolo_labels,
    save_class_map,
    save_yolo_labels,
    write_dataset_yaml,
    write_json,
)
from src.config import AppConfig


@dataclass
class Annotation:
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]


def launch_annotation_gui(
    config: AppConfig,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    classes_path: Path | None = None,
    manifest_path: Path | None = None,
) -> None:
    dataset_dir = config.paths.annotation_dir
    source_image_dir = image_dir or config.paths.image_dir
    annotation_image_dir = dataset_images_dir(dataset_dir)
    annotation_label_dir = label_dir or dataset_labels_dir(dataset_dir)
    classes_file_path = classes_path or dataset_classes_path(dataset_dir)
    dataset_manifest_file_path = manifest_path or dataset_manifest_path(dataset_dir)

    ensure_dir(annotation_image_dir)
    ensure_dir(annotation_label_dir)
    ensure_dir(dataset_predictions_dir(dataset_dir))

    AnnotatorApp(
        source_image_dir=source_image_dir,
        annotation_image_dir=annotation_image_dir,
        label_dir=annotation_label_dir,
        classes_path=classes_file_path,
        manifest_path=dataset_manifest_file_path,
        dataset_dir=dataset_dir,
    ).run()


class AnnotatorApp:
    def __init__(
        self,
        source_image_dir: Path,
        annotation_image_dir: Path,
        label_dir: Path,
        classes_path: Path,
        manifest_path: Path,
        dataset_dir: Path,
    ) -> None:
        self.source_image_dir = source_image_dir
        self.annotation_image_dir = annotation_image_dir
        self.label_dir = label_dir
        self.classes_path = classes_path
        self.manifest_path = manifest_path
        self.dataset_dir = dataset_dir

        self.source_image_paths = discover_images(source_image_dir)
        if not self.source_image_paths:
            raise FileNotFoundError(f"No images found in {source_image_dir}")

        self.image_paths = self.sync_images_into_dataset()
        self.class_map = load_class_map(classes_path)
        self.current_image_index = 0
        self.current_image: Image.Image | None = None
        self.current_photo: ImageTk.PhotoImage | None = None
        self.current_annotations: list[Annotation] = []
        self.display_scale = 1.0
        self.display_offset_x = 0.0
        self.display_offset_y = 0.0
        self.image_width = 1
        self.image_height = 1
        self.drag_start: tuple[float, float] | None = None
        self.drag_end: tuple[float, float] | None = None

        self.root = tk.Tk()
        self.root.title("YOLO Annotator")
        self.root.geometry("1280x900")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.load_image(0)

    def run(self) -> None:
        self.root.mainloop()

    def build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.grid(sticky="nsew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(container, background="#202124", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        sidebar = ttk.Frame(container, padding=(12, 0, 0, 0))
        sidebar.grid(row=0, column=1, sticky="ns")
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(3, weight=1)

        self.image_status_label = ttk.Label(sidebar, text="", wraplength=260)
        self.image_status_label.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(sidebar, text="Class name").grid(row=1, column=0, sticky="w")
        self.class_name_entry = ttk.Entry(sidebar)
        self.class_name_entry.grid(row=2, column=0, sticky="ew", pady=(4, 8))

        self.annotation_listbox = tk.Listbox(sidebar, height=18)
        self.annotation_listbox.grid(row=3, column=0, sticky="nsew")

        action_row = ttk.Frame(sidebar)
        action_row.grid(row=4, column=0, sticky="ew", pady=(8, 8))
        action_row.columnconfigure(0, weight=1)
        action_row.columnconfigure(1, weight=1)

        ttk.Button(action_row, text="Delete Selected", command=self.delete_selected_annotation).grid(row=0, column=0, sticky="ew")
        ttk.Button(action_row, text="Delete Last", command=self.delete_last_annotation).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        navigation_row = ttk.Frame(sidebar)
        navigation_row.grid(row=5, column=0, sticky="ew")
        for column_index in range(4):
            navigation_row.columnconfigure(column_index, weight=1)

        ttk.Button(navigation_row, text="Prev", command=self.show_previous_image).grid(row=0, column=0, sticky="ew")
        ttk.Button(navigation_row, text="Save", command=self.save_current_annotations).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(navigation_row, text="Next", command=self.show_next_image).grid(row=0, column=2, sticky="ew")
        ttk.Button(navigation_row, text="Finish", command=self.finish_annotation).grid(row=0, column=3, sticky="ew", padx=(8, 0))

        instructions = "Draw a box with the mouse. Type the class name before you release the mouse button."
        ttk.Label(sidebar, text=instructions, wraplength=260, justify="left").grid(row=6, column=0, sticky="ew", pady=(12, 0))

    def sync_images_into_dataset(self) -> list[Path]:
        copied_paths: list[Path] = []
        for source_image_path in self.source_image_paths:
            relative_path = source_image_path.relative_to(self.source_image_dir)
            destination_image_path = self.annotation_image_dir / relative_path
            copy_file(source_image_path, destination_image_path)
            copied_paths.append(destination_image_path)
        copied_paths.sort()
        return copied_paths

    def load_image(self, image_index: int) -> None:
        self.current_image_index = image_index
        image_path = self.image_paths[image_index]

        with Image.open(image_path) as opened_image:
            self.current_image = opened_image.convert("RGB")

        self.image_width, self.image_height = self.current_image.size
        self.current_annotations = []

        for class_id, bbox in parse_yolo_labels(self.label_path_for_image(image_path)):
            self.current_annotations.append(
                Annotation(class_id=class_id, class_name=self.class_name_for_id(class_id), bbox=bbox)
            )

        relative_path = image_path.relative_to(self.dataset_dir)
        self.image_status_label.config(text=f"{image_index + 1}/{len(self.image_paths)}\n{relative_path}")
        self.refresh_annotation_list()
        self.render()

    def label_path_for_image(self, image_path: Path) -> Path:
        return image_label_path(image_path, self.annotation_image_dir, self.label_dir)

    def refresh_annotation_list(self) -> None:
        self.annotation_listbox.delete(0, tk.END)
        for index, annotation in enumerate(self.current_annotations, start=1):
            x_center, y_center, width, height = annotation.bbox
            self.annotation_listbox.insert(
                tk.END,
                f"{index}. {annotation.class_name} [{x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f}]",
            )

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

        for annotation in self.current_annotations:
            x1, y1, x2, y2 = self.canvas_coordinates_for_bbox(annotation.bbox)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00d084", width=2)
            self.canvas.create_text(
                x1 + 4,
                max(y1 - 12, 4),
                text=annotation.class_name,
                anchor="nw",
                fill="#00d084",
                font=("TkDefaultFont", 10, "bold"),
            )

        if self.drag_start is not None and self.drag_end is not None:
            self.canvas.create_rectangle(*self.drag_start, *self.drag_end, outline="#ffb703", width=2, dash=(6, 4))

    def canvas_coordinates_for_bbox(self, bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x_center, y_center, width, height = bbox
        left = (x_center - width / 2) * self.image_width
        top = (y_center - height / 2) * self.image_height
        right = (x_center + width / 2) * self.image_width
        bottom = (y_center + height / 2) * self.image_height
        return (
            self.display_offset_x + left * self.display_scale,
            self.display_offset_y + top * self.display_scale,
            self.display_offset_x + right * self.display_scale,
            self.display_offset_y + bottom * self.display_scale,
        )

    def clamp_canvas_point_to_image(self, x: float, y: float) -> tuple[float, float]:
        minimum_x = self.display_offset_x
        minimum_y = self.display_offset_y
        maximum_x = self.display_offset_x + self.image_width * self.display_scale
        maximum_y = self.display_offset_y + self.image_height * self.display_scale
        return min(max(x, minimum_x), maximum_x), min(max(y, minimum_y), maximum_y)

    def image_coordinates_for_canvas_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            (x - self.display_offset_x) / self.display_scale,
            (y - self.display_offset_y) / self.display_scale,
        )

    def bbox_from_drag(self, start_point: tuple[float, float], end_point: tuple[float, float]) -> tuple[float, float, float, float]:
        start_x, start_y = self.image_coordinates_for_canvas_point(*start_point)
        end_x, end_y = self.image_coordinates_for_canvas_point(*end_point)
        left = min(start_x, end_x)
        right = max(start_x, end_x)
        top = min(start_y, end_y)
        bottom = max(start_y, end_y)
        return (
            ((left + right) / 2) / self.image_width,
            ((top + bottom) / 2) / self.image_height,
            (right - left) / self.image_width,
            (bottom - top) / self.image_height,
        )

    def on_canvas_resize(self, _event: tk.Event[tk.Canvas]) -> None:
        self.render()

    def on_mouse_press(self, event: tk.Event[tk.Canvas]) -> None:
        if self.current_image is None:
            return
        self.drag_start = self.clamp_canvas_point_to_image(event.x, event.y)
        self.drag_end = self.drag_start
        self.render()

    def on_mouse_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self.drag_start is None:
            return
        self.drag_end = self.clamp_canvas_point_to_image(event.x, event.y)
        self.render()

    def on_mouse_release(self, event: tk.Event[tk.Canvas]) -> None:
        if self.drag_start is None:
            return

        self.drag_end = self.clamp_canvas_point_to_image(event.x, event.y)
        class_name = self.class_name_entry.get().strip()
        if not class_name:
            self.clear_drag()
            messagebox.showerror("Missing class name", "Enter a class name before drawing a box.")
            return

        bbox = self.bbox_from_drag(self.drag_start, self.drag_end)
        if bbox[2] < 0.005 or bbox[3] < 0.005:
            self.clear_drag()
            self.render()
            return

        self.current_annotations.append(
            Annotation(class_id=self.class_id_for_name(class_name), class_name=class_name, bbox=bbox)
        )
        self.clear_drag()
        self.save_current_annotations()

    def clear_drag(self) -> None:
        self.drag_start = None
        self.drag_end = None

    def class_id_for_name(self, class_name: str) -> int:
        if class_name in self.class_map:
            return self.class_map[class_name]
        next_class_id = max(self.class_map.values(), default=-1) + 1
        self.class_map[class_name] = next_class_id
        return next_class_id

    def class_name_for_id(self, class_id: int) -> str:
        for class_name, mapped_class_id in self.class_map.items():
            if mapped_class_id == class_id:
                return class_name
        return f"class_{class_id}"

    def save_current_annotations(self) -> None:
        image_path = self.image_paths[self.current_image_index]
        label_lines = [f"{annotation.class_id} {annotation.bbox[0]:.6f} {annotation.bbox[1]:.6f} {annotation.bbox[2]:.6f} {annotation.bbox[3]:.6f}" for annotation in self.current_annotations]
        save_yolo_labels(self.label_path_for_image(image_path), label_lines)
        save_class_map(self.classes_path, self.class_map)
        write_dataset_yaml(self.dataset_dir, ordered_class_names(self.class_map), val_split="train2017")
        self.write_manifest()
        self.refresh_annotation_list()
        self.render()

    def write_manifest(self) -> None:
        annotated_images: list[dict[str, str | int]] = []
        for image_path in self.image_paths:
            label_path = self.label_path_for_image(image_path)
            if not label_path.exists():
                continue
            annotated_images.append(
                {
                    "image": str(image_path),
                    "label": str(label_path),
                    "num_annotations": len(parse_yolo_labels(label_path)),
                }
            )

        write_json(
            self.manifest_path,
            {
                "dataset_dir": str(self.dataset_dir),
                "source_image_dir": str(self.source_image_dir),
                "image_dir": str(self.annotation_image_dir),
                "label_dir": str(self.label_dir),
                "classes_path": str(self.classes_path),
                "num_images": len(self.image_paths),
                "annotated_images": annotated_images,
            },
        )

    def delete_selected_annotation(self) -> None:
        selection = self.annotation_listbox.curselection()
        if selection:
            self.current_annotations.pop(selection[0])
            self.save_current_annotations()

    def delete_last_annotation(self) -> None:
        if self.current_annotations:
            self.current_annotations.pop()
            self.save_current_annotations()

    def show_previous_image(self) -> None:
        if self.current_image_index > 0:
            self.save_current_annotations()
            self.load_image(self.current_image_index - 1)

    def show_next_image(self) -> None:
        if self.current_image_index < len(self.image_paths) - 1:
            self.save_current_annotations()
            self.load_image(self.current_image_index + 1)

    def on_close(self) -> None:
        self.save_current_annotations()
        self.root.destroy()

    def finish_annotation(self) -> None:
        self.on_close()
