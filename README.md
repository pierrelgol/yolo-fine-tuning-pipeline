# yolo-fine-tuning-pipeline

Minimal, beginner-friendly YOLO fine-tuning pipeline.

Everything is driven by `config.toml` and `cli.py`. Each step stays in its own script under `src/`, and the dataset layout stays consistent across the whole pipeline:

```text
dataset/
  raw/
  coco128/
  annotation/
  augmented/
  train/
  eval/
  infer/
```

Each dataset folder uses the same basic shape when relevant:

```text
<dataset>/
  images/
  labels/
  predictions/
  classes.json
  dataset.yaml
  manifest.json
```

## Install

```text
just install
```

This project is intentionally CUDA-first on Windows through the PyTorch CUDA index configured in `pyproject.toml`.

## Typical Flow

```text
just fetch
just setup
just annotate
just augment <background_dir>
just train
just eval
just infer
just watch <video_path>
just show
just visualize
```

## Commands

- `just fetch`: download the base dataset archive into `dataset/raw`
- `just setup`: unpack the archive into `dataset/coco128`
- `just annotate`: annotate images from `image/` into `dataset/annotation`
- `just augment <background_dir>`: build `dataset/augmented`
- `just train`: build `dataset/train`, train YOLO, save stable weights, log to Trackio
- `just eval`: evaluate the latest trained weights
- `just infer`: run inference on the augmented validation split by default
- `just watch <video_path>`: open a live video window with YOLO predictions
- `just show`: open the Trackio UI
- `just visualize`: inspect labels and predictions, defaulting to the latest infer target
- `just prune`: remove train/eval/infer artifacts and predictions
- `just clean`: remove the full generated dataset tree

## Configuration

`config.toml` is the single source of truth for:

- paths
- fetch source
- train defaults
- train hyperparameters
- eval defaults
- infer defaults
- Trackio logging frequency and limits

Override any command from the CLI when needed:

```text
uv run python cli.py train --epochs 20 --batch 16
```
