# AGENTS.md — yolo-fine-tuning-pipeline

## Quick Start

```bash
just install          # uv sync — creates .venv
just prepare          # fetch + setup (download coco128, unpack into dataset/)
just annotate         # opens annotation GUI for labeling images
just train            # builds dataset/train, trains model, logs to Trackio
just eval             # evaluates latest trained weights
just infer            # runs inference on dataset/augmented (default)
just visualize        # opens dataset visualizer
just prune            # removes train/eval/infer artifacts and predictions
just clean            # removes entire generated dataset tree
```

## Toolchain

- **Package manager**: `uv` (lockfile `uv.lock` is checked in)
- **Task runner**: `just` (see `Justfile`)
- **Python**: >=3.12,<3.13
- **Entry point**: `cli.py` — all commands are `uv run python cli.py <command> [args]`
- **Config**: `config.toml` is the single source of truth. CLI flags override config values. Pass `--config <path>` to use a different config.

## PyTorch / CUDA

- `pyproject.toml` configures an explicit `pytorch-cu128` index for `torch` and `torchvision`. Reinstalling the env picks up CUDA 12.8 wheels automatically.
- Device resolution in `src/train.py`: `"auto"` → CUDA if available, else CPU. `"0"` or `"cuda"` raises if CUDA unavailable. Workers forced to 0 on Windows.

## Dataset Layout

All dataset subdirectories follow the same structure:

```
dataset/
  raw/            # downloaded archive
  coco128/        # unpacked base dataset
  annotation/     # user-labeled images (output of `just annotate`)
  augmented/      # synthetic samples (output of `just augment <bg_dir>`)
  train/          # assembled train/val split + weights (output of `just train`)
  eval/           # evaluation results
  infer/          # inference predictions
```

Each dataset folder contains `images/<split>/`, `labels/<split>/`, `predictions/<split>/`, `classes.json`, `dataset.yaml`, `manifest.json`. Splits are `train2017` and `val2017` (defined in `src/common.py`).

## Pipeline Order Matters

Commands depend on prior steps producing artifacts:
1. `fetch` → `dataset/raw/coco128.zip`
2. `setup` → `dataset/coco128/`
3. `annotate` → `dataset/annotation/` (required before `train`)
4. `augment` → `dataset/augmented/` (optional, enriches training data)
5. `train` → `dataset/train/` + weights at `dataset/train/best.pt` and `dataset/train/latest.pt`
6. `eval` / `infer` / `watch` → consume trained weights

`just prepare` combines fetch + setup.

## src/ Module Map

| File | Purpose |
|---|---|
| `cli.py` | Argument parser, dispatches to src/ modules |
| `src/config.py` | Loads `config.toml` into frozen dataclasses |
| `src/common.py` | Shared utilities: path resolution, YOLO label I/O, class maps, dataset YAML generation |
| `src/fetch.py` | Downloads dataset archive |
| `src/setup.py` | Unpacks archive, creates initial split |
| `src/annotate.py` | Annotation GUI |
| `src/augment.py` | Background compositing augmentation |
| `src/train.py` | Builds training dataset, runs ultralytics YOLO training with Trackio callbacks |
| `src/eval.py` | Model evaluation |
| `src/infer.py` | Batch inference |
| `src/watch.py` | Live video inference |
| `src/tracking.py` | Trackio experiment logging |
| `src/visualize.py` | Dataset visualizer |
| `src/clean.py` | Artifact cleanup |

## No Tests, Lint, or CI

This repo has no test suite, linter, formatter, type checker, or CI workflows. Do not assume one exists.

## Key Gotchas

- `dataset/` is gitignored. All pipeline output lives there and is ephemeral.
- Run names are auto-incremented via `dataset/run_versions.json`. Training refuses to overwrite an existing run unless `--force` is passed.
- The training dataset is rebuilt from scratch each time `train` runs (annotation + augmented samples are re-split).
- `config.toml` paths are resolved relative to the config file's parent directory (project root).
