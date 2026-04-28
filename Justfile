# Install the locked Python environment.
install:
    uv sync

sync: install

# Download the base COCO128 archive into dataset/raw.
fetch *args:
    uv run python cli.py fetch {{args}}

# Validate source classes and unpack COCO128 into dataset/coco128.
setup *args:
    uv run python cli.py setup {{args}}

# Build dataset/augmented. Defaults to [augment].background_dir.
augment *args:
    uv run python cli.py augment {{args}}

# Open the dataset visualizer.
visualize *args:
    uv run python cli.py visualize {{args}}

# Open the Trackio UI.
show *args:
    uv run trackio show {{args}}

# Build dataset/train and run curriculum YOLO fine-tuning.
train *args:
    uv run python cli.py train {{args}}

# Evaluate the latest trained weights.
eval *args:
    uv run python cli.py eval {{args}}

# Run inference on a dataset folder, defaulting to dataset/augmented.
infer *args:
    uv run python cli.py infer {{args}}

# Run live video inference.
watch *args:
    uv run python cli.py watch {{args}}

# Fetch, setup, and augment in sequence.
prepare *args:
    uv run python cli.py prepare {{args}}

# Remove train/eval/infer artifacts and predictions.
prune:
    uv run python cli.py prune

# Remove the full generated dataset tree and local model file.
clean:
    uv run python cli.py clean
