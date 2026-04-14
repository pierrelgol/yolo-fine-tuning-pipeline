install:
    uv sync

sync: install

fetch *args:
    uv run python cli.py fetch {{args}}

setup *args:
    uv run python cli.py setup {{args}}

annotate *args:
    uv run python cli.py annotate {{args}}

augment background_dir *args:
    uv run python cli.py augment {{background_dir}} {{args}}

visualize *args:
    uv run python cli.py visualize {{args}}

show *args:
    uv run trackio show {{args}}

train *args:
    uv run python cli.py train {{args}}

eval *args:
    uv run python cli.py eval {{args}}

infer *args:
    uv run python cli.py infer {{args}}

watch *args:
    uv run python cli.py watch {{args}}

prepare *args:
    uv run python cli.py prepare {{args}}

prune:
    uv run python cli.py prune

clean:
    uv run python cli.py clean
