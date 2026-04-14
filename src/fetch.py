from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

from src.common import ensure_dir, non_empty_file
from src.config import AppConfig


def fetch_dataset(
    config: AppConfig,
    dataset_url: str | None = None,
    archive_name: str | None = None,
) -> Path:
    selected_url = dataset_url or config.fetch.dataset_url
    selected_archive_name = archive_name or config.fetch.archive_name

    raw_dir = config.paths.raw_dir
    ensure_dir(raw_dir)

    destination_path = raw_dir / selected_archive_name
    if non_empty_file(destination_path):
        print(f"Skipping download, file already exists: {destination_path}")
        return destination_path

    downloaded_path, _ = urlretrieve(selected_url, destination_path)
    downloaded_archive_path = Path(downloaded_path)
    print(f"Downloaded dataset archive to {downloaded_archive_path}")
    return downloaded_archive_path
