"""Dataset download and extraction for MovieLens datasets."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


# base url for our datasource
BASE_URL = "https://files.grouplens.org/datasets/movielens"

# dataset registry
DATASET_REGISTRY = {
    "ml-1m": {
        "url": f"{BASE_URL}/ml-1m.zip",
        "extract_dir": "ml-1m",
        "required_files": ["ratings.dat", "users.dat", "movies.dat"],
    },
    "ml-10m": {
        "url": f"{BASE_URL}/ml-10m.zip",
        "extract_dir": "ml-10m",
        "required_files": ["ratings.dat", "movies.dat"],
    },
    # other datasets will be added later in the experiment
}


# private fn to retrieve the information of a dataset from our registry
def _dataset_info(name: str) -> dict:
    try:
        return DATASET_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"Unknown dataset '{name}'. Available datasets: {available}"
        ) from None


# public fn used to validate the dataset directory to ensure it contains all the required files
def validate_dataset(dataset_dir: Path, required_files: list[str]) -> bool:
    """Check that a dataset directory contains all required, non-empty files."""
    for filename in required_files:
        path = dataset_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            return False
    return True


# private fn that used to download the dataset from the remote source
def _download(url: str, dest: Path) -> None:
    def progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = min(downloaded * 100 // total_size, 100)
        sys.stdout.write(f"\rDownloading {dest.name}: {percent:3d}%")
        sys.stdout.flush()
        if downloaded >= total_size:
            sys.stdout.write("\n")

    urllib.request.urlretrieve(url, dest, reporthook=progress)


# private fn that extracts the file with the dataset zip folder
def _extract(
    zip_path: Path, data_root: Path, extract_dir: str, dest_dir: Path
) -> None:
    tmp_dir = data_root / f".{extract_dir}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp_dir)
        source = tmp_dir / extract_dir
        if not source.is_dir():
            raise ValueError(
                f"Archive does not contain expected directory '{extract_dir}'"
            )
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(source), str(dest_dir))
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


# public fn used to download and validate the dataset with idempotency handling to avoid duplicate or redownloads
def download_dataset(
    name: str, data_dir: str = "data", force: bool = False
) -> Path:
    """Download and extract a dataset into data_dir.

    Returns the path to the extracted dataset directory. Skips re-downloading
    when the dataset is already present and valid unless force is True.
    """
    info = _dataset_info(name)
    data_root = Path(data_dir)
    dataset_dir = data_root / info["extract_dir"]

    if not force and validate_dataset(dataset_dir, info["required_files"]):
        print(f"Dataset '{name}' already present at {dataset_dir}")
        return dataset_dir

    data_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = data_root / info["extract_dir"]
    zip_path = data_root / f"{name}.zip"

    _download(info["url"], zip_path)
    _extract(zip_path, data_root, info["extract_dir"], dataset_dir)

    if not validate_dataset(dataset_dir, info["required_files"]):
        raise RuntimeError(
            f"Extraction produced an invalid dataset at {dataset_dir}"
        )
    print(f"Dataset '{name}' ready at {dataset_dir}")
    return dataset_dir



# main fn used for using cli to trigger data ingestion
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract a MovieLens dataset."
    )
    parser.add_argument(
        "dataset",
        help="Dataset name, e.g. ml-1m, ml-10m",
        choices=sorted(DATASET_REGISTRY),
    )
    parser.add_argument(
        "--data-dir", default="data", help="Directory to download into (default: data)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the dataset is already present",
    )
    args = parser.parse_args()

    try:
        download_dataset(args.dataset, data_dir=args.data_dir, force=args.force)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
