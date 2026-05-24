#!/usr/bin/env python3
"""Prepare per-client image splits for Flower federated learning.

The input zip can contain class folders at the archive root, or one top-level
directory that contains the class folders.
"""

import argparse
import os
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

DEFAULT_BASE_DIR = Path("flower-fl") / "client" / "data"


def split_for_client(
    zip_path: Path,
    client_id: int,
    base_client_dir: Path,
    fraction: float,
    train_ratio: float,
    seed: Optional[int] = None,
):
    """
    Extract classes from zip_path, sample `fraction` of images per class,
    splits into train/test by `train_ratio`, and copies them into:
      base_client_dir/client_<ID>/{train,test}/{class}/

    Existing directories are reused and files are not overwritten.
    Supports zips with either class folders at top-level or one top-level folder containing classes.
    """
    if not zip_path.is_file():
        print(f"ERROR: ZIP file not found at {zip_path}", file=sys.stderr)
        sys.exit(1)
    if not 0 < fraction <= 1:
        print("ERROR: --fraction must be greater than 0 and less than or equal to 1.", file=sys.stderr)
        sys.exit(1)
    if not 0 < train_ratio < 1:
        print("ERROR: --train-ratio must be greater than 0 and less than 1.", file=sys.stderr)
        sys.exit(1)

    base_client_dir.mkdir(parents=True, exist_ok=True)
    client_dir = base_client_dir / f"client_{client_id}"
    train_root = client_dir / "train"
    test_root  = client_dir / "test"

    # Inform about existing structure
    if client_dir.exists():
        print(f"Directory {client_dir} already exists; new images will be added without overwriting.")
    else:
        print(f"Creating new directories at {client_dir}.")

    # Create train/test directories if missing
    for path in (train_root, test_root):
        path.mkdir(parents=True, exist_ok=True)

    # Extract ZIP to temporary folder
    with tempfile.TemporaryDirectory(prefix="fl_dataset_") as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmp_path)

        # Find actual class parent folder(s)
        root_dirs = sorted(
            [p for p in tmp_path.iterdir() if p.is_dir() and not p.name.startswith("__")]
        )
        if not root_dirs:
            print(f"ERROR: No directories found in ZIP at {tmp_path}", file=sys.stderr)
            sys.exit(1)

        # Case A: direct class folders containing image files
        direct = [d for d in root_dirs if any((d / f).is_file() for f in os.listdir(d))]
        if direct:
            class_parent = tmp_path
            classes = sorted(d.name for d in direct)
        else:
            # Case B: one top-level folder containing classes
            class_parent = root_dirs[0]
            classes = sorted(
                d.name for d in class_parent.iterdir() if d.is_dir() and not d.name.startswith("__")
            )
            if not classes:
                print(f"ERROR: No class folders under {class_parent}", file=sys.stderr)
                sys.exit(1)

        print(f"Detected classes: {classes}")

        rng = random.Random(seed if seed is not None else client_id)
        # Process each class
        for cls in classes:
            src = class_parent / cls
            images = sorted(f.name for f in src.iterdir() if f.is_file())
            if not images:
                print(f"Warning: No images in class '{cls}'", file=sys.stderr)
                continue

            # Sample and split
            k = max(1, int(len(images) * fraction))
            sampled = rng.sample(images, k)
            n_train = max(1, int(len(sampled) * train_ratio))
            if len(sampled) > 1:
                n_train = min(n_train, len(sampled) - 1)
            train_imgs, test_imgs = sampled[:n_train], sampled[n_train:]

            for split, imgs in [('train', train_imgs), ('test', test_imgs)]:
                out_dir = client_dir / split / cls
                out_dir.mkdir(parents=True, exist_ok=True)
                for fname in imgs:
                    dest = out_dir / fname
                    if dest.exists():
                        print(f"Skipping existing file: {dest}")
                    else:
                        shutil.copy2(src / fname, dest)

    print(f"Client {client_id} data prepared at {client_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a zipped image dataset into one Flower client partition."
    )
    parser.add_argument("zip_file", type=Path, help="Path to the dataset zip.")
    parser.add_argument(
        "-c",
        "--client-id",
        type=int,
        default=1,
        help="Client id used for flower-fl/client/data/client_<id>.",
    )
    parser.add_argument(
        "-o",
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f"Base output directory for client data. Default: {DEFAULT_BASE_DIR}",
    )
    parser.add_argument(
        "-f",
        "--fraction",
        type=float,
        default=0.2,
        help="Fraction of each class to sample for this client.",
    )
    parser.add_argument(
        "-t",
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of sampled images assigned to train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed. Defaults to the client id.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_for_client(
        zip_path=args.zip_file,
        client_id=args.client_id,
        base_client_dir=args.base_dir,
        fraction=args.fraction,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print("Dataset split complete.")
