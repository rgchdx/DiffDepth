#!/usr/bin/env python3
"""
NYUv2 preprocessing script.

This mirrors the `notebooks/preprocess.ipynb` workflow:
1) Build RGB/depth pair CSV from folders (optional).
2) Load pair CSV with `NYUDepthDataset`.
3) Resize and convert to tensors.
4) Save processed samples to chunked `.pt` files (good for external drive storage).

Examples
--------
Build train/test CSVs from split folders:
    python scripts/preprocess.py make-csv --data-root data --out-dir data

Process train CSV into chunked tensors:
    python scripts/preprocess.py process \
      --csv-path data/nyu2_train.csv \
      --project-root . \
      --save-root /media/$USER/ExternalDrive/diffdepth_data/processed/train
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Iterable
import cv2
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm


COLOR = "_colors.png"
DEPTH = "_depth.png"

# for more flexible filename parsing, we can use suffixes instead of fixed positions
def _to_abs(path_like: str, data_root: Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (data_root / p)

# finding specific columns in a CSV with some common name variations
def _find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


# this is for building the initial CSVs from the raw dataset folders, which have a consistent naming
# pattern but we want to be robust to different folder structures and naming conventions
def build_pairs_from_folder(folder: Path) -> pd.DataFrame:
    color_map: dict[str, Path] = {}
    depth_map: dict[str, Path] = {}

    for image_path in sorted(folder.glob("*.png")):
        name = image_path.name
        if name.endswith(COLOR):
            key = name[: -len(COLOR)]
            color_map[key] = image_path
        elif name.endswith(DEPTH):
            key = name[: -len(DEPTH)]
            depth_map[key] = image_path
    
    keys = sorted(set(color_map) & set(depth_map))
    rows = [{"rgb_path": str(color_map[k]), "depth_path": str(depth_map[k])} for k in keys]
    return pd.DataFrame(rows, columns=["rgb_path", "depth_path"])


# this is for loading the CSVs, which may have diff formats
def load_pairs_csv(csv_path: Path, project_root: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None)
    if df.empty:
        return pd.DataFrame(columns=["rgb_path", "depth_path"])
    
    if df.shape[1] >= 2:
        df = df.iloc[:, :2].copy()
        df.columns = ["rgb_path", "depth_path"]
    else:
        named = pd.read_csv(csv_path)
        rgb_col = _find_column(named.columns, ["rgb_path", "rgb", "color_path", "image", "input"])
        depth_col = _find_column(named.columns, ["depth_path", "depth", "target", "label"])
        if rgb_col is None or depth_col is None:
            raise ValueError(
                f"Could not infer rgb/depth columns from '{csv_path}'. Use a 2-column CSV."
            )
        df = pd.DataFrame({"rgb_path": named[rgb_col], "depth_path": named[depth_col]})
    
    df["rgb_path"] = df["rgb_path"].astype(str).map(lambda p: str(_to_abs(p, project_root)))
    df["depth_path"] = df["depth_path"].astype(str).map(lambda p: str(_to_abs(p, project_root)))
    return df


def validate_pairs(df: pd.DataFrame, strict: bool) -> pd.DataFrame:
    exists_rgb = df["rgb_path"].map(lambda p: Path(p).exists())
    exists_depth = df["depth_path"].map(lambda p: Path(p).exists())
    keep = exists_rgb & exists_depth

    missing = int((~keep).sum())
    if missing > 0:
        msg = f"Found {missing} missing rgb/depth paths."
        if strict:
            raise FileNotFoundError(msg)
        print(f"Warning: {msg} Dropping those rows.")

    return df[keep].reset_index(drop=True)


class NYUDepthDataset(Dataset):
    def __init__(self, csv_path: str | Path, project_root: str | Path, size: tuple[int, int] = (256, 256)):
        self.project_root = Path(project_root).resolve()
        self.df = load_pairs_csv(Path(csv_path), self.project_root)
        self.df = validate_pairs(self.df, strict=False)
        self.resize = T.Resize(size)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        rgb_path = self.df.iloc[idx]["rgb_path"]
        depth_path = self.df.iloc[idx]["depth_path"]

        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if rgb is None:
            raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")
        if depth is None:
            raise FileNotFoundError(f"Could not read depth image: {depth_path}")

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        rgb_tensor = torch.tensor(rgb).permute(2, 0, 1).float() / 255.0
        depth_tensor = torch.tensor(depth).unsqueeze(0).float()

        rgb_tensor = self.resize(rgb_tensor)
        depth_tensor = self.resize(depth_tensor)

        return {"rgb": rgb_tensor, "depth": depth_tensor}


# This is the main function for saving the processed dataset into chunked .pt files,
# which is more efficient for loading during training and works well with external drive storage.
def save_chunks(
    dataset: Dataset,
    save_root: Path,
    chunk_size: int,
    limit: int | None = None,
) -> None:
    save_root.mkdir(parents=True, exist_ok=True)
    total = len(dataset) if limit is None else min(limit, len(dataset))

    buffer: list[dict[str, torch.Tensor]] = []
    chunk_id = 0

    for i in tqdm(range(total), desc="Preprocessing"):
        buffer.append(dataset[i])

        if len(buffer) == chunk_size:
            out = save_root / f"chunk_{chunk_id:04d}.pt"
            torch.save(buffer, out, _use_new_zipfile_serialization=False)
            buffer = []
            chunk_id += 1

    if buffer:
        out = save_root / f"chunk_{chunk_id:04d}_last.pt"
        torch.save(buffer, out, _use_new_zipfile_serialization=False)

    print(f"Saved {total} samples into chunks at: {save_root}")


# Command functions for argparse subcommands
# this basically handles the two main steps: building the CSVs from folders, and processing the CSV
# into the final chunked .pt files. we can run these separately to avoid conflict
def cmd_make_csv(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = build_pairs_from_folder(data_root / args.train_folder)
    test_df = build_pairs_from_folder(data_root / args.test_folder)

    train_out = out_dir / args.train_name
    test_out = out_dir / args.test_name
    train_df.to_csv(train_out, header=False, index=False)
    test_df.to_csv(test_out, header=False, index=False)

    print(f"Train pairs: {len(train_df)} -> {train_out}")
    print(f"Test pairs: {len(test_df)} -> {test_out}")


# this is the main processing function that does everything in the command line:
# load the CSV, create the dataset, and save the processed samples into chunked .pt files
def cmd_process(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    dataset = NYUDepthDataset(
        csv_path=args.csv_path,
        project_root=project_root,
        size=(args.height, args.width),
    )

    if len(dataset) == 0:
        raise RuntimeError("No valid samples found in dataset after path validation.")

    save_root = Path(args.save_root).resolve()
    save_chunks(dataset, save_root=save_root, chunk_size=args.chunk_size, limit=args.limit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NYUv2 preprocessing and chunk export")
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_csv = subparsers.add_parser("make-csv", help="Build train/test pair CSVs from *_colors/*_depth png folders")
    make_csv.add_argument("--data-root", required=True, help="Dataset root containing split folders")
    make_csv.add_argument("--train-folder", default="train", help="Train split folder name")
    make_csv.add_argument("--test-folder", default="nyu2_test", help="Test split folder name")
    make_csv.add_argument("--out-dir", default="data", help="Output folder for generated CSV files")
    make_csv.add_argument("--train-name", default="nyu2_train.csv", help="Output train CSV filename")
    make_csv.add_argument("--test-name", default="nyu2_test.csv", help="Output test CSV filename")
    make_csv.set_defaults(func=cmd_make_csv)

    process = subparsers.add_parser("process", help="Load pair CSV, preprocess samples, and save chunked .pt files")
    process.add_argument("--csv-path", required=True, help="CSV with rgb/depth paths (2 columns)")
    process.add_argument("--project-root", required=True, help="Project root to resolve relative paths in CSV")
    process.add_argument("--save-root", required=True, help="Output directory for chunked .pt files (external drive path is fine)")
    process.add_argument("--height", type=int, default=256, help="Resize height")
    process.add_argument("--width", type=int, default=256, help="Resize width")
    process.add_argument("--chunk-size", type=int, default=100, help="Samples per saved chunk")
    process.add_argument("--limit", type=int, default=None, help="Optional max number of samples for quick tests")
    process.set_defaults(func=cmd_process)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()