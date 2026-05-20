"""
Reorganize preprocessed images into a clean, portable directory (mainly for cloud based compute).

Run this ONCE locally after preprocessing.py has produced split CSVs.

Output layout:
    data/asl_clean/
        train/
            A/  B/  C/  ...  0/  1/  ...  nothing/
        val/
            A/  B/  C/  ...
        test/
            A/  B/  C/  ...

Each image is copied (not moved) with a deterministic name:
    <source_dataset>_<original_stem>.<ext>
This ensures no filename collisions across datasets and preserves traceability.

The split CSVs are then rewritten to use paths RELATIVE to data_root so that
they work on Kaggle/Colab when you upload asl_clean/ as a single dataset.

Usage:
    python -m src.reorganize
    python -m src.reorganize --config configs/config.yaml
    python -m src.reorganize --config configs/config.yaml --copy   (default)
    python -m src.reorganize --config configs/config.yaml --dry_run
"""

import argparse
import pathlib
import shutil
import sys
from typing import Dict

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from config import Config, load_config


def reorganize(cfg: Config, dry_run: bool = False) -> None:
    splits_dir = cfg.splits_dir
    clean_dir = cfg.clean_dir

    for split in ("train", "val", "test"):
        csv_path = splits_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"[ERROR] {csv_path} not found. Run preprocessing.py first.")
            sys.exit(1)

    print(f"Clean directory: {clean_dir}")
    if not dry_run:
        clean_dir.mkdir(parents=True, exist_ok=True)

    updated_dfs: Dict[str, pd.DataFrame] = {}

    for split in ("train", "val", "test"):
        csv_path = splits_dir / f"{split}.csv"
        df = pd.read_csv(csv_path)
        new_paths = []

        split_dir = clean_dir / split
        if not dry_run:
            split_dir.mkdir(parents=True, exist_ok=True)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split}"):
            src = pathlib.Path(row["filepath"])
            class_name = row["class_name"]
            source = row["source_dataset"]

            dst_dir = split_dir / class_name
            if not dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)

            # Deterministic name: {source}_{original_stem}{ext}
            ext = src.suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                ext = ".jpg"
            dst_name = f"{source}_{src.stem}{ext}"
            dst = dst_dir / dst_name

            if not dry_run and not dst.exists():
                shutil.copy2(src, dst)

            # Store path relative to data_root for portability
            try:
                rel = dst.relative_to(cfg.data_root)
            except ValueError:
                rel = dst  # fallback: keep absolute if outside data_root
            new_paths.append(str(rel))

        df["filepath"] = new_paths
        updated_dfs[split] = df

    if dry_run:
        print("\nDry run complete - no files copied.")
        for split, df in updated_dfs.items():
            print(f"  {split}: {len(df):,} images would be written")
        return

    # Rewrite split CSVs with relative paths
    for split, df in updated_dfs.items():
        out = splits_dir / f"{split}.csv"
        df.to_csv(out, index=False)
        print(f"Updated {out} with relative paths")

    print(f"\nDone. Upload {clean_dir} to Kaggle / Google Drive as a single dataset.")
    print(f"Set kaggle.data_root in config.yaml to the parent of asl_clean/ on Kaggle.")
    _print_summary(clean_dir)


def _print_summary(clean_dir: pathlib.Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = clean_dir / split
        if not split_dir.exists():
            continue
        total = sum(1 for _ in split_dir.rglob("*") if _.is_file())
        print(f"  {split}: {total:,} images")


def main():
    parser = argparse.ArgumentParser(description="Reorganize images into asl_clean/")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Simulate without copying")
    args = parser.parse_args()

    cfg = load_config(args.config)
    reorganize(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
