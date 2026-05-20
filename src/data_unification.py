"""
Dataset unification: walk each raw dataset, apply the canonical class map,
and write data/manifest_raw.csv.

This script reads from the raw dataset directories (local only).
After running, pipe the output through preprocessing.py → reorganize.py to
produce the clean portable asl_clean/ directory.

Usage:
    python -m src.data_unification
    python -m src.data_unification --config configs/config.yaml
    python -m src.data_unification --config configs/config.yaml --out data/manifest_raw.csv

Manifest columns:
    filepath, class_id, class_name, source_dataset, original_split
"""

import argparse
import pathlib
import sys
from typing import List

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from config import Config, load_config


# Per dataset walkers

def _walk_flat(
    root: pathlib.Path,
    class_map: dict,
    class_to_idx: dict,
    source: str,
    original_split: str = "all",
) -> List[dict]:
    """Walk a flat class-folder structure: root/class_name/img.jpg"""
    rows = []
    if not root.exists():
        print(f"  [WARN] {root} does not exist - skipping{source}")
        return rows
    for class_folder in sorted(root.iterdir()):
        if not class_folder.is_dir():
            continue
        canonical = class_map.get(class_folder.name)
        if canonical is None:
            continue  # not in map → skip (e.g. del, space)
        class_id = class_to_idx[canonical]
        for img_path in class_folder.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                rows.append({
                    "filepath": str(img_path.resolve()),
                    "class_id": class_id,
                    "class_name": canonical,
                    "source_dataset": source,
                    "original_split": original_split,
                })
    return rows


def _walk_split(
    root: pathlib.Path,
    class_map: dict,
    class_to_idx: dict,
    source: str,
    splits: List[str],
) -> List[dict]:
    """Walk a pre-split structure: root/split/class_name/img.jpg"""
    rows = []
    if not root.exists():
        print(f"  [WARN] {root} does not exist - skipping{source}")
        return rows
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        rows.extend(_walk_flat(split_dir, class_map, class_to_idx, source, original_split=split))
    return rows


# unification

def unify(cfg: Config, out_path: pathlib.Path) -> pd.DataFrame:
    rows: List[dict] = []

    # 1. combine_asl_dataset - flat, no preset split
    src = cfg.raw_dataset_paths.get("combine_asl")
    if src:
        cmap = cfg.class_maps.get("combine_asl", {})
        print(f"Scanning combine_asl → {src}")
        rows.extend(_walk_flat(src, cmap, cfg.class_to_idx, "combine_asl"))
    else:
        print("[WARN] combine_asl not found in raw_dataset_paths")

    # 2. ASL-HG raw - flat, no preset split
    src = cfg.raw_dataset_paths.get("asl_hg_raw")
    if src:
        cmap = cfg.class_maps.get("asl_hg_raw", {})
        print(f"Scanning asl_hg_raw → {src}")
        # Check if there are split subdirectories or flat class folders
        if (src / "train").exists():
            rows.extend(_walk_split(src, cmap, cfg.class_to_idx, "asl_hg_raw", ["train", "test"]))
        else:
            rows.extend(_walk_flat(src, cmap, cfg.class_to_idx, "asl_hg_raw"))
    else:
        print("[WARN] asl_hg_raw not found in raw_dataset_paths")

    # 3. ASL Alphabet - flat, letters only + nothing (del/space skipped via class_map)
    src = cfg.raw_dataset_paths.get("asl_alphabet")
    if src:
        cmap = cfg.class_maps.get("asl_alphabet", {})
        print(f"Scanning asl_alphabet → {src}")
        rows.extend(_walk_flat(src, cmap, cfg.class_to_idx, "asl_alphabet"))
    else:
        print("[WARN] asl_alphabet not found in raw_dataset_paths")

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] No images found. Check that data_root and raw_dataset paths are correct.")
        return df

    print(f"\nTotal images: {len(df):,}")
    print("\nPer source:")
    print(df.groupby("source_dataset").size().to_string())
    print("\nPer class:")
    print(df.groupby("class_name").size().sort_values(ascending=False).to_string())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nManifest written → {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build unified manifest CSV from raw datasets")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out", default=None, help="Output CSV (default: <data_root>/manifest_raw.csv)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = pathlib.Path(args.out) if args.out else cfg.data_root / "manifest_raw.csv"
    unify(cfg, out)


if __name__ == "__main__":
    main()
