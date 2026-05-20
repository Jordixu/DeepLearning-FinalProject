"""
Preprocessing pipeline: validation, near-duplicate detection, and
group-aware stratified splitting.

Steps
------
1. Load manifest_raw.csv produced by data_unification.py.
2. Drop images that are corrupted, too small, or wrong format.
3. Compute perceptual hash (pHash) per image.
4. Build duplicate groups via union-find (Hamming distance <= threshold).
5. Assign each group a split (train / val / test) using GroupShuffleSplit
   so no group ever spans two splits - prevents sequential-frame leakage.
6. Write data/splits/{train,val,test}.csv and data/dedup_report.json.

Usage:
    python -m src.preprocessing
    python -m src.preprocessing --config configs/config.yaml
    python -m src.preprocessing --manifest data/manifest_raw.csv --config configs/config.yaml
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from config import Config, PreprocessingConfig, load_config


# Validation

def validate_images(df: pd.DataFrame, min_size: int) -> Tuple[pd.DataFrame, List[str]]:
    """Drop corrupted images and images smaller than min_size in either dim."""
    valid_rows = []
    dropped = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
        path = pathlib.Path(row["filepath"])
        try:
            with Image.open(path) as img:
                w, h = img.size
                if w < min_size or h < min_size:
                    dropped.append(str(path))
                    continue
                valid_rows.append(row)
        except (UnidentifiedImageError, OSError, Exception):
            dropped.append(str(path))
    return pd.DataFrame(valid_rows).reset_index(drop=True), dropped


# pHash computation

def compute_phashes(df: pd.DataFrame, hash_size: int = 8) -> List[imagehash.ImageHash]:
    """Compute perceptual hashes"""
    hashes = []
    for path in tqdm(df["filepath"], desc="Computing pHashes"):
        try:
            with Image.open(path) as img:
                hashes.append(imagehash.phash(img, hash_size=hash_size))
        except Exception:
            hashes.append(imagehash.phash(Image.new("RGB", (8, 8)), hash_size=hash_size))
    return hashes


# find duplicate groups

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def build_duplicate_groups_per_class(
    df: pd.DataFrame,
    hashes: List[imagehash.ImageHash],
    threshold: int,
) -> List[int]:
    """
    Assign group IDs; only compare images within the same class (same sign) to avoid
    false near-duplicate matches between visually similar but distinct signs.
    """
    group_ids = list(range(len(df)))
    class_indices: Dict[int, List[int]] = defaultdict(list)
    for i, cid in enumerate(df["class_id"]):
        class_indices[int(cid)].append(i)

    global_offset = 0
    for cid, indices in tqdm(class_indices.items(), desc="Deduplication per class"):
        n = len(indices)
        uf = UnionFind(n)
        local_hashes = [hashes[i] for i in indices]
        for a in range(n):
            for b in range(a + 1, n):
                if local_hashes[a] - local_hashes[b] <= threshold:
                    uf.union(a, b)
        for local_idx, global_idx in enumerate(indices):
            group_ids[global_idx] = global_offset + uf.find(local_idx)
        global_offset += n

    return group_ids


# split per group

def group_aware_split(
    df: pd.DataFrame,
    group_ids: List[int],
    ratios: Dict[str, float],
    seed: int,
) -> pd.DataFrame:
    df = df.copy()
    df["_group"] = group_ids

    train_ratio = ratios["train"]
    val_ratio = ratios["val"]
    val_test_ratio = 1.0 - train_ratio
    val_of_val_test = val_ratio / val_test_ratio if val_test_ratio > 0 else 0.5

    groups = np.array(group_ids)
    y = df["class_id"].values
    splits_col = np.full(len(df), "train", dtype=object)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=seed)
    train_idx, valtest_idx = next(gss1.split(df, y, groups=groups))

    valtest_groups = groups[valtest_idx]
    valtest_y = y[valtest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=1.0 - val_of_val_test, random_state=seed)
    val_local, test_local = next(gss2.split(df.iloc[valtest_idx], valtest_y, groups=valtest_groups))

    splits_col[valtest_idx[val_local]] = "val"
    splits_col[valtest_idx[test_local]] = "test"
    df["split"] = splits_col
    df = df.drop(columns=["_group"])
    return df


# main

def run_preprocessing(cfg: Config, manifest_path: pathlib.Path) -> None:
    pp: PreprocessingConfig = cfg.preprocessing
    splits_dir = cfg.data_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    total_raw = len(df)
    print(f"  {total_raw:,} images in raw manifest")

    print("\nStep 1: Validating images...")
    df, dropped = validate_images(df, pp.min_image_size)
    print(f"  Dropped (corrupt / too small): {len(dropped):,}  |  Remaining: {len(df):,}")

    print("\nStep 2: Computing perceptual hashes...")
    hashes = compute_phashes(df, hash_size=pp.phash_hash_size)

    print(f"\nStep 3: Building duplicate groups (threshold={pp.phash_threshold})...")
    group_ids = build_duplicate_groups_per_class(df, hashes, pp.phash_threshold)
    n_groups = len(set(group_ids))
    print(f"  Groups: {n_groups:,}")

    per_class_before = df.groupby("class_name").size().to_dict()

    print("\nStep 4: Group-aware stratified split...")
    df = group_aware_split(df, group_ids, pp.split_ratios, pp.random_seed)

    for split in ("train", "val", "test"):
        sub = df[df["split"] == split].drop(columns=["split"])
        out = splits_dir / f"{split}.csv"
        sub.to_csv(out, index=False)
        print(f"  {split}: {len(sub):,} images → {out}")

    per_class_after = df.groupby("class_name").size().to_dict()

    report = {
        "total_raw": total_raw,
        "dropped_invalid": len(dropped),
        "dropped_files_sample": dropped[:50],
        "after_validation": len(df),
        "n_groups": n_groups,
        "phash_threshold": pp.phash_threshold,
        "per_class_before": per_class_before,
        "per_class_after": per_class_after,
        "split_counts": df.groupby(["split", "class_name"]).size().unstack(fill_value=0).to_dict(),
    }
    report_path = cfg.data_root / "dedup_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDedup report → {report_path}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess: validate, dedup, split")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = pathlib.Path(args.manifest) if args.manifest else cfg.data_root / "manifest_raw.csv"

    if not manifest.exists():
        print(f"[ERROR] Manifest not found: {manifest}\nRun data_unification.py first.")
        sys.exit(1)

    run_preprocessing(cfg, manifest)


if __name__ == "__main__":
    main()
