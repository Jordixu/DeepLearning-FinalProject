"""
Data preparation pipeline for the merged ASL dataset.

Reads data/merged_dataset/{class}/ and produces:
  - data/splits/merged/{train,val,test}.csv  : stratified 70/15/15 split
  - data/norm_stats.json                     : per-channel mean & std (train only)
  - data/asl_shards/{train,val,test}/*.tar   : WebDataset shards (optional)

Mini dataset mode
-----------------
Pass --mini N to sample N images per class from merged_dataset/ into
asl_clean_mini/, then generate split CSVs for local dev/testing.
No shards are produced in mini mode.

    python -m src.prepare_data --mini 10

Splitting strategy
------------------
Each class is shuffled independently with a high-quality PRNG (NumPy PCG64),
then split 70 / 15 / 15.  The per-split lists are globally shuffled again so
that shards contain a mix of all classes.  No source-dataset bias is applied.

Usage
-----
    python -m src.prepare_data
    python -m src.prepare_data --config configs/config.yaml
    python -m src.prepare_data --mini N          # build mini dataset (N images/class)
    python -m src.prepare_data --skip_shards      # stop after CSVs + norm stats
    python -m src.prepare_data --skip_validation  # skip corrupt/size check
    python -m src.prepare_data --dry_run          # print stats only
"""

import argparse
import io
import json
import pathlib
import random
import shutil
import sys
import tarfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from config import Config, load_config


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Step 0 (optional): Sample mini dataset from merged_dataset/
# ---------------------------------------------------------------------------

def _sample_mini(cfg: Config, n_per_class: int) -> None:
    merged_dir = cfg.merged_dir
    clean_dir = cfg.clean_dir

    if not merged_dir.exists():
        raise FileNotFoundError(
            f"merged_dataset not found at {merged_dir}. "
            "Run scripts/merge_datasets.py first."
        )

    rng = random.Random(cfg.preprocessing.random_seed)
    clean_dir.mkdir(parents=True, exist_ok=True)

    for cls_dir in sorted(merged_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        if cls_name not in cfg.class_to_idx:
            continue
        imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        sample = rng.sample(imgs, min(n_per_class, len(imgs)))
        out_dir = clean_dir / cls_name
        out_dir.mkdir(exist_ok=True)
        for img in sample:
            shutil.copy2(img, out_dir / img.name)
        print(f"  {cls_name}: {len(sample)} images -> {out_dir}")

    print(f"Mini dataset written to {clean_dir}")


# ---------------------------------------------------------------------------
# Step 1: Build manifest from source directory
# ---------------------------------------------------------------------------

def _build_manifest(source_dir: pathlib.Path, cfg: Config) -> pd.DataFrame:
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source directory not found at {source_dir}."
        )

    rows: List[dict] = []
    for cls_dir in sorted(source_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        if cls_name not in cfg.class_to_idx:
            print(f"  [WARN] Unknown class folder '{cls_name}', skipping")
            continue
        cls_id = cfg.class_to_idx[cls_name]
        for img in cls_dir.iterdir():
            if img.suffix.lower() in IMAGE_EXTS:
                rows.append({
                    "filepath": str(img.resolve()),
                    "class_name": cls_name,
                    "class_id": cls_id,
                })

    df = pd.DataFrame(rows)
    print(f"Total images found: {len(df):,}  ({df['class_name'].nunique()} classes)")
    print("\nPer class:")
    print(df.groupby("class_name").size().sort_values(ascending=False).to_string())
    return df


# ---------------------------------------------------------------------------
# Step 2: Image validation
# ---------------------------------------------------------------------------

def _validate_images(df: pd.DataFrame, min_size: int) -> Tuple[pd.DataFrame, List[str]]:
    valid, dropped = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
        path = pathlib.Path(row["filepath"])
        try:
            with Image.open(path) as img:
                w, h = img.size
                if w >= min_size and h >= min_size:
                    valid.append(row)
                else:
                    dropped.append(str(path))
        except (UnidentifiedImageError, OSError, Exception):
            dropped.append(str(path))
    return pd.DataFrame(valid).reset_index(drop=True), dropped


# ---------------------------------------------------------------------------
# Step 3: Stratified 70/15/15 split
# ---------------------------------------------------------------------------

def _stratified_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> pd.DataFrame:
    """
    Per-class stratified split with a high-quality shuffle.

    Each class is shuffled independently using NumPy's PCG64 generator,
    then sliced at the 70 / 85 / 100 percentile boundaries.  After
    concatenation the three splits are globally shuffled again so that
    consecutive samples in every shard span all classes.
    """
    rng = np.random.default_rng(seed)

    train_rows, val_rows, test_rows = [], [], []

    for cls_name, grp in df.groupby("class_name"):
        indices = grp.index.to_numpy().copy()
        # Two independent shuffles for thorough randomisation
        rng.shuffle(indices)
        rng.shuffle(indices)

        n = len(indices)
        n_train = max(1, round(n * train_frac))
        n_val = max(1, round(n * val_frac))
        # Remaining goes to test; at minimum 1 sample per split for large classes
        if n >= 3:
            n_test = n - n_train - n_val
            if n_test < 1:
                n_val -= 1
                n_test = 1
        else:
            # Tiny classes: put everything in train
            n_train, n_val, n_test = n, 0, 0

        train_rows.append(grp.loc[indices[:n_train]])
        if n_val > 0:
            val_rows.append(grp.loc[indices[n_train:n_train + n_val]])
        if n_test > 0:
            test_rows.append(grp.loc[indices[n_train + n_val:]])

    def _concat_and_shuffle(parts: list) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame()
        combined = pd.concat(parts, ignore_index=True)
        # Three global shuffles for extra entropy
        idx = combined.index.to_numpy().copy()
        rng.shuffle(idx)
        rng.shuffle(idx)
        rng.shuffle(idx)
        return combined.iloc[idx].reset_index(drop=True)

    train_df = _concat_and_shuffle(train_rows)
    val_df = _concat_and_shuffle(val_rows)
    test_df = _concat_and_shuffle(test_rows)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    result = pd.concat([train_df, val_df, test_df], ignore_index=True)

    for split in ("train", "val", "test"):
        n = (result["split"] == split).sum()
        print(f"  {split:5s}: {n:6,} images")

    return result


# ---------------------------------------------------------------------------
# Step 4: Write split CSVs
# ---------------------------------------------------------------------------

def _write_splits(df: pd.DataFrame, splits_dir: pathlib.Path) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split].drop(columns=["split"])
        out = splits_dir / f"{split}.csv"
        sub.to_csv(out, index=False)
        print(f"  {split}.csv -> {len(sub):,} rows  ->  {out}")


# ---------------------------------------------------------------------------
# Step 5: Compute normalisation statistics from train set only
# ---------------------------------------------------------------------------

def _compute_norm_stats(
    train_df: pd.DataFrame,
    image_size: int,
    max_samples: int = 8000,
    seed: int = 42,
) -> dict:
    """
    Compute per-channel (R, G, B) mean and std over the training set.

    Images are resized to image_size x image_size before computing statistics
    so the values match what the model will see after the resize transform.
    Only train samples are used; val/test images must not influence the stats.
    """
    rng = np.random.default_rng(seed)
    paths = train_df["filepath"].to_numpy().copy()
    rng.shuffle(paths)
    paths = paths[:max_samples]

    # Welford online algorithm for numerical stability
    count = 0
    mean = np.zeros(3, dtype=np.float64)
    M2 = np.zeros(3, dtype=np.float64)

    failed = 0
    for path in tqdm(paths, desc="Computing norm stats"):
        try:
            with Image.open(path) as img:
                img = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
                arr = np.array(img, dtype=np.float64) / 255.0  # (H, W, 3)
                pixels = arr.reshape(-1, 3)                      # (H*W, 3)
                for pixel in pixels:
                    count += 1
                    delta = pixel - mean
                    mean += delta / count
                    M2 += delta * (pixel - mean)
        except Exception:
            failed += 1

    if failed:
        print(f"  [WARN] {failed} images skipped during norm stats computation")

    std = np.sqrt(M2 / max(count - 1, 1))

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_samples": int(len(paths) - failed),
        "n_pixels": int(count),
        "image_size": image_size,
    }
    print(f"  mean (R/G/B): {mean[0]:.4f} / {mean[1]:.4f} / {mean[2]:.4f}")
    print(f"  std  (R/G/B): {std[0]:.4f} / {std[1]:.4f} / {std[2]:.4f}")
    print(f"  computed over {count:,} pixels from {len(paths) - failed:,} images")
    return stats


# ---------------------------------------------------------------------------
# Step 6: Pack into WebDataset tar shards
# ---------------------------------------------------------------------------

def _to_jpeg_bytes(img_path: pathlib.Path) -> bytes:
    with Image.open(img_path) as img:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        return buf.getvalue()


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _pack_shards(
    cfg: Config,
    splits_dir: pathlib.Path,
    shard_size: int,
) -> None:
    shards_dir = cfg.shards_dir
    shards_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        csv_path = splits_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {csv_path} missing")
            continue

        df = pd.read_csv(csv_path)
        rows = list(df.itertuples(index=False))

        # Train is already globally shuffled; re-shuffle from CSV for good measure
        if split == "train":
            rng = np.random.default_rng(cfg.preprocessing.random_seed)
            perm = np.arange(len(rows))
            rng.shuffle(perm)
            rows = [rows[i] for i in perm]

        split_dir = shards_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        shard_idx = 0
        batch: list = []
        shard_names: List[str] = []
        class_counts: Dict[str, int] = {}

        for i, row in enumerate(tqdm(rows, desc=f"  Sharding {split}")):
            src = pathlib.Path(row.filepath)
            if not src.exists():
                src = cfg.data_root / row.filepath
            class_id = int(row.class_id)
            class_counts[str(class_id)] = class_counts.get(str(class_id), 0) + 1
            batch.append((src, class_id))

            if len(batch) == shard_size or i == len(rows) - 1:
                shard_name = f"{split}-{shard_idx:06d}.tar"
                shard_path = split_dir / shard_name
                written = 0
                with tarfile.open(shard_path, "w") as tar:
                    for j, (img_path, cls_id) in enumerate(
                        tqdm(batch, desc=f"    {shard_name}", leave=False)
                    ):
                        key = f"{shard_idx * shard_size + j:08d}"
                        try:
                            _add_bytes(tar, f"{key}.jpg", _to_jpeg_bytes(img_path))
                            _add_bytes(tar, f"{key}.cls", str(cls_id).encode())
                            written += 1
                        except Exception as exc:
                            print(f"[WARN] {img_path}: {exc}")
                shard_names.append(shard_name)
                shard_idx += 1
                batch = []

        info = {
            "split": split,
            "total": len(rows),
            "num_shards": len(shard_names),
            "shard_size": shard_size,
            "class_counts": class_counts,
            "shards": shard_names,
        }
        (split_dir / "_info.json").write_text(json.dumps(info, indent=2))
        print(f"  {split}: {len(rows):,} images -> {len(shard_names)} shards in {split_dir}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare(
    cfg: Config,
    skip_shards: bool = False,
    dry_run: bool = False,
    skip_validation: bool = False,
    mini_n=None,
) -> None:
    pp = cfg.preprocessing

    if mini_n is not None:
        print("=" * 60)
        print(f"Step 0: Sampling {mini_n} images/class into {cfg.clean_dir} ...")
        _sample_mini(cfg, mini_n)
        source_dir = cfg.clean_dir
        splits_dir = cfg.data_root / "splits" / "asl_clean_mini"
        skip_shards = True
    else:
        source_dir = cfg.merged_dir
        splits_dir = cfg.splits_dir

    print("=" * 60)
    print(f"Step 1: Scanning {source_dir.name}/ ...")
    df = _build_manifest(source_dir, cfg)

    print("\n" + "=" * 60)
    print("Step 2: Validating images ...")
    if skip_validation:
        print("  Skipping (--skip_validation set)")
        dropped: List[str] = []
    else:
        df, dropped = _validate_images(df, pp.min_image_size)
        print(f"  Dropped: {len(dropped):,}  |  Remaining: {len(df):,}")

    print("\n" + "=" * 60)
    ratios = pp.split_ratios
    print(
        f"Step 3: Stratified split  "
        f"train={ratios['train']:.0%}  val={ratios['val']:.0%}  "
        f"test={ratios['test']:.0%}  (seed={pp.random_seed}) ..."
    )
    df = _stratified_split(df, ratios["train"], ratios["val"], pp.random_seed)

    if dry_run:
        print("\nDry run: no files written.")
        return

    print("\n" + "=" * 60)
    print("Step 4: Writing split CSVs ...")
    _write_splits(df, splits_dir)

    print("\n" + "=" * 60)
    print("Step 5: Computing normalisation statistics (train set only) ...")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    norm_stats = _compute_norm_stats(
        train_df,
        image_size=pp.image_size,
        max_samples=8000,
        seed=pp.random_seed,
    )
    norm_path = cfg.data_root / "norm_stats.json"
    norm_path.write_text(json.dumps(norm_stats, indent=2))
    print(f"  Saved -> {norm_path}")

    # Write preprocessing report
    report = {
        "total_images": len(df),
        "dropped_invalid": len(dropped),
        "split_counts": {
            split: int((df["split"] == split).sum())
            for split in ("train", "val", "test")
        },
        "class_split_counts": (
            df.groupby(["class_name", "split"])
            .size()
            .unstack(fill_value=0)
            .to_dict()
        ),
        "norm_stats": norm_stats,
    }
    report_path = cfg.data_root / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Report -> {report_path}")

    if skip_shards:
        print("\nDone (shards skipped).")
        return

    print("\n" + "=" * 60)
    print("Step 6: Packing WebDataset shards ...")
    _pack_shards(cfg, splits_dir, pp.shard_size)

    print(f"\nAll done.  Upload {cfg.shards_dir} to Kaggle / Google Drive.")


def main() -> None:
    parser = argparse.ArgumentParser(description="ASL merged dataset preparation pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mini", type=int, metavar="N", default=None,
                        help="Sample N images per class into asl_clean_mini/ and generate split CSVs")
    parser.add_argument("--skip_shards", action="store_true")
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    prepare(
        cfg,
        skip_shards=args.skip_shards,
        skip_validation=args.skip_validation,
        dry_run=args.dry_run,
        mini_n=args.mini,
    )


if __name__ == "__main__":
    main()
