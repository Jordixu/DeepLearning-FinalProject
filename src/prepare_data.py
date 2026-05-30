"""
Data preparation pipeline: unify raw sources, validate images, assign splits
by source dataset, and pack into WebDataset tar shards.

Run once locally (no GPU needed) to produce data/asl_shards/ for cloud training.

Split assignment is fully controlled by preprocessing.dataset_splits in
configs/config.yaml:
    dataset_splits:
      combine_asl: train
      asl_hg_raw: val
      asl_alphabet: test

Each source dataset becomes exactly one split - no random splitting is performed.

Steps
------
1. Walk each raw dataset directory, apply canonical class map  →  manifest_raw.csv
2. Validate images (drop corrupt / too small)
3. Assign each row its split based on source_dataset → config mapping
4. Write data/splits/asl_clean/{train,val,test}.csv + preprocessing_report.json
5. Pack each split into WebDataset tar shards under data/asl_shards/

Usage:
    python -m src.prepare_data
    python -m src.prepare_data --config configs/config.yaml
    python -m src.prepare_data --skip_shards      # stop after writing split CSVs
    python -m src.prepare_data --dry_run           # print stats, write nothing
"""

import argparse
import io
import json
import pathlib
import random
import sys
import tarfile
from typing import Dict, List

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from config import Config, load_config


# ---------------------------------------------------------------------------
# Step 1: Dataset unification
# ---------------------------------------------------------------------------

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
        print(f"  [WARN] {root} does not exist - skipping {source}")
        return rows
    for class_folder in sorted(root.iterdir()):
        if not class_folder.is_dir():
            continue
        canonical = class_map.get(class_folder.name)
        if canonical is None:
            continue
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
        print(f"  [WARN] {root} does not exist - skipping {source}")
        return rows
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        rows.extend(_walk_flat(split_dir, class_map, class_to_idx, source, original_split=split))
    return rows


def _build_manifest(cfg: Config, out_path: pathlib.Path) -> pd.DataFrame:
    rows: List[dict] = []

    src = cfg.raw_dataset_paths.get("combine_asl")
    if src:
        print(f"Scanning combine_asl → {src}")
        rows.extend(_walk_flat(src, cfg.class_maps.get("combine_asl", {}), cfg.class_to_idx, "combine_asl"))
    else:
        print("[WARN] combine_asl not found in raw_dataset_paths")

    src = cfg.raw_dataset_paths.get("asl_hg_raw")
    if src:
        print(f"Scanning asl_hg_raw → {src}")
        if (src / "train").exists():
            rows.extend(_walk_split(src, cfg.class_maps.get("asl_hg_raw", {}), cfg.class_to_idx, "asl_hg_raw", ["train", "test"]))
        else:
            rows.extend(_walk_flat(src, cfg.class_maps.get("asl_hg_raw", {}), cfg.class_to_idx, "asl_hg_raw"))
    else:
        print("[WARN] asl_hg_raw not found in raw_dataset_paths")

    src = cfg.raw_dataset_paths.get("asl_alphabet")
    if src:
        print(f"Scanning asl_alphabet → {src}")
        rows.extend(_walk_flat(src, cfg.class_maps.get("asl_alphabet", {}), cfg.class_to_idx, "asl_alphabet"))
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


# ---------------------------------------------------------------------------
# Step 2: Image validation
# ---------------------------------------------------------------------------

def _validate_images(df: pd.DataFrame, min_size: int) -> tuple:
    """Drop corrupted images and images smaller than min_size in either dimension."""
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


# ---------------------------------------------------------------------------
# Step 3: Split assignment from config
# ---------------------------------------------------------------------------

def _assign_splits(df: pd.DataFrame, dataset_splits: Dict[str, str]) -> pd.DataFrame:
    """Assign each row its split based on source_dataset → config mapping."""
    df = df.copy()
    df["split"] = df["source_dataset"].map(dataset_splits)
    missing = df["split"].isna()
    if missing.any():
        unknown = df.loc[missing, "source_dataset"].unique().tolist()
        print(f"[WARN] {missing.sum():,} rows have no split assignment "
              f"(source_datasets: {unknown}) - dropping")
        df = df[~missing].copy()
    return df


# ---------------------------------------------------------------------------
# Step 3b: Balance train sources
# ---------------------------------------------------------------------------

def _balance_train_sources(
    df: pd.DataFrame,
    seed: int,
    dominant_source: str = "combine_asl",
) -> pd.DataFrame:
    """
    Undersample the dominant source in the train split so all sources
    contribute equally (by total count).  Other splits are untouched.
    Prevents the model from overfitting to one dataset's visual style.
    """
    train_mask = df["split"] == "train"
    train_df = df[train_mask]
    other_df = df[~train_mask]

    minority_count = (
        train_df[train_df["source_dataset"] != dominant_source]
        .shape[0]
    )
    if minority_count == 0:
        return df  # nothing to balance

    dominant_df = train_df[train_df["source_dataset"] == dominant_source]
    minority_df = train_df[train_df["source_dataset"] != dominant_source]

    if len(dominant_df) > minority_count:
        dominant_df = dominant_df.sample(n=minority_count, random_state=seed)
        print(
            f"  [balance] Undersampled '{dominant_source}' in train: "
            f"{len(train_df[train_df['source_dataset'] == dominant_source]):,} → {minority_count:,}  "
            f"(minority total: {minority_count:,})"
        )

    balanced_train = pd.concat([dominant_df, minority_df], ignore_index=True)
    return pd.concat([balanced_train, other_df], ignore_index=True)


# ---------------------------------------------------------------------------
# Step 4: Write split CSVs and preprocessing report
# ---------------------------------------------------------------------------

def _write_splits(
    df: pd.DataFrame,
    splits_dir: pathlib.Path,
    data_root: pathlib.Path,
    total_raw: int,
    dropped: List[str],
) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split].drop(columns=["split"])
        out = splits_dir / f"{split}.csv"
        sub.to_csv(out, index=False)
        print(f"  {split}: {len(sub):,} images → {out}")

    report = {
        "total_raw": total_raw,
        "dropped_invalid": len(dropped),
        "dropped_files_sample": dropped[:50],
        "after_validation": len(df),
        "split_counts": df.groupby(["split", "class_name"]).size().unstack(fill_value=0).to_dict(),
    }
    report_path = data_root / "preprocessing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport → {report_path}")


# ---------------------------------------------------------------------------
# Step 5: Pack into WebDataset tar shards
# ---------------------------------------------------------------------------

def _resolve_src(raw: str, data_root: pathlib.Path, clean_dir: pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(raw)
    if p.is_absolute() and p.exists():
        return p
    for base in (data_root, clean_dir):
        candidate = base / p
        if candidate.exists():
            return candidate
    return data_root / p


def _to_jpeg_bytes(img_path: pathlib.Path) -> bytes:
    img = Image.open(img_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _write_shard(shard_path: pathlib.Path, samples: list, start_idx: int) -> int:
    written = 0
    with tarfile.open(shard_path, "w") as tar:
        for i, (src, class_id) in enumerate(tqdm(samples, desc=f"  {shard_path.stem}", leave=False)):
            key = f"{start_idx + i:08d}"
            try:
                img_bytes = _to_jpeg_bytes(src)
            except Exception as exc:
                print(f"[WARN] Skipping {src}: {exc}")
                continue
            _add_bytes(tar, f"{key}.jpg", img_bytes)
            _add_bytes(tar, f"{key}.cls", str(class_id).encode("ascii"))
            written += 1
    return written


def _pack_shards(cfg: Config, splits_dir: pathlib.Path, shard_size: int) -> None:
    shards_dir = cfg.shards_dir
    print(f"\nShards directory: {shards_dir}")
    shards_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        csv_path = splits_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"[SKIP] {csv_path} not found - skipping shard for {split}")
            continue
        df = pd.read_csv(csv_path)
        rows = list(df.itertuples(index=False))

        if split == "train":
            rng = random.Random(cfg.preprocessing.random_seed)
            rng.shuffle(rows)

        total = len(rows)
        split_dir = shards_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        class_counts: Dict[str, int] = {}
        shard_names: List[str] = []
        shard_idx = 0
        batch: list = []

        for i, row in enumerate(tqdm(rows, total=total, desc=f"  {split}")):
            src = _resolve_src(row.filepath, cfg.data_root, cfg.clean_dir)
            class_id = int(row.class_id)
            class_counts[str(class_id)] = class_counts.get(str(class_id), 0) + 1
            batch.append((src, class_id))

            flush = len(batch) == shard_size or i == total - 1
            if flush and batch:
                shard_name = f"{split}-{shard_idx:06d}.tar"
                shard_path = split_dir / shard_name
                _write_shard(shard_path, batch, shard_idx * shard_size)
                shard_names.append(shard_name)
                shard_idx += 1
                batch = []

        info = {
            "split": split,
            "total": total,
            "num_shards": len(shard_names),
            "shard_size": shard_size,
            "class_counts": class_counts,
            "shards": shard_names,
        }
        with open(split_dir / "_info.json", "w") as f:
            json.dump(info, f, indent=2)
        print(f"  {split}: {total:,} images → {len(shard_names)} shards")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(cfg: Config, skip_shards: bool = False, dry_run: bool = False, skip_validation: bool = False) -> None:
    pp = cfg.preprocessing
    manifest_path = cfg.data_root / "manifest_raw.csv"
    splits_dir = cfg.splits_dir

    print("=" * 60)
    print("Step 1: Building raw manifest...")
    df = _build_manifest(cfg, manifest_path)
    if df.empty:
        return
    total_raw = len(df)

    print("\n" + "=" * 60)
    print("Step 2: Validating images...")
    if skip_validation:
        print("  Skipping validation (--skip_validation set).")
        dropped = []
    else:
        df, dropped = _validate_images(df, pp.min_image_size)
        print(f"  Dropped (corrupt / too small): {len(dropped):,}  |  Remaining: {len(df):,}")

    print("\n" + "=" * 60)
    print("Step 3: Assigning splits from config...")
    print(f"  Mapping: {pp.dataset_splits}")
    df = _assign_splits(df, pp.dataset_splits)
    for split in ("train", "val", "test"):
        n = (df["split"] == split).sum()
        src = [k for k, v in pp.dataset_splits.items() if v == split]
        print(f"  {split}: {n:,} images  (source: {src})")

    print("\n" + "=" * 60)
    print("Step 3b: Balancing train sources...")
    df = _balance_train_sources(df, seed=pp.random_seed)
    print(f"  train: {(df['split'] == 'train').sum():,} images after balancing")

    if dry_run:
        print("\nDry run - no files written.")
        return

    print("\n" + "=" * 60)
    print("Step 4: Writing split CSVs...")
    _write_splits(df, splits_dir, cfg.data_root, total_raw, dropped)

    if skip_shards:
        print("\nDone (shards skipped).")
        return

    print("\n" + "=" * 60)
    print("Step 5: Packing WebDataset shards...")
    _pack_shards(cfg, splits_dir, pp.shard_size)

    print(f"\nDone. Upload {cfg.shards_dir} to Kaggle / Google Drive as a single dataset.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full data preparation pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip_shards", action="store_true",
                        help="Stop after writing split CSVs, skip sharding")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print stats without writing any files")
    parser.add_argument("--skip_validation", action="store_true",
                        help="Skip image validation (faster, skips corrupt/small-file check)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    prepare(cfg, skip_shards=args.skip_shards, dry_run=args.dry_run, skip_validation=args.skip_validation)


if __name__ == "__main__":
    main()
