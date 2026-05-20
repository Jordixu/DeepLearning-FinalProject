"""
Reorganize preprocessed images into WebDataset tar shards.

Run ONCE locally after preprocessing.py has produced split CSVs.

Output layout:
    data/asl_shards/
        train/
            train-000000.tar
            train-000001.tar
            ...
            _info.json
        val/
            val-000000.tar
            ...
            _info.json
        test/
            test-000000.tar
            ...
            _info.json

Each sample inside a shard tar:
    {key}.jpg   -- JPEG image (re-encoded, quality 95)
    {key}.cls   -- class_id as ASCII bytes (e.g. b"5")

Train rows are shuffled before writing so each shard is class-diverse.
A per-split _info.json records total count, per-class counts, and the
shard file list -- used by dataset.py and training.py.

Upload asl_shards/ as a single dataset to Kaggle / Google Drive for
cloud training.

Usage:
    python -m src.reorganize
    python -m src.reorganize --config configs/config.yaml
    python -m src.reorganize --shard_size 1000
    python -m src.reorganize --dry_run
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
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from config import Config, load_config

SHARD_SIZE_DEFAULT = 1000


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
        for i, (src, class_id) in enumerate(samples):
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


def reorganize(cfg: Config, shard_size: int = SHARD_SIZE_DEFAULT, dry_run: bool = False) -> None:
    splits_dir = cfg.splits_dir
    shards_dir = cfg.shards_dir

    for split in ("train", "val", "test"):
        csv_path = splits_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"[ERROR] {csv_path} not found. Run preprocessing.py first.")
            sys.exit(1)

    print(f"Shards directory: {shards_dir}")

    if not dry_run:
        shards_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        csv_path = splits_dir / f"{split}.csv"
        df = pd.read_csv(csv_path)
        rows = list(df.itertuples(index=False))

        if split == "train":
            rng = random.Random(cfg.preprocessing.random_seed)
            rng.shuffle(rows)

        total = len(rows)
        n_shards = (total + shard_size - 1) // shard_size

        if dry_run:
            print(f"  {split}: {total:,} images -> {n_shards} shards of {shard_size}")
            continue

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

        print(f"  {split}: {total:,} images -> {len(shard_names)} shards")

    if dry_run:
        print("\nDry run complete - no files written.")
        return

    print(f"\nDone. Upload {shards_dir} to Kaggle / Google Drive as a single dataset.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reorganize images into WebDataset tar shards")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--shard_size", type=int, default=SHARD_SIZE_DEFAULT,
                        help=f"Images per shard (default {SHARD_SIZE_DEFAULT})")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print stats without writing any files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    reorganize(cfg, shard_size=args.shard_size, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
