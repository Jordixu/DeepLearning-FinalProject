"""
WebDataset-based DataLoaders for ASL classification.

Primary entry point: build_dataloaders(shards_dir, ...)
  Reads tar shards produced by reorganize.py from:
      shards_dir/{train,val,test}/*.tar
      shards_dir/{train,val,test}/_info.json

ASLDataset is kept for EDA / offline inspection (reads split CSVs directly).

Shard format (per sample inside each tar):
    {key}.jpg   -- JPEG image
    {key}.cls   -- class_id as ASCII bytes (e.g. b"5")
"""

import hashlib
import json
import pathlib
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SegMode = Literal["none", "mediapipe_crop"]


# ---------------------------------------------------------------------------
# Shard metadata helper
# ---------------------------------------------------------------------------

def load_shard_info(split_dir: Union[str, pathlib.Path]) -> dict:
    """Load _info.json from a split shard directory."""
    info_path = pathlib.Path(split_dir) / "_info.json"
    with open(info_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# WebDataset loader (primary training path)
# ---------------------------------------------------------------------------

def _decode_cls(raw) -> int:
    if isinstance(raw, bytes):
        return int(raw.decode("ascii").strip())
    return int(str(raw).strip())


def _build_split_loader(
    split_dir: pathlib.Path,
    transform: Callable,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    info = load_shard_info(split_dir)
    shard_paths = [str(split_dir / s) for s in info["shards"]]
    total = info["total"]

    dataset = (
        wds.WebDataset(shard_paths, shardshuffle=500 if shuffle else False,
                       nodesplitter=wds.split_by_node)
        .shuffle(1000 if shuffle else 0)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(transform, _decode_cls)
    )

    n_batches = total // batch_size if drop_last else (total + batch_size - 1) // batch_size

    pw = persistent_workers and num_workers > 0
    pf = prefetch_factor if num_workers > 0 else None

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        prefetch_factor=pf,
        drop_last=drop_last,
    ).with_length(n_batches)

    return loader


def build_dataloaders(
    shards_dir: Union[str, pathlib.Path],
    train_transform: Callable,
    eval_transform: Callable,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from WebDataset tar shards.

    Args:
        shards_dir: Path to the shards root (contains train/, val/, test/).
        train_transform: Transform applied to training images.
        eval_transform: Transform applied to val/test images.
        use_weighted_sampler: Not supported with WebDataset (IterableDataset).
            Use use_class_weights=True in the Trainer for weighted loss instead.
    """
    if use_weighted_sampler:
        import warnings
        warnings.warn(
            "use_weighted_sampler is not supported with WebDataset shards. "
            "Use use_class_weights=True in the Trainer for class-weighted loss instead.",
            UserWarning,
        )

    shards_dir = pathlib.Path(shards_dir)

    train_loader = _build_split_loader(
        shards_dir / "train", train_transform, batch_size, num_workers,
        pin_memory, prefetch_factor, persistent_workers,
        shuffle=True, drop_last=True,
    )
    val_loader = _build_split_loader(
        shards_dir / "val", eval_transform, batch_size, num_workers,
        pin_memory, prefetch_factor, persistent_workers,
        shuffle=False, drop_last=False,
    )
    test_loader = _build_split_loader(
        shards_dir / "test", eval_transform, batch_size, num_workers,
        pin_memory, prefetch_factor, persistent_workers,
        shuffle=False, drop_last=False,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# CSV-based dataset (EDA / offline inspection only)
# ---------------------------------------------------------------------------

class ASLDataset(Dataset):
    """
    CSV-based dataset for EDA and offline inspection.
    Not used during training (use build_dataloaders + shards instead).
    """

    def __init__(
        self,
        manifest_csv: Union[str, pathlib.Path],
        transform: Optional[Callable] = None,
        segmentation_mode: SegMode = "none",
        data_root: Optional[Union[str, pathlib.Path]] = None,
        cache_dir: Optional[Union[str, pathlib.Path]] = None,
        cache_resized: bool = False,
        image_size: int = 224,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        self.segmentation_mode = segmentation_mode
        self.data_root = pathlib.Path(data_root) if data_root else None
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None
        self.cache_resized = cache_resized
        self.image_size = image_size
        self._cropper = None

        if cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        row = self.df.iloc[idx]
        filepath = self._resolve(row["filepath"])
        class_id = int(row["class_id"])

        img = self._load_image(filepath)

        if self.segmentation_mode == "mediapipe_crop":
            img = self._mediapipe_crop(img, filepath)

        if self.transform is not None:
            img = self.transform(img)

        return img, class_id

    def _resolve(self, raw_path: str) -> pathlib.Path:
        p = pathlib.Path(raw_path)
        if p.is_absolute():
            return p
        if self.data_root is not None:
            return self.data_root / p
        return p

    def _load_image(self, filepath: pathlib.Path) -> Image.Image:
        if self.cache_resized and self.cache_dir is not None:
            cached = self._resize_cache_path(filepath)
            if cached.exists():
                return Image.open(cached).convert("RGB")
            img = Image.open(filepath).convert("RGB")
            img_small = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            cached.parent.mkdir(parents=True, exist_ok=True)
            img_small.save(cached, format="JPEG", quality=95)
            return img_small
        return Image.open(filepath).convert("RGB")

    def _resize_cache_path(self, filepath: pathlib.Path) -> pathlib.Path:
        key = hashlib.md5(str(filepath).encode()).hexdigest()
        return self.cache_dir / "resize_cache" / f"{key}.jpg"

    def _mediapipe_crop(self, img: Image.Image, filepath: pathlib.Path) -> Image.Image:
        if self.cache_dir is not None:
            cached = self._mp_cache_path(filepath)
            if cached.exists():
                return Image.open(cached).convert("RGB")

        if self._cropper is None:
            from segmentation import MediaPipeHandCropper
            self._cropper = MediaPipeHandCropper()

        cropped = self._cropper.crop(img)

        if self.cache_dir is not None:
            cached = self._mp_cache_path(filepath)
            cached.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(cached, format="JPEG", quality=95)

        return cropped

    def _mp_cache_path(self, filepath: pathlib.Path) -> pathlib.Path:
        key = hashlib.md5(str(filepath).encode()).hexdigest()
        return self.cache_dir / "mediapipe_cache" / f"{key}.jpg"

    @property
    def class_names(self) -> List[str]:
        return sorted(
            self.df["class_name"].unique(),
            key=lambda c: self.df.loc[self.df["class_name"] == c, "class_id"].iloc[0],
        )

    @property
    def class_counts(self) -> Dict[str, int]:
        return self.df["class_name"].value_counts().to_dict()

    @property
    def class_ids(self) -> List[int]:
        return self.df["class_id"].tolist()
