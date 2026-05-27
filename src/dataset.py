"""
DataLoaders for ASL classification.

Primary entry point: build_dataloaders(...)
    Can read either:
            - WebDataset tar shards from shards_dir/{train,val,test}/*.tar
            - Raw images through data/splits/{train,val,test}.csv

ASLDataset is kept for EDA / offline inspection and is also reused by the
raw-image loading path.

Shard formats (per sample inside each tar):
    JPEG (default):  {key}.jpg + {key}.cls
    Fast numpy:      {key}.npy + {key}.cls  (_info.json has "format": "npy")
"""

import hashlib
import io
import json
import pathlib
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SegMode = Literal["none", "mediapipe_crop"]
DatasetSource = Literal["shards", "raw", "auto"]


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


class _NpyTransform:
    """Deserialize a raw .npy bytes blob → PIL image → transform tensor.

    Stored as a callable class so it is picklable by DataLoader workers.
    """

    def __init__(self, transform: Callable) -> None:
        self.transform = transform

    def __call__(self, data: bytes) -> torch.Tensor:
        arr = np.load(io.BytesIO(data))   # uint8 (H, W, 3)
        return self.transform(Image.fromarray(arr))


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
    debug: bool = False,
) -> DataLoader:
    info = load_shard_info(split_dir)
    shard_paths = [str(split_dir / s) for s in info["shards"]]
    total = info["total"]
    fmt = info.get("format", "jpeg")

    if debug:
        print(f"  [DEBUG] _build_split_loader: split={split_dir.name} | shards={len(shard_paths)} | total={total} | fmt={fmt} | num_workers={num_workers}", flush=True)

    wds_base = wds.WebDataset(
        shard_paths, shardshuffle=500 if shuffle else False,
        nodesplitter=wds.split_by_node,
    ).shuffle(1000 if shuffle else 0)

    if fmt == "npy":
        dataset = (
            wds_base
            .to_tuple("npy", "cls")
            .map_tuple(_NpyTransform(transform), _decode_cls)
        )
    else:
        dataset = (
            wds_base
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


def _build_raw_loader(
    csv_path: pathlib.Path,
    data_root: pathlib.Path,
    transform: Callable,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    shuffle: bool,
    drop_last: bool,
    debug: bool = False,
    segmentation_mode: SegMode = "none",
    cache_dir: Optional[str] = None,
) -> DataLoader:
    import platform, time as _time
    if debug:
        print(f"  [DEBUG] _build_raw_loader: csv={csv_path} | num_workers={num_workers} | pin_memory={pin_memory} | OS={platform.system()}", flush=True)

    t0 = _time.perf_counter()
    dataset = ASLDataset(
        csv_path,
        transform=transform,
        data_root=data_root,
        segmentation_mode=segmentation_mode,
        cache_dir=cache_dir,
    )
    if debug:
        print(f"  [DEBUG] ASLDataset built: {len(dataset)} samples in {_time.perf_counter()-t0:.3f}s | seg={segmentation_mode}", flush=True)

    pw = persistent_workers and num_workers > 0
    pf = prefetch_factor if num_workers > 0 else None

    if debug:
        print(f"  [DEBUG] DataLoader: batch_size={batch_size} | workers={num_workers} | persistent={pw} | pin_memory={pin_memory}", flush=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        prefetch_factor=pf,
        drop_last=drop_last,
    )


def _has_shards(shards_dir: pathlib.Path) -> bool:
    return all((shards_dir / split / "_info.json").exists() for split in ("train", "val", "test"))


def build_dataloaders(
    shards_dir: Optional[Union[str, pathlib.Path]] = None,
    *,
    train_transform: Callable,
    eval_transform: Callable,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    use_weighted_sampler: bool = False,
    dataset_source: str = "shards",
    data_root: Optional[Union[str, pathlib.Path]] = None,
    splits_dir: Optional[Union[str, pathlib.Path]] = None,
    debug: bool = False,
    segmentation_mode: SegMode = "none",
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from either WebDataset shards or
    raw-image split CSVs.

    Args:
        shards_dir: Path to the shards root (contains train/, val/, test/).
        train_transform: Transform applied to training images.
        eval_transform: Transform applied to val/test images.
        use_weighted_sampler: Not supported with WebDataset (IterableDataset).
            Use use_class_weights=True in the Trainer for weighted loss instead.
        dataset_source: shards | raw | auto.
        data_root: Root directory used to resolve relative image paths in raw mode.
        splits_dir: Directory that contains train.csv / val.csv / test.csv in raw mode.
    """
    if use_weighted_sampler:
        import warnings
        warnings.warn(
            "use_weighted_sampler is not supported with WebDataset shards. "
            "Use use_class_weights=True in the Trainer for class-weighted loss instead.",
            UserWarning,
        )

    dataset_source = str(dataset_source).strip().lower()
    if dataset_source not in {"shards", "raw", "auto"}:
        raise ValueError("dataset_source must be one of: shards, raw, auto")

    if dataset_source == "auto":
        if shards_dir is not None and _has_shards(pathlib.Path(shards_dir)):
            dataset_source = "shards"
        else:
            dataset_source = "raw"

    if dataset_source == "shards":
        if shards_dir is None:
            raise ValueError("shards_dir is required when dataset_source='shards'")
        shards_dir = pathlib.Path(shards_dir)

        train_loader = _build_split_loader(
            shards_dir / "train", train_transform, batch_size, num_workers,
            pin_memory, prefetch_factor, persistent_workers,
            shuffle=True, drop_last=True, debug=debug,
        )
        val_loader = _build_split_loader(
            shards_dir / "val", eval_transform, batch_size, num_workers,
            pin_memory, prefetch_factor, persistent_workers,
            shuffle=False, drop_last=False, debug=debug,
        )
        test_loader = _build_split_loader(
            shards_dir / "test", eval_transform, batch_size, num_workers,
            pin_memory, prefetch_factor, persistent_workers,
            shuffle=False, drop_last=False, debug=debug,
        )
    else:
        if splits_dir is None:
            raise ValueError("splits_dir is required when dataset_source='raw'")
        if data_root is None:
            raise ValueError("data_root is required when dataset_source='raw'")

        splits_dir = pathlib.Path(splits_dir)
        data_root = pathlib.Path(data_root)

        train_loader = _build_raw_loader(
            splits_dir / "train.csv", data_root, train_transform, batch_size, num_workers,
            pin_memory, prefetch_factor, persistent_workers,
            shuffle=True, drop_last=True,
            debug=debug, segmentation_mode=segmentation_mode, cache_dir=cache_dir,
        )
        val_loader = _build_raw_loader(
            splits_dir / "val.csv", data_root, eval_transform, batch_size, num_workers,
            pin_memory, prefetch_factor, persistent_workers,
            shuffle=False, drop_last=False,
            debug=debug, segmentation_mode=segmentation_mode, cache_dir=cache_dir,
        )
        test_loader = _build_raw_loader(
            splits_dir / "test.csv", data_root, eval_transform, batch_size, num_workers,
            pin_memory, prefetch_factor, persistent_workers,
            shuffle=False, drop_last=False,
            debug=debug, segmentation_mode=segmentation_mode, cache_dir=cache_dir,
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
        # Normalize Windows backslashes so paths work on Linux (Colab/Kaggle)
        p = pathlib.Path(raw_path.replace("\\", "/"))
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
