"""
Manifest-CSV-driven PyTorch Dataset for ASL classification.

Reads split CSVs produced by preprocessing.py (then rewritten by reorganize.py).

After reorganize.py, filepaths in the CSV are RELATIVE to data_root.
Pass data_root so the dataset can resolve them correctly on any machine.
Absolute paths are used as-is (backward compatible with pre-reorganize CSVs).

Segmentation modes:
  - "none": load image as-is
  - "mediapipe_crop": apply MediaPipeHandCropper on-the-fly (with caching)

The CSV must have at minimum these columns:
  filepath, class_id, class_name, source_dataset, original_split
"""

import hashlib
import pathlib
from typing import Callable, Dict, List, Literal, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


SegMode = Literal["none", "mediapipe_crop"]


class ASLDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | pathlib.Path,
        transform: Optional[Callable] = None,
        segmentation_mode: SegMode = "none",
        data_root: Optional[str | pathlib.Path] = None,
        cache_dir: Optional[str | pathlib.Path] = None,
        cache_resized: bool = False,
        image_size: int = 224,
    ):
        """
        Args:
            manifest_csv: Path to a split CSV (train.csv / val.csv / test.csv).
            transform: torchvision transform applied after loading / cropping.
            segmentation_mode: "none" or "mediapipe_crop".
            data_root: Root used to resolve relative filepaths in the CSV.
                       After reorganize.py, paths are relative to data_root.
                       Absolute paths in the CSV are used as-is.
            cache_dir: Directory for MediaPipe crop cache and resize cache.
            cache_resized: Cache 224×224 resized images on first access.
                           Big win on Colab with slow Drive I/O.
            image_size: Target size (square) for the resize cache.
        """
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        self.segmentation_mode = segmentation_mode
        self.data_root = pathlib.Path(data_root) if data_root else None
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None
        self.cache_resized = cache_resized
        self.image_size = image_size

        self._cropper = None  # lazy-loaded MediaPipeHandCropper

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

    # helpers

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

    # properties for EDA and utils

    @property
    def class_names(self) -> List[str]:
        return sorted(self.df["class_name"].unique(), key=lambda c: self.df.loc[self.df["class_name"] == c, "class_id"].iloc[0])

    @property
    def class_counts(self) -> Dict[str, int]:
        return self.df["class_name"].value_counts().to_dict()

    @property
    def class_ids(self) -> List[int]:
        return self.df["class_id"].tolist()


def build_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    train_transform,
    eval_transform,
    data_root: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    segmentation_mode: SegMode = "none",
    cache_dir: Optional[str] = None,
    cache_resized: bool = False,
    image_size: int = 224,
    use_weighted_sampler: bool = False,
):
    """
    Build train / val / test DataLoaders from manifest CSVs.

    On Windows, DataLoaders with num_workers > 0 must be created inside
    a main() or __name__ == '__main__' guard to avoid spawn issues.
    This function is safe to call from a notebook or a guarded main().
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_ds = ASLDataset(train_csv, train_transform, segmentation_mode, data_root, cache_dir, cache_resized, image_size)
    val_ds = ASLDataset(val_csv, eval_transform, segmentation_mode, data_root, cache_dir, False, image_size)
    test_ds = ASLDataset(test_csv, eval_transform, segmentation_mode, data_root, cache_dir, False, image_size)

    sampler = None
    shuffle_train = True
    if use_weighted_sampler:
        counts = train_ds.df["class_id"].value_counts().sort_index()
        weights_per_class = 1.0 / counts.values
        sample_weights = torch.tensor(
            [weights_per_class[cid] for cid in train_ds.df["class_id"]], dtype=torch.float
        )
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle_train = False

    # persistent_workers requires num_workers > 0
    pw = persistent_workers and num_workers > 0
    pf = prefetch_factor if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        prefetch_factor=pf,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        prefetch_factor=pf,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        prefetch_factor=pf,
    )

    return train_loader, val_loader, test_loader
