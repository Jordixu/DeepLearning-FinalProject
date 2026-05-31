"""
Image augmentation pipelines for ASL classification.

Guardrails (enforced here, not overridable per experiment):
  - NO horizontal flip: mirrors left/right hand, breaking orientation-sensitive
    letters (G, H, P, Q) and the two-handed "0" vs "O" distinction.
  - NO vertical flip: gravity-inverts hand posture, never seen in real signing.
  - NO 90-degree rotations: same reason.

Allowed: small rotation (±30°), small translation/shear, mild colour jitter,
light Gaussian blur, random erasing.

Norm stats
----------
Dataset-specific mean/std are computed by prepare_data.py from the training
set only and stored in data/norm_stats.json.  Pass them via the `mean`/`std`
parameters, or load them from a Config object with build_transforms_from_cfg().
ImageNet stats are used as a fallback when the JSON is not yet available.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torchvision.transforms.v2 as T


# ImageNet statistics — fallback when dataset stats are not yet available.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_train_transform(
    image_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> T.Compose:
    """On-the-fly training transform. Applied per sample in Dataset.__getitem__."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomAffine(
            degrees=(-30, 30),
            translate=(0.15, 0.15),
            shear=(-10, 10, -10, 10),
        ),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
        T.Normalize(mean=mean, std=std),
    ])


def build_eval_transform(
    image_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> T.Compose:
    """Deterministic eval/test transform — no stochastic augmentation."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=mean, std=std),
    ])


def build_transforms_from_cfg(cfg) -> Tuple[T.Compose, T.Compose]:
    """
    Build (train_transform, eval_transform) using norm stats from Config.

    Uses dataset-specific stats when prepare_data.py has been run
    (cfg.norm_stats_from_dataset == True), otherwise falls back to ImageNet.
    """
    pp = cfg.preprocessing
    return (
        build_train_transform(pp.image_size, cfg.norm_mean, cfg.norm_std),
        build_eval_transform(pp.image_size, cfg.norm_mean, cfg.norm_std),
    )
