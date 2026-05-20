"""
Image augmentation pipelines for ASL classification.

Guardrails (enforced here, not overridable per experiment):
  - NO horizontal flip: mirrors left/right hand, breaking orientation-sensitive
    letters (G, H, P, Q) and the two-handed "0" vs "O" distinction.
  - NO vertical flip: gravity-inverts hand posture, never seen in real signing.
  - NO 90-degree rotations: same reason.

Allowed: small rotation (±15°), small translation/shear, mild
colour jitter, light Gaussian blur, random erasing.
"""

import torch
import torchvision.transforms.v2 as T


# ImageNet statistics - used when the backbone was pretrained on ImageNet.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(
    image_size: int = 224,
    use_imagenet_norm: bool = True,
) -> T.Compose:
    """
    On-the-fly training transform.  Applied per sample in Dataset.__getitem__.
    """
    mean = IMAGENET_MEAN if use_imagenet_norm else (0.5, 0.5, 0.5)
    std = IMAGENET_STD if use_imagenet_norm else (0.5, 0.5, 0.5)

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomAffine(
            degrees=15,
            translate=(0.10, 0.10),
            shear=(-10, 10, -10, 10),
        ),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
        T.Normalize(mean=mean, std=std),
    ])


def build_eval_transform(
    image_size: int = 224,
    use_imagenet_norm: bool = True,
) -> T.Compose:
    """Deterministic eval/test transform - no stochastic augmentation."""
    mean = IMAGENET_MEAN if use_imagenet_norm else (0.5, 0.5, 0.5) # change to own stats? 
    std = IMAGENET_STD if use_imagenet_norm else (0.5, 0.5, 0.5)

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=mean, std=std),
    ])
