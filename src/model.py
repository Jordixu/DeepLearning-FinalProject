"""
CNN architectures for ASL classification.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from typing import Literal


# Scratch CNNs -----------------------------------

class BaselineCNN(nn.Module):
    """Shallow 2-block CNN - establishes the performance floor."""

    def __init__(self, in_channels: int = 3, num_classes: int = 37):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class DeepCNN(nn.Module):
    """4-conv-block CNN - doubles depth vs baseline."""

    def __init__(self, in_channels: int = 3, num_classes: int = 37):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class DeepCNNRegularized(nn.Module):
    """DeepCNN + Dropout in the classifier."""

    def __init__(self, in_channels: int = 3, num_classes: int = 37, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class DeepCNNBatchNorm(nn.Module):
    """DeepCNN + BatchNorm2d after every conv."""

    def __init__(self, in_channels: int = 3, num_classes: int = 37):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class DeepCNNBatchNormRegularized(nn.Module):
    """DeepCNN + BatchNorm2d + Dropout - the fully-regularized scratch model."""

    def __init__(self, in_channels: int = 3, num_classes: int = 37, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
    

# Transfer learning backbone

BackboneName = Literal["resnet18", "resnet50", "vgg16", "mobilenet_v3_small", "efficientnet_b0"]

_BACKBONE_REGISTRY = {
    "resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights.DEFAULT),
    "resnet50": (tv_models.resnet50, tv_models.ResNet50_Weights.DEFAULT),
    "vgg16": (tv_models.vgg16, tv_models.VGG16_Weights.DEFAULT),
    "mobilenet_v3_small": (tv_models.mobilenet_v3_small, tv_models.MobileNet_V3_Small_Weights.DEFAULT),
    "efficientnet_b0": (tv_models.efficientnet_b0, tv_models.EfficientNet_B0_Weights.DEFAULT),
}


class TransferModel(nn.Module):
    """
    Pretrained backbone with a replaced classification head.

    Supports a two-phase training strategy:
      1. freeze_backbone=True  → train only the head (feature extractor frozen)
      2. freeze_backbone=False → fine-tune the whole network

    Call model.unfreeze() to switch from phase 1 to phase 2.
    """

    def __init__(
        self,
        backbone_name: BackboneName,
        num_classes: int = 37,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        if backbone_name not in _BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone '{backbone_name}'. Choose from {list(_BACKBONE_REGISTRY)}")

        factory, weights = _BACKBONE_REGISTRY[backbone_name]
        backbone = factory(weights=weights)

        # Replace the classifier head depending on architecture family.
        if backbone_name.startswith("resnet"):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        elif backbone_name == "vgg16":
            in_features = backbone.classifier[6].in_features
            backbone.classifier[6] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        elif backbone_name == "mobilenet_v3_small":
            in_features = backbone.classifier[3].in_features
            backbone.classifier[3] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        elif backbone_name == "efficientnet_b0":
            in_features = backbone.classifier[1].in_features
            backbone.classifier[1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )

        self.backbone = backbone
        self.backbone_name = backbone_name

        if freeze_backbone:
            self.freeze()

    def freeze(self):
        """Freeze all parameters except the classifier head."""
        for name, param in self.backbone.named_parameters():
            head_names = {"fc", "classifier"}
            if not any(h in name for h in head_names):
                param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_model(name: str, num_classes: int = 37, **kwargs) -> nn.Module:
    """Factory: returns a model by name string (for config-driven training)."""
    scratch_models = {
        "baseline": BaselineCNN,
        "deep": DeepCNN,
        "deep_regularized": DeepCNNRegularized,
        "deep_batchnorm": DeepCNNBatchNorm,
        "deep_batchnorm_regularized": DeepCNNBatchNormRegularized,
    }
    if name in scratch_models:
        return scratch_models[name](num_classes=num_classes, **kwargs)
    if name in _BACKBONE_REGISTRY:
        return TransferModel(backbone_name=name, num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model '{name}'. Scratch: {list(scratch_models)}. Transfer: {list(_BACKBONE_REGISTRY)}")
