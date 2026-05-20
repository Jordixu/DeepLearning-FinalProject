"""
Configuration loader. Reads configs/config.yaml and resolves
environment-specific paths (local / colab / kaggle).

Environment detection and all path constants live here — no other file
should branch on the execution environment.

Auto-detection order (first match wins):
  1. ASL_ENV env var ("local" | "colab" | "kaggle")
  2. google.colab importable → "colab"
  3. KAGGLE_KERNEL_RUN_TYPE set or /kaggle/working exists → "kaggle"
  4. fallback → "local"

Colab usage:
  Call mount_drive() before load_config() so the Drive paths resolve.
  Data layout on Drive (mirrors the local data/ folder):
    MyDrive/asl/
      asl_clean/{train,val,test}/<class>/   ← images
      splits/{train,val,test}.csv           ← manifests (relative paths)
      results/
      checkpoints/
"""

import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


def detect_env() -> str:
    """Return the active execution environment: 'local', 'colab', or 'kaggle'."""
    env_override = os.environ.get("ASL_ENV", "").strip().lower()
    if env_override in ("local", "colab", "kaggle"):
        return env_override
    # Check Colab first
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or pathlib.Path("/kaggle/working").exists():
        return "kaggle"
    return "local"


def mount_drive(mount_point: str = "/content/drive") -> None:
    """Mount Google Drive on Colab. No-op when not running on Colab."""
    if detect_env() != "colab":
        return
    try:
        from google.colab import drive  # noqa: F401
        drive.mount(mount_point)
    except Exception as exc:
        print(f"[config] Drive mount failed: {exc}")


@dataclass
class DataloaderConfig:
    batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True


@dataclass
class PreprocessingConfig:
    image_size: int = 224
    min_image_size: int = 32
    phash_threshold: int = 5
    phash_hash_size: int = 8
    split_ratios: Dict[str, float] = field(default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15})
    random_seed: int = 33
    cache_resized: bool = False


@dataclass
class TrainingConfig:
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    early_stopping_patience: int = 7
    use_amp: bool = True
    use_weighted_sampler: bool = False
    use_class_weights: bool = True
    scheduler: str = "cosine"
    torch_compile: bool = False


@dataclass
class TransferConfig:
    freeze_epochs: int = 5
    backbones: List[str] = field(default_factory=lambda: ["resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0"])


@dataclass
class Config:
    env: str
    data_root: pathlib.Path
    results_root: pathlib.Path
    checkpoints_root: pathlib.Path
    clean_dir: pathlib.Path           # <data_root>/asl_clean/
    splits_dir: pathlib.Path          # <data_root>/splits/
    # Raw dataset paths - only valid locally (used by data_unification + reorganize)
    raw_dataset_paths: Dict[str, pathlib.Path]
    classes: List[str]
    class_to_idx: Dict[str, int]
    class_maps: Dict[str, Dict[str, str]]
    preprocessing: PreprocessingConfig
    dataloader: DataloaderConfig
    training: TrainingConfig
    transfer: TransferConfig
    device: str = "cpu"


def load_config(config_path: str = "configs/config.yaml") -> Config:
    config_path = pathlib.Path(config_path)
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    active_env = detect_env()
    print(f"[config] environment: {active_env}")

    path_block = raw["paths"][active_env]
    data_root = pathlib.Path(path_block["data_root"])
    results_root = pathlib.Path(path_block["results_root"])
    checkpoints_root = pathlib.Path(path_block["checkpoints_root"])

    clean_dir = data_root / raw.get("clean_dir", "asl_clean/")
    splits_dir = data_root / "splits"

    # Raw dataset paths - local only
    raw_ds = raw.get("raw_datasets", {})
    local_data_root = pathlib.Path(raw["paths"]["local"]["data_root"])
    raw_dataset_paths = {
        name: local_data_root / rel
        for name, rel in raw_ds.items()
    }

    classes = raw["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    dl_block = raw["dataloader"].get(active_env, raw["dataloader"]["local"])
    dataloader_cfg = DataloaderConfig(
        batch_size=dl_block.get("batch_size", 64),
        num_workers=dl_block.get("num_workers", 4),
        prefetch_factor=dl_block.get("prefetch_factor", 2),
        persistent_workers=dl_block.get("persistent_workers", True),
        pin_memory=dl_block.get("pin_memory", True),
    )

    pp = raw.get("preprocessing", {})
    preproc_cfg = PreprocessingConfig(
        image_size=pp.get("image_size", 224),
        min_image_size=pp.get("min_image_size", 32),
        phash_threshold=pp.get("phash_threshold", 5),
        phash_hash_size=pp.get("phash_hash_size", 8),
        split_ratios=pp.get("split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15}),
        random_seed=pp.get("random_seed", 33),
        cache_resized=pp.get("cache_resized", False),
    )

    tr = raw.get("training", {})
    train_cfg = TrainingConfig(
        num_epochs=tr.get("num_epochs", 30),
        learning_rate=tr.get("learning_rate", 1e-3),
        weight_decay=tr.get("weight_decay", 1e-4),
        grad_clip=tr.get("grad_clip", 1.0),
        grad_accum_steps=tr.get("grad_accum_steps", 1),
        early_stopping_patience=tr.get("early_stopping_patience", 7),
        use_amp=tr.get("use_amp", True),
        use_weighted_sampler=tr.get("use_weighted_sampler", False),
        use_class_weights=tr.get("use_class_weights", True),
        scheduler=tr.get("scheduler", "cosine"),
        torch_compile=tr.get("torch_compile", False),
    )

    tf = raw.get("transfer", {})
    transfer_cfg = TransferConfig(
        freeze_epochs=tf.get("freeze_epochs", 5),
        backbones=tf.get("backbones", ["resnet18"]),
    )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return Config(
        env=active_env,
        data_root=data_root,
        results_root=results_root,
        checkpoints_root=checkpoints_root,
        clean_dir=clean_dir,
        splits_dir=splits_dir,
        raw_dataset_paths=raw_dataset_paths,
        classes=classes,
        class_to_idx=class_to_idx,
        class_maps=raw.get("class_maps", {}),
        preprocessing=preproc_cfg,
        dataloader=dataloader_cfg,
        training=train_cfg,
        transfer=transfer_cfg,
        device=device,
    )
