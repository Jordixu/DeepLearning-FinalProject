"""
Configuration loader for the ASL classification project.

Reads configs/config.yaml and resolves environment-specific paths.

Environment detection (first match wins):
  1. ASL_ENV env var  ("local" | "colab" | "kaggle")
  2. google.colab importable  → "colab"
  3. KAGGLE_KERNEL_RUN_TYPE set or /kaggle/working exists  → "kaggle"
  4. fallback  → "local"

Dataset variants:
  merged: data/merged_dataset/ + splits/merged/  (all 5 sources, local)
  shards: data/asl_shards/                       (WebDataset tars, cloud)

Splitting strategy:
  Random stratified 70/15/15 split performed once by prepare_data.py.
  No per-source dataset bias.

Norm stats:
  Per-channel mean/std computed by prepare_data.py from the train split only.
  Loaded from data/norm_stats.json if present; otherwise ImageNet fallback.
"""

import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml


def detect_env() -> str:
    env_override = os.environ.get("ASL_ENV", "").strip().lower()
    if env_override in ("local", "colab", "kaggle"):
        return env_override
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or pathlib.Path("/kaggle/working").exists():
        return "kaggle"
    return "local"


def mount_drive(mount_point: str = "/content/drive") -> None:
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
    shard_size: int = 1000
    random_seed: int = 33
    split_ratios: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15}
    )


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
    backbones: List[str] = field(
        default_factory=lambda: ["resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0"]
    )


# ImageNet fallback constants (used when norm_stats.json is absent)
_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class Config:
    env: str
    data_root: pathlib.Path
    results_root: pathlib.Path
    checkpoints_root: pathlib.Path
    merged_dir: pathlib.Path        # data/merged_dataset/
    shards_dir: pathlib.Path        # data/asl_shards/
    splits_dir: pathlib.Path        # data/splits/merged/
    dataset_variant: str            # "merged" | "shards"
    dataset_source: str             # "raw" | "shards"
    classes: List[str]
    class_to_idx: Dict[str, int]
    preprocessing: PreprocessingConfig
    dataloader: DataloaderConfig
    training: TrainingConfig
    transfer: TransferConfig
    norm_mean: Tuple[float, float, float] = _IMAGENET_MEAN
    norm_std: Tuple[float, float, float] = _IMAGENET_STD
    norm_stats_from_dataset: bool = False  # True when loaded from norm_stats.json
    device: str = "cpu"
    debug: bool = False

    @property
    def class_weight_source(self) -> pathlib.Path:
        if self.dataset_source == "shards":
            return self.shards_dir / "train"
        return self.splits_dir / "train.csv"

    def dataloader_kwargs(self) -> dict:
        dl = self.dataloader
        return dict(
            shards_dir=self.shards_dir if self.dataset_source == "shards" else None,
            batch_size=dl.batch_size,
            num_workers=dl.num_workers,
            pin_memory=dl.pin_memory,
            persistent_workers=dl.persistent_workers,
            prefetch_factor=dl.prefetch_factor,
            dataset_source=self.dataset_source,
            data_root=self.data_root,
            splits_dir=self.splits_dir,
            debug=self.debug,
        )


def _load_norm_stats(
    data_root: pathlib.Path,
) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    """Load norm_stats.json if present; return (mean, std) or None."""
    path = data_root / "norm_stats.json"
    if not path.exists():
        return None
    try:
        stats = json.loads(path.read_text())
        mean = tuple(float(v) for v in stats["mean"])
        std = tuple(float(v) for v in stats["std"])
        return mean, std  # type: ignore[return-value]
    except Exception as exc:
        print(f"[config] Could not load norm_stats.json: {exc}")
        return None


def load_config(config_path: str = "configs/config.yaml") -> Config:
    config_path = pathlib.Path(config_path)
    content_bytes = config_path.read_bytes()

    raw = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            raw = yaml.safe_load(content_bytes.decode(enc))
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raw = yaml.safe_load(content_bytes.decode("utf-8", errors="replace"))

    active_env = detect_env()
    print(f"[config] environment: {active_env}")

    path_block = raw["paths"][active_env]
    data_root = pathlib.Path(path_block["data_root"])
    results_root = pathlib.Path(path_block["results_root"])
    checkpoints_root = pathlib.Path(path_block["checkpoints_root"])

    merged_dir = data_root / raw.get("merged_dir", "merged_dataset/")
    shards_dir = data_root / raw.get("shards_dir", "asl_shards/")

    dataset_variant = str(raw.get("dataset_variant", "merged")).strip().lower()
    if dataset_variant not in {"merged", "shards"}:
        raise ValueError("dataset_variant must be 'merged' or 'shards'")

    dataset_source = "shards" if dataset_variant == "shards" else "raw"
    splits_dir = data_root / "splits" / "merged"

    classes = raw["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    pp = raw.get("preprocessing", {})
    ratios_raw = pp.get("split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15})
    preproc_cfg = PreprocessingConfig(
        image_size=pp.get("image_size", 224),
        min_image_size=pp.get("min_image_size", 32),
        shard_size=pp.get("shard_size", 1000),
        random_seed=pp.get("random_seed", 33),
        split_ratios={k: float(v) for k, v in ratios_raw.items()},
    )

    dl_block = raw["dataloader"].get(active_env, raw["dataloader"]["local"])
    dataloader_cfg = DataloaderConfig(
        batch_size=dl_block.get("batch_size", 64),
        num_workers=dl_block.get("num_workers", 4),
        prefetch_factor=dl_block.get("prefetch_factor", 2),
        persistent_workers=dl_block.get("persistent_workers", True),
        pin_memory=dl_block.get("pin_memory", True),
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

    # Norm stats: prefer dataset-specific stats, fall back to ImageNet
    norm_loaded = _load_norm_stats(data_root)
    if norm_loaded is not None:
        norm_mean, norm_std = norm_loaded
        norm_from_dataset = True
        print(f"[config] Loaded norm stats from norm_stats.json  "
              f"mean={[f'{v:.4f}' for v in norm_mean]}  "
              f"std={[f'{v:.4f}' for v in norm_std]}")
    else:
        norm_mean, norm_std = _IMAGENET_MEAN, _IMAGENET_STD
        norm_from_dataset = False
        print("[config] norm_stats.json not found — using ImageNet fallback. "
              "Run prepare_data.py to generate dataset-specific stats.")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return Config(
        env=active_env,
        data_root=data_root,
        results_root=results_root,
        checkpoints_root=checkpoints_root,
        merged_dir=merged_dir,
        shards_dir=shards_dir,
        splits_dir=splits_dir,
        dataset_variant=dataset_variant,
        dataset_source=dataset_source,
        classes=classes,
        class_to_idx=class_to_idx,
        preprocessing=preproc_cfg,
        dataloader=dataloader_cfg,
        training=train_cfg,
        transfer=transfer_cfg,
        norm_mean=norm_mean,
        norm_std=norm_std,
        norm_stats_from_dataset=norm_from_dataset,
        device=device,
        debug=raw.get("debug", False),
    )
