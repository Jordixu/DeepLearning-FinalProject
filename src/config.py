"""
Configuration loader. Detects whether it runs locally or on the cluster
(Pirineus 3 / SLURM) and resolves all paths accordingly.

Environment detection (see detect_env):
  - ASL_ENV=local|cluster overrides everything.
  - Otherwise 'cluster' if SLURM_JOB_ID / SLURM_NODELIST is set, else 'local'.

Path env-var overrides (useful in SLURM scripts):
  ASL_DATA_ROOT   - override data_root (e.g. copy shards to $SCRATCH first)
  ASL_SHARDS_DIR  - override shards_dir directly

On the cluster, TORCH_HOME is set automatically from paths.cluster.weights_dir
so torchvision finds the pretrained weights without internet access.
"""

import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List

import yaml


def _expand(p: str) -> pathlib.Path:
    return pathlib.Path(os.path.expandvars(str(p)))


def detect_env() -> str:
    env_override = os.environ.get("ASL_ENV", "").strip().lower()
    if env_override in ("local", "cluster"):
        return env_override
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_NODELIST"):
        return "cluster"
    return "local"


@dataclass
class DataloaderConfig:
    batch_size: int = 256
    num_workers: int = 16
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True


@dataclass
class PreprocessingConfig:
    image_size: int = 224
    min_image_size: int = 32
    split_ratios: Dict[str, float] = field(default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15})
    random_seed: int = 33
    cache_resized: bool = False


@dataclass
class TrainingConfig:
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    use_amp: bool = True
    use_weighted_sampler: bool = False
    use_class_weights: bool = True
    scheduler: str = "cosine"
    torch_compile: bool = False


@dataclass
class TransferConfig:
    freeze_epochs: int = 5
    backbones: List[str] = field(default_factory=lambda: [
        "resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0"
    ])


@dataclass
class Config:
    env: str
    data_root: pathlib.Path
    results_root: pathlib.Path
    checkpoints_root: pathlib.Path
    merged_dir: pathlib.Path
    clean_dir: pathlib.Path
    shards_dir: pathlib.Path
    dataset_variant: str
    dataset_source: str
    splits_dir: pathlib.Path
    raw_dataset_paths: Dict[str, pathlib.Path]
    classes: List[str]
    class_to_idx: Dict[str, int]
    class_maps: Dict[str, Dict[str, str]]
    preprocessing: PreprocessingConfig
    dataloader: DataloaderConfig
    training: TrainingConfig
    transfer: TransferConfig
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


def load_config(config_path: str = "configs/config.yaml") -> Config:
    config_path = pathlib.Path(config_path)
    with open(config_path, "rb") as f:
        content_bytes = f.read()

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

    path_block = raw["paths"].get(active_env, raw["paths"]["local"])

    # Set TORCH_HOME for offline weight loading before any model import.
    # Weights are stored flat in weights_dir/*.pth; we create a thin wrapper
    # so torchvision finds them at TORCH_HOME/hub/checkpoints/*.pth.
    if "weights_dir" in path_block:
        weights_dir = _expand(path_block["weights_dir"])
        torch_home = weights_dir.parent / ".torch_home"
        hub_checkpoints = torch_home / "hub" / "checkpoints"
        hub_checkpoints.parent.mkdir(parents=True, exist_ok=True)
        if not hub_checkpoints.exists():
            os.symlink(str(weights_dir), str(hub_checkpoints))
        os.environ.setdefault("TORCH_HOME", str(torch_home))
        print(f"[config] TORCH_HOME -> {torch_home}")

    # Allow per-job data root override (e.g. shards copied to $SCRATCH)
    data_root_override = os.environ.get("ASL_DATA_ROOT")
    if data_root_override:
        data_root = pathlib.Path(data_root_override)
        print(f"[config] data_root overridden by ASL_DATA_ROOT: {data_root}")
    else:
        data_root = _expand(path_block["data_root"])

    results_root = _expand(path_block["results_root"])
    checkpoints_root = _expand(path_block["checkpoints_root"])

    # Allow direct shards_dir override
    shards_dir_override = os.environ.get("ASL_SHARDS_DIR")
    if shards_dir_override:
        shards_dir = pathlib.Path(shards_dir_override)
        print(f"[config] shards_dir overridden by ASL_SHARDS_DIR: {shards_dir}")
    else:
        shards_dir = data_root / raw.get("shards_dir", "asl_shards/")

    dataset_variant = str(raw.get("dataset_variant", "shards")).strip().lower()
    if dataset_variant not in {"mini", "shards"}:
        raise ValueError("dataset_variant must be one of: mini | shards")

    merged_dir = data_root / "merged_dataset"
    clean_dir = data_root / raw.get("mini_dir", "asl_clean_mini/")

    if dataset_variant == "mini":
        dataset_source = "raw"
        splits_dir = data_root / "splits" / "asl_clean_mini"
    else:
        dataset_source = "shards"
        splits_dir = None

    raw_ds = raw.get("raw_datasets", {})
    local_data_root = _expand(raw["paths"]["local"]["data_root"])
    raw_dataset_paths = {name: local_data_root / rel for name, rel in raw_ds.items()}

    classes = raw["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    dl_block = raw["dataloader"].get(active_env, raw["dataloader"]["local"])
    dataloader_cfg = DataloaderConfig(
        batch_size=dl_block.get("batch_size", 256),
        num_workers=dl_block.get("num_workers", 16),
        prefetch_factor=dl_block.get("prefetch_factor", 4),
        persistent_workers=dl_block.get("persistent_workers", True),
        pin_memory=dl_block.get("pin_memory", True),
    )

    pp = raw.get("preprocessing", {})
    preproc_cfg = PreprocessingConfig(
        image_size=pp.get("image_size", 224),
        min_image_size=pp.get("min_image_size", 32),
        split_ratios=pp.get("split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15}),
        random_seed=pp.get("random_seed", 33),
        cache_resized=pp.get("cache_resized", False),
    )

    tr = raw.get("training", {})
    train_cfg = TrainingConfig(
        num_epochs=tr.get("num_epochs", 50),
        learning_rate=tr.get("learning_rate", 1e-3),
        weight_decay=tr.get("weight_decay", 1e-4),
        grad_clip=tr.get("grad_clip", 1.0),
        grad_accum_steps=tr.get("grad_accum_steps", 1),
        use_amp=tr.get("use_amp", True),
        use_weighted_sampler=tr.get("use_weighted_sampler", False),
        use_class_weights=tr.get("use_class_weights", True),
        scheduler=tr.get("scheduler", "cosine"),
        torch_compile=tr.get("torch_compile", False),
    )

    tf = raw.get("transfer", {})
    transfer_cfg = TransferConfig(
        freeze_epochs=tf.get("freeze_epochs", 5),
        backbones=tf.get("backbones", ["resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0"]),
    )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[config] device: {device}")

    return Config(
        env=active_env,
        data_root=data_root,
        results_root=results_root,
        checkpoints_root=checkpoints_root,
        merged_dir=merged_dir,
        clean_dir=clean_dir,
        shards_dir=shards_dir,
        dataset_variant=dataset_variant,
        dataset_source=dataset_source,
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
        debug=raw.get("debug", False),
    )
