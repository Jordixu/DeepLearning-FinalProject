"""
Trainer: wraps the training loop with:
  - CosineAnnealingLR scheduler
  - Early stopping on val macro-F1
  - Mixed precision (AMP) via torch.amp
  - Gradient clipping
  - Gradient accumulation
  - Class-weighted CrossEntropyLoss
  - Best-checkpoint saving
  - Per-epoch logging to JSON + GPU memory diagnostics
"""

import pathlib
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import evaluate_model, plot_training_history, save_history, save_model


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: str,
        exp_name: str,
        results_root: str | pathlib.Path = "results",
        checkpoints_root: str | pathlib.Path = "checkpoints",
        # Training hyperparams
        num_epochs: int = 30,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 1,
        early_stopping_patience: int = 7,
        use_amp: bool = True,
        scheduler: str = "cosine",
        # Loss
        class_weights: Optional[torch.Tensor] = None,
        # Misc
        num_classes: int = 37,
        debug: bool = False,
    ):
        self.debug = debug
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.exp_name = exp_name
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.num_classes = num_classes

        self.results_dir = pathlib.Path(results_root) / exp_name
        self.ckpt_dir = pathlib.Path(checkpoints_root) / exp_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Loss
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        use_amp = use_amp and device == "cuda"
        self.scaler = GradScaler(enabled=use_amp)
        self.use_amp = use_amp

        total_steps = num_epochs * len(train_loader) // self.grad_accum_steps
        if scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-6)
        elif scheduler == "step":
            self.scheduler = StepLR(self.optimizer, step_size=max(1, num_epochs // 3), gamma=0.1)
        else:
            self.scheduler = None

        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [], "train_macro_f1": [],
            "val_loss": [], "val_acc": [], "val_macro_f1": [], "val_balanced_acc": [],
            "epoch_time_s": [], "peak_gpu_mb": [],
        }
        self.best_val_f1 = -float("inf")

    def train(self) -> Dict[str, List[float]]:
        print(f"\n{'='*60}")
        print(f"  Experiment: {self.exp_name}")
        print(f"  Device: {self.device}  |  AMP: {self.use_amp}  |  Epochs: {self.num_epochs}")
        print(f"{'='*60}")

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()
            train_loss, train_acc, train_f1 = self._train_epoch(epoch)
            val_loss, val_acc, val_f1, val_bal = self._val_epoch()
            elapsed = time.time() - t0

            peak_mb = 0.0
            if self.device == "cuda":
                peak_mb = torch.cuda.max_memory_allocated() / 1e6
                torch.cuda.reset_peak_memory_stats()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["train_macro_f1"].append(train_f1)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_macro_f1"].append(val_f1)
            self.history["val_balanced_acc"].append(val_bal)
            self.history["epoch_time_s"].append(round(elapsed, 2))
            self.history["peak_gpu_mb"].append(round(peak_mb, 1))

            print(
                f"Epoch {epoch:03d}/{self.num_epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                f"val_F1={val_f1:.4f}  [{elapsed:.1f}s]"
            )

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                best_path = str(self.ckpt_dir / "best.pt")
                save_model(self.model, best_path)
                print(f"  New best val macro-F1={val_f1:.4f} - checkpoint saved")

            save_history(self.history, str(self.results_dir / "history.json"))

            if self.early_stopping.step(val_f1):
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

        plot_training_history(self.history, save_path=str(self.results_dir / "training_history.png"))
        return self.history

    def _train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_pred: List[int] = []
        all_true: List[int] = []

        self.optimizer.zero_grad()

        import platform, sys
        if self.debug:
            print(f"  [DEBUG] _train_epoch start | OS={platform.system()} | Python={sys.version.split()[0]}", flush=True)
            print(f"  [DEBUG] DataLoader: num_workers={self.train_loader.num_workers}  batch_size={self.train_loader.batch_size}", flush=True)
            print(f"  [DEBUG] Creating iterator (if stuck here -> multiprocessing deadlock)...", flush=True)

        t_epoch_start = time.perf_counter()
        pbar = tqdm(self.train_loader, desc=f"  Train E{epoch}", leave=False)

        if self.debug:
            print(f"  [DEBUG] Iterator created in {time.perf_counter()-t_epoch_start:.2f}s, entering batch loop...", flush=True)

        for step, (inputs, labels) in enumerate(pbar):
            if self.debug and step == 0:
                print(f"  [DEBUG] First batch in {time.perf_counter()-t_epoch_start:.2f}s | shape={tuple(inputs.shape)}", flush=True)
            if self.debug and step == 1:
                print(f"  [DEBUG] Second batch in {time.perf_counter()-t_epoch_start:.2f}s", flush=True)
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels) / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler is not None and isinstance(self.scheduler, CosineAnnealingLR):
                    self.scheduler.step()

            total_loss += loss.item() * self.grad_accum_steps * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_pred.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

        if self.scheduler is not None and isinstance(self.scheduler, StepLR):
            self.scheduler.step()

        avg_loss = total_loss / total
        acc = correct / total
        macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
        return avg_loss, acc, macro_f1

    def _val_epoch(self):
        if self.debug:
            print(f"  [DEBUG] _val_epoch start | num_workers={self.val_loader.num_workers}", flush=True)
        val_loss, val_acc, val_f1, val_bal, _, _ = evaluate_model(
            self.model, self.val_loader, self.device, self.criterion
        )
        if self.debug:
            print(f"  [DEBUG] _val_epoch done", flush=True)
        return val_loss, val_acc, val_f1, val_bal


# Compute class weights for imbalanced loss

def compute_class_weights(source: str | pathlib.Path, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a shard directory or a CSV.

    Args:
        source: Path to a split shard directory (contains _info.json) OR a
                train.csv file. Shard directories are preferred.
        num_classes: Total number of classes.
    """
    import json as _json
    source = pathlib.Path(source)
    counts = np.zeros(num_classes, dtype=np.float32)

    if source.is_dir():
        info_path = source / "_info.json"
        with open(info_path) as f:
            info = _json.load(f)
        for cid_str, cnt in info["class_counts"].items():
            counts[int(cid_str)] = cnt
    else:
        df = pd.read_csv(source)
        for cid, cnt in df["class_id"].value_counts().items():
            counts[int(cid)] = cnt

    counts = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
