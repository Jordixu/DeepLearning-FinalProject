"""
Shared utilities: seeding, visualization, model persistence, device helpers,
and lightweight evaluation helpers used by notebooks and the Trainer.
"""

import json
import pathlib
import random
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import os
import subprocess
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

warnings.filterwarnings("ignore")


# Reproductivity

def set_seed(seed: int = 33, device: str = "cpu") -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Device

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# Visualization

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str | None = None,
    figsize: Tuple[int, int] = (14, 12),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=len(class_names) <= 20,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str | None = None,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()

    if "val_macro_f1" in history:
        axes[2].plot(epochs, history["val_macro_f1"], label="Val Macro-F1", color="green")
        axes[2].set(title="Macro-F1 (primary metric)", xlabel="Epoch", ylabel="Macro-F1")
        axes[2].legend()
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()


# Evaluation

def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    criterion: nn.Module,
) -> Tuple[float, float, float, float, List[int], List[int]]:
    """
    Returns (avg_loss, accuracy, macro_f1, balanced_acc, true_labels, pred_labels).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true: List[int] = []
    all_pred: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(all_true, all_pred))

    return avg_loss, accuracy, macro_f1, bal_acc, all_true, all_pred


def show_results(
    true_labels: List[int],
    pred_labels: List[int],
    class_names: List[str],
    save_dir: str | None = None,
) -> None:
    cm = confusion_matrix(true_labels, pred_labels)
    cm_path = str(pathlib.Path(save_dir) / "cm.png") if save_dir else None
    plot_confusion_matrix(cm, class_names, save_path=cm_path)

    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(true_labels, pred_labels)
    print(f"Macro-F1: {macro_f1:.4f}   Balanced Acc: {bal_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=class_names, zero_division=0))


# model persistance

def save_model(model: nn.Module, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str, device: str) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def save_history(history: dict, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def load_history(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# Cloud/bootstrap helper for notebooks (Colab / Kaggle)
def bootstrap_cloud(ensure_packages: list | None = None) -> None:
    """Detect Colab/Kaggle and install minimal packages + adjust sys.path.

    Call this from notebooks before importing project modules when running
    on hosted runtimes.
    """
    if ensure_packages is None:
        ensure_packages = ["pyyaml"]
    if "KAGGLE_URL_BASE" in os.environ or "COLAB_GPU" in os.environ:
        os.environ["ASL_ENV"] = "kaggle"
        for pkg in ensure_packages:
            try:
                __import__(pkg)
            except Exception:
                subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)
        # Kaggle dataset mount path convention
        sys.path.insert(0, "/kaggle/input/asl-repo/src")
