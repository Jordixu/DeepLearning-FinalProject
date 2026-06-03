#!/usr/bin/env python3
"""
CLI training entry point for cluster batch jobs.

Usage examples:
  # Scratch model
  python src/train.py --model baseline --exp-name baseline_v1

  # Transfer model (two-phase: freeze then fine-tune)
  python src/train.py --model resnet50 --exp-name resnet50_v1

  # Override config values
  python src/train.py --model efficientnet_b0 --exp-name effb0_v1 --epochs 60 --batch-size 512

  # Skip fine-tuning phase
  python src/train.py --model mobilenet_v3_small --exp-name mobilenet_headonly --no-finetune
"""

import argparse
import json
import pathlib
import sys
import time as _time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix as _confusion_matrix

ROOT = pathlib.Path(__file__).parent        # src/
sys.path.insert(0, str(ROOT))

from config import load_config
from augmentation import build_train_transform, build_eval_transform
from dataset import build_dataloaders
from model import get_model, _BACKBONE_REGISTRY
from training import Trainer, compute_class_weights
from utils import set_seed, evaluate_model, load_model, plot_confusion_matrix


def build_trainer(model, cfg, exp_name, train_loader, val_loader):
    cw = compute_class_weights(cfg.class_weight_source, len(cfg.classes)) if cfg.training.use_class_weights else None
    tr = cfg.training
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=cfg.device,
        exp_name=exp_name,
        results_root=cfg.results_root,
        checkpoints_root=cfg.checkpoints_root,
        num_epochs=tr.num_epochs,
        learning_rate=tr.learning_rate,
        weight_decay=tr.weight_decay,
        grad_clip=tr.grad_clip,
        grad_accum_steps=tr.grad_accum_steps,
        use_amp=tr.use_amp,
        scheduler=tr.scheduler,
        class_weights=cw,
        num_classes=len(cfg.classes),
        debug=cfg.debug,
    )


def _measure_latency(model, device, img_size, warmup=10, runs=100):
    """Return single-image inference latency in milliseconds."""
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = _time.perf_counter()
        for _ in range(runs):
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
    return (_time.perf_counter() - t0) / runs * 1000


def main():
    parser = argparse.ArgumentParser(description="ASL cluster training")
    parser.add_argument("--model", required=True,
                        help="Model name: baseline | deep | deep_batchnorm_regularized | "
                             "resnet18 | resnet50 | mobilenet_v3_small | efficientnet_b0")
    parser.add_argument("--exp-name", required=True, help="Experiment name for outputs")
    parser.add_argument("--config", default=str(ROOT.parent / "configs" / "config.yaml"))
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--freeze-epochs", type=int, default=None,
                        help="Epochs to train with frozen backbone (transfer only)")
    parser.add_argument("--finetune-epochs", type=int, default=None,
                        help="Epochs for fine-tuning phase (transfer only, default: same as --epochs)")
    parser.add_argument("--no-finetune", action="store_true",
                        help="Skip the fine-tuning phase for transfer models")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate for transfer model head (default: 0.3)")
    parser.add_argument("--augment", action="store_true",
                        help="Apply data augmentation during training (scratch models only; "
                             "transfer models always augment)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.dataloader.batch_size = args.batch_size

    freeze_epochs = args.freeze_epochs if args.freeze_epochs is not None else cfg.transfer.freeze_epochs
    finetune_epochs = args.finetune_epochs if args.finetune_epochs is not None else cfg.training.num_epochs

    set_seed(cfg.preprocessing.random_seed, cfg.device)

    is_transfer = args.model in _BACKBONE_REGISTRY

    # Transfer models always augment and use ImageNet normalisation.
    # Scratch models use the dataset-specific norm stats; augmentation only when --augment is set.
    if is_transfer:
        train_tf = build_train_transform(cfg.preprocessing.image_size)
        eval_tf  = build_eval_transform(cfg.preprocessing.image_size)
    else:
        eval_tf  = build_eval_transform(cfg.preprocessing.image_size, use_imagenet_norm=False)
        train_tf = (
            build_train_transform(cfg.preprocessing.image_size, use_imagenet_norm=False)
            if args.augment else eval_tf
        )

    train_loader, val_loader, test_loader = build_dataloaders(
        train_transform=train_tf,
        eval_transform=eval_tf,
        **cfg.dataloader_kwargs(),
    )

    phase1_time_s = 0.0
    phase2_time_s = 0.0
    phase1_epochs = 0
    phase2_epochs = 0
    trainer2 = None

    if is_transfer:
        # Phase 1: train head only (frozen backbone)
        print(f"\n[train] Phase 1: {args.model}, head only for {freeze_epochs} epochs")
        model = get_model(args.model, num_classes=len(cfg.classes),
                          freeze_backbone=True, dropout=args.dropout)

        cfg.training.num_epochs = freeze_epochs
        trainer = build_trainer(model, cfg, f"{args.exp_name}_phase1", train_loader, val_loader)
        trainer.train()
        phase1_time_s = sum(trainer.history["epoch_time_s"])
        phase1_epochs = len(trainer.history["train_loss"])

        if not args.no_finetune:
            # Phase 2: fine-tune the full network at a lower LR
            print(f"\n[train] Phase 2: {args.model}, full fine-tune for {finetune_epochs} epochs")
            model.unfreeze()
            cfg.training.num_epochs = finetune_epochs
            cfg.training.learning_rate = cfg.training.learning_rate * 0.1
            trainer2 = build_trainer(model, cfg, args.exp_name, train_loader, val_loader)
            trainer2.train()
            phase2_time_s = sum(trainer2.history["epoch_time_s"])
            phase2_epochs = len(trainer2.history["train_loss"])
            final_trainer = trainer2
        else:
            final_trainer = trainer
    else:
        model = get_model(args.model, num_classes=len(cfg.classes))
        trainer = build_trainer(model, cfg, args.exp_name, train_loader, val_loader)
        trainer.train()
        phase1_time_s = sum(trainer.history["epoch_time_s"])
        phase1_epochs = len(trainer.history["train_loss"])
        final_trainer = trainer

    # Final evaluation on test set
    best_ckpt = final_trainer.ckpt_dir / "best.pt"
    if best_ckpt.exists():
        load_model(model, str(best_ckpt), cfg.device)
        print(f"\n[train] Loaded best checkpoint from {best_ckpt}")

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_f1, test_bal, y_true, y_pred = evaluate_model(
        model, test_loader, cfg.device, criterion
    )

    print(f"\n{'='*60}")
    print(f"  TEST RESULTS: {args.exp_name}")
    print(f"  loss={test_loss:.4f}  acc={test_acc:.4f}  macro-F1={test_f1:.4f}  bal-acc={test_bal:.4f}")
    print(f"{'='*60}")

    results_dir = final_trainer.results_dir

    # Save raw predictions so the comparison notebook can replot CMs locally
    np.save(str(results_dir / "y_true.npy"), np.array(y_true))
    np.save(str(results_dir / "y_pred.npy"), np.array(y_pred))

    # Measure single-image inference latency
    latency_ms = _measure_latency(model, cfg.device, cfg.preprocessing.image_size)

    num_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    total_time_s = phase1_time_s + phase2_time_s

    test_summary = {
        "exp_name": args.exp_name,
        "model": args.model,
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
        "test_macro_f1": round(test_f1, 6),
        "test_balanced_acc": round(test_bal, 6),
        "inference_latency_ms": round(latency_ms, 3),
        "num_params_M": round(num_params_m, 2),
        "phase1_epochs": phase1_epochs,
        "phase2_epochs": phase2_epochs,
        "phase1_time_s": round(phase1_time_s, 1),
        "phase2_time_s": round(phase2_time_s, 1),
        "total_training_time_s": round(total_time_s, 1),
    }
    with open(results_dir / "test_results.json", "w") as f:
        json.dump(test_summary, f, indent=2)

    cm = _confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm,
                          class_names=cfg.classes,
                          save_path=str(results_dir / "confusion_matrix.png"))
    print(f"[train] Results saved to {results_dir}")


if __name__ == "__main__":
    main()
