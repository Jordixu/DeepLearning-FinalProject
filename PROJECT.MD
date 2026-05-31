# ASL Hand Gesture Classification — Project Summary

A deep learning pipeline for American Sign Language (ASL) recognition across **37 classes** (digits 1–10, letters A–Z, and "blank"). The project covers multi-source data unification, near-duplicate frame filtering, preprocessing, augmentation, custom CNN architectures, transfer learning, and real-time camera inference.

---

## Table of Contents

1. [Directory Structure](#1-directory-structure)
2. [Dataset & Sources](#2-dataset--sources)
3. [Data Preparation Pipeline](#3-data-preparation-pipeline)
4. [Normalization](#4-normalization)
5. [Augmentation](#5-augmentation)
6. [Models](#6-models)
7. [Training Setup](#7-training-setup)
8. [Configuration](#8-configuration)
9. [Results & Metrics](#9-results--metrics)
10. [Real-Time Inference](#10-real-time-inference)
11. [Notebooks](#11-notebooks)
12. [Dependencies](#12-dependencies)

---

## 1. Directory Structure

```
DeepLearning-FinalProject/
├── configs/
│   └── config.yaml                        # Master configuration (paths, hyperparams)
├── data/
│   ├── merged_dataset/                    # Unified dataset (all sources merged)
│   ├── asl_shards/                        # WebDataset shards for cloud training
│   ├── splits/merged/                     # train.csv / val.csv / test.csv manifests
│   ├── norm_stats.json                    # Per-channel mean & std
│   ├── preprocessing_report.json          # Split counts and validation report
│   ├── combine_asl_dataset/               # Source 1: Combined ASL Dataset
│   ├── ASL_Raw_Images/                    # Source 2: ASL-HG Hand Gesture Dataset
│   ├── archive (2)/                       # Source 3: ASL Alphabet Dataset
│   ├── synthetic_alphabet/                # Synthetic letter images
│   ├── synthetic_numbers/                 # Synthetic digit images
│   └── description.md                     # Dataset source documentation
├── src/
│   ├── config.py                          # Config loader with environment detection
│   ├── augmentation.py                    # Training & eval transform pipelines
│   ├── dataset.py                         # DataLoaders (WebDataset + CSV-based)
│   ├── model.py                           # All model architectures
│   ├── training.py                        # Trainer (AMP, early stopping, scheduling)
│   ├── prepare_data.py                    # Full data preparation pipeline
│   ├── realtime_inference.py              # Live camera inference
│   └── utils.py                           # Seeding, metrics, visualization, I/O
├── scripts/
│   ├── merge_datasets.py                  # Unify multiple dataset sources
│   ├── filter_dataset.py                  # Remove corrupted/invalid images
│   └── filter_color_frames.py             # Color-based frame filtering
├── experiments/
│   ├── 01_baseline.ipynb
│   ├── 02_deeper.ipynb
│   ├── 03_regularization.ipynb
│   ├── 04_transfer.ipynb
│   └── model_comparison.ipynb
├── results/
│   ├── baseline/                          # BaselineCNN outputs
│   ├── deep_cnn/                          # DeepCNN outputs
│   ├── deep_dropout/                      # DeepCNN + Dropout outputs
│   ├── deep_batchnorm/                    # DeepCNN + BatchNorm outputs
│   ├── deep_bn_dropout/                   # DeepCNN + BatchNorm + Dropout outputs
│   ├── deep_bn_dropout_aug/               # + Augmentation
│   ├── transfer_resnet18/
│   ├── transfer_resnet50/
│   ├── transfer_mobilenet_v3_small/
│   ├── transfer_efficientnet_b0/
│   ├── transfer_*_phase1/                 # Phase 1 (frozen backbone) runs
│   └── comparison/                        # Cross-model comparison visualizations
├── cluster/                               # Cloud (SLURM) training scripts & configs
├── checkpoints/                           # Saved model weights (best.pt, last.pt)
├── pretrained_weights/                    # Downloaded ImageNet backbone weights
└── requirements.txt
```

Each `results/{exp_name}/` folder contains:
- `history.json` / `history.csv` — per-epoch metrics (loss, acc, F1, balanced accuracy)
- `training_history.png` — loss/accuracy/F1 curves
- `confusion_matrix.png` — test-set confusion matrix
- `test_results.json` — final test metrics
- `y_pred.npy` / `y_true.npy` — raw prediction arrays

---

## 2. Dataset & Sources

### Class List (37 classes)

```
Digits:  1  2  3  4  5  6  7  8  9  10
Letters: A  B  C  D  E  F  G  H  I  J  K  L  M
         N  O  P  Q  R  S  T  U  V  W  X  Y  Z
Special: blank
```

Excluded during merge: `0` (redundant with `10`), `del`, `space`. The label `nothing` is normalized to `blank`.

### Source Datasets

All five datasets are merged into a single pool before splitting. The 70/15/15 stratified split is applied **across the entire pool**, so every source contributes images to every split.

| # | Dataset | Source | Classes | Notes |
| --- | ------- | ------ | ------- | ----- |
| 1 | **ASL Alphabet** | Kaggle: [`grassknoted/asl-alphabet`](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | A–Z, blank | 200×200 px; near-duplicate frames thinned (see below) |
| 2 | **ASL-HG Raw Images** | Mendeley: [`j4y5w2c8w9/1`](https://data.mendeley.com/datasets/j4y5w2c8w9/1) | A–Z, 0–9 | Variable HD; near-duplicate frames thinned (see below) |
| 3 | **Combined ASL Dataset** | Kaggle: [`vignonantoine/combinedasldatasets`](https://www.kaggle.com/datasets/vignonantoine/combinedasldatasets) | A–Z, 0–9 | 400×400 px; non-400×400 images removed |
| 4 | **Synthetic ASL Alphabet** | Kaggle: [`lexset/synthetic-asl-alphabet`](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet) | A–Z | Computer-rendered hand images; Train + Test folders both used |
| 5 | **Synthetic ASL Numbers** | Kaggle: [`lexset/synthetic-asl-numbers`](https://www.kaggle.com/datasets/lexset/synthetic-asl-numbers) | 1–10 | Computer-rendered hand images; Train + Test folders both used |

### Near-Duplicate Frame Filtering

Datasets 1 and 2 were collected by recording continuous video, which produces near-identical consecutive frames. Including all frames would inflate apparent dataset size without adding diversity and would cause data leakage (frames from the same shot ending up in both train and test). Frames were thinned before merging:

**ASL-HG (Dataset 2)** — `delete_asl_files.py`

- Source: 1,000 images per class
- Keep rule: retain index `n` if `n % 50 == 0` **or** `n % 100 == 1`
- Retained indices: 1, 50, 100, 101, 150, 200, 201, … → **30 images per class**
- Removed: 970 images per class

**ASL Alphabet (Dataset 1)** — `delete_archive_train.py`

- Source: 3,000 images per class
- Keep rule: retain index `n` if `n == 1` **or** `n % 20 == 0`
- Retained indices: 1, 20, 40, 60, … 3000 → **~151 images per class**
- Removed: ~2,849 images per class

Synthetic datasets (4 & 5) contain no sequential frames and are used as-is.

### Unified Dataset Statistics

```json
{
  "total_images": 48183,
  "dropped_invalid": 0,
  "split_counts": {
    "train": 33727,
    "val": 7231,
    "test": 7225
  }
}
```

**Per-class distribution (training set):**
- Digits (1–10): ~700–717 images each
- Letters (A–Z): ~738–1,082 images each
- `blank`: 1,422 images (over-represented, handled via class weights)

### Image Specifications

| Property | Value |
|----------|-------|
| Native resolutions | 200×200 (Alphabet), 400×400 (Combined), variable HD (ASL-HG), variable (Synthetic) |
| Pipeline size | 224×224 (resized on-the-fly) |
| Format | JPEG decoded to PIL, then converted to torch tensors |
| Minimum valid size | 32×32 px (smaller images dropped during validation step) |
| Color space | RGB |

---

## 3. Data Preparation Pipeline

Entry point: `src/prepare_data.py`

```bash
python -m src.prepare_data                # Full pipeline (CSVs + shards)
python -m src.prepare_data --skip_shards  # CSVs only
python -m src.prepare_data --dry_run      # Print stats without writing
```

### Pipeline Steps

```
1. Build manifest
   ├── Scan merged_dataset/ class directories
   ├── Collect valid image files (.jpg, .jpeg, .png, .bmp, .webp)
   └── Create DataFrame: (filepath, class_name, class_id)

2. Image validation
   ├── Open each image with PIL
   ├── Verify dimensions >= 32 px on each side
   └── Drop corrupted or unreadable files

3. Stratified 70 / 15 / 15 split
   ├── Per-class shuffle (NumPy PCG64 PRNG, seed=33)
   ├── Split: 70% train, 15% val, 15% test — per class
   ├── Concatenate all classes, global shuffle
   └── Guarantee at least 1 image per class per split

4. Write CSV manifests
   ├── data/splits/merged/train.csv  (33,727 rows)
   ├── data/splits/merged/val.csv    (7,231 rows)
   └── data/splits/merged/test.csv   (7,225 rows)

5. Compute normalization statistics (train split only)
   ├── Load up to 8,000 training images
   ├── Compute per-channel mean and std
   └── Save to data/norm_stats.json

6. Create WebDataset shards (optional, for cloud training)
   ├── Group images into ~1,000 images per shard
   ├── Write .tar archives to data/asl_shards/{train,val,test}/
   ├── Store as: {key}.jpg + {key}.cls  (or {key}.npy for fast loading)
   └── Write _info.json per split with shard list, totals, class counts
```

### Shard Metadata Format (`_info.json`)

```json
{
  "shards": ["shard_001.tar", "shard_002.tar", "..."],
  "total": 33727,
  "format": "jpeg",
  "class_counts": {"0": 717, "1": 700, "...", "blank": 1422}
}
```

---

## 4. Normalization

Statistics are computed from the **training split only** (8,000 sampled images):

```json
{
  "mean": [0.4876, 0.4538, 0.4196],
  "std":  [0.2514, 0.2589, 0.2716],
  "image_size": 224
}
```

These dataset-specific values are used instead of the standard ImageNet defaults (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

Applied as the final step of all transforms:
```python
T.Normalize(mean=[0.4876, 0.4538, 0.4196], std=[0.2514, 0.2589, 0.2716])
```

---

## 5. Augmentation

Defined in `src/augmentation.py`. The key constraint is that **horizontal flips, vertical flips, and 90° rotations are explicitly excluded** — ASL signs are orientation-sensitive (e.g., "b" vs "d" or "p" vs "q").

### Training Transform

```python
T.Compose([
    T.Resize((224, 224)),
    T.RandomAffine(
        degrees=(-30, 30),           # ±30° rotation
        translate=(0.15, 0.15),      # ±15% translation in x and y
        shear=(-10, 10, -10, 10),    # ±10° shear on both axes
    ),
    T.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.2,
        hue=0.05,
    ),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
    T.Normalize(mean=[0.4876, 0.4538, 0.4196], std=[0.2514, 0.2589, 0.2716]),
])
```

### Evaluation / Inference Transform (deterministic)

```python
T.Compose([
    T.Resize((224, 224)),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.4876, 0.4538, 0.4196], std=[0.2514, 0.2589, 0.2716]),
])
```

---

## 6. Models

All models defined in `src/model.py`. Output is always 37-dimensional logits.

### 6.1 Scratch Models

#### BaselineCNN (performance floor)

```
Input: (3, 224, 224)
→ Conv2d(3→32, 3×3, padding=1) + ReLU + MaxPool2d(2×2)
→ Conv2d(32→64, 3×3, padding=1) + ReLU + MaxPool2d(2×2)
→ AdaptiveAvgPool2d(4×4)
→ Flatten → Linear(1024→128) → ReLU → Linear(128→37)
Output: (37,)
```

#### DeepCNN (4 convolutional blocks)

```
Input: (3, 224, 224)
→ Conv2d(3→32) + ReLU
→ Conv2d(32→64) + ReLU + MaxPool2d(2×2)
→ Conv2d(64→128) + ReLU
→ Conv2d(128→256) + ReLU + MaxPool2d(2×2)
→ AdaptiveAvgPool2d(4×4)
→ Flatten → Linear(4096→256) → ReLU → Linear(256→37)
Output: (37,)
```

#### Regularized Variants

| Model | Description |
|-------|-------------|
| `DeepCNNRegularized` | DeepCNN + Dropout(0.5) in classifier head |
| `DeepCNNBatchNorm` | DeepCNN + BatchNorm2d after every conv layer |
| `DeepCNNBatchNormRegularized` | BatchNorm2d after every conv + Dropout(0.5) in head |

### 6.2 Transfer Learning Models

`TransferModel` wrapper class supports four ImageNet-pretrained backbones:

| Backbone | Params (M) | Notes |
|----------|-----------|-------|
| ResNet18 | ~11M | Best overall accuracy |
| ResNet50 | ~25M | More parameters, comparable results |
| MobileNetV3-Small | ~2.5M | Most GPU-efficient option |
| EfficientNet-B0 | ~5.3M | Compound scaling |

```python
class TransferModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,       # "resnet18" | "resnet50" | "mobilenet_v3_small" | "efficientnet_b0"
        num_classes: int = 37,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
    )
```

The classifier head replaces each backbone's original output layer with a `Dropout(0.3) → Linear(in_features → 37)` head.

#### Two-Phase Training Strategy

| Phase | Backbone | Head | Duration |
|-------|----------|------|----------|
| Phase 1 | Frozen | Trained | `freeze_epochs=5` |
| Phase 2 | Unfrozen | Trained | Remaining epochs |

Phase 1 quickly adapts the classification head to ASL features; Phase 2 fine-tunes the full network end-to-end.

---

## 7. Training Setup

Defined in `src/training.py` (`Trainer` class).

### Optimizer & Loss

| Component | Value |
|-----------|-------|
| Loss | `CrossEntropyLoss` with optional class weights |
| Class weighting | Inverse frequency: `weight = total / (num_classes × count_per_class)` |
| Optimizer | Adam (`weight_decay=1e-4`) |
| Scheduler | `CosineAnnealingLR` (`T_max=total_steps`, `eta_min=1e-6`) |

### Regularization & Efficiency

| Feature | Setting |
|---------|---------|
| Early stopping | Patience = 7 epochs, monitored on val macro-F1 |
| Gradient clipping | Max norm = 1.0 |
| Gradient accumulation | Steps = 1 (configurable) |
| Automatic Mixed Precision | Enabled (`torch.amp.GradScaler`) |

### Per-Experiment Training Configurations

Scratch and transfer models differ in normalization, augmentation, learning rate, and epoch count:

| Experiment | Normalization | Augmentation | LR | Epochs | Early Stop |
| ---------- | ------------- | ------------ | --- | ------ | ---------- |
| `baseline` | Dataset-specific | None (eval transform on train) | 1e-3 | 50 | — |
| `deep_cnn` | Dataset-specific | None (eval transform on train) | 1e-3 | 50 | — |
| `deep_dropout` | Dataset-specific | None (eval transform on train) | 1e-3 | 50 | — |
| `deep_batchnorm` | Dataset-specific | None (eval transform on train) | 1e-3 | 50 | — |
| `deep_bn_dropout` | Dataset-specific | None (eval transform on train) | 1e-3 | 50 | — |
| `deep_bn_dropout_aug` | Dataset-specific | Full training transform | 1e-3 | 50 | — |
| `transfer_*` Phase 1 | ImageNet | Full training transform | 1e-3 | 5 (frozen backbone) | disabled |
| `transfer_*` Phase 2 | ImageNet | Full training transform | 1e-4 | up to 25, patience=7 | val macro-F1 |

**Normalization choices:**

- Scratch models use dataset-specific stats (`data/norm_stats.json`): `mean=[0.4876, 0.4538, 0.4196]`
- Transfer models use ImageNet stats: `mean=[0.485, 0.456, 0.406]` — required because the pretrained backbones were trained on ImageNet

**Augmentation choices:**

- `01_baseline` and `02_deeper` intentionally use the eval transform during training (no stochastic augmentation) to isolate raw model capacity from regularization effects
- `03_regularization` introduces augmentation only in the `deep_bn_dropout_aug` run
- `04_transfer` applies the full training transform throughout both phases

**Transfer learning LR schedule:**

- Phase 1 (`lr=1e-3`): only the new classifier head is trained; backbone weights are frozen
- Phase 2 (`lr=1e-4`, 10× lower): full network fine-tuned end-to-end; lower LR avoids destroying pretrained features
- Phase 1 early stopping is disabled (`patience = FREEZE_EPOCHS + 1`) so the head always completes its warm-up

### Checkpointing

```
checkpoints/{exp_name}/
├── best.pt   # state_dict at best val macro-F1
└── last.pt   # full checkpoint dict (model + optimizer state) for resumption
```

### Logging

Per-epoch metrics saved to `results/{exp_name}/`:
- `history.json` — dict of lists: `{train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, ...}`
- `history.csv` — same data in CSV format
- `training_history.png` — three-panel plot: loss / accuracy / F1

### Evaluation Metrics

`evaluate_model()` in `src/utils.py` returns:
- Cross-entropy loss
- Overall accuracy
- Macro-averaged F1 score
- Balanced accuracy (accounts for class imbalance)

---

## 8. Configuration

`configs/config.yaml` — master config file, environment-aware.

### Environment Detection (`src/config.py`)

Priority order:
1. `ASL_ENV` environment variable (`local` | `colab` | `kaggle`)
2. `google.colab` importable → `colab`
3. `KAGGLE_KERNEL_RUN_TYPE` set or `/kaggle/working` exists → `kaggle`
4. Default → `local`

### Key Config Values

```yaml
dataset_variant: merged          # "merged" (local CSV) or "shards" (WebDataset)

classes:                         # 37 total
  - "1" ... "10"                 # Digits
  - "A" ... "Z"                  # Letters
  - "blank"

preprocessing:
  image_size: 224
  min_image_size: 32
  shard_size: 1000
  random_seed: 33
  split_ratios: {train: 0.70, val: 0.15, test: 0.15}

dataloader:
  local:   {batch_size: 64,  num_workers: 4}
  colab:   {batch_size: 128, num_workers: 2}

training:
  num_epochs: 30
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  grad_clip: 1.0
  grad_accum_steps: 1
  early_stopping_patience: 7
  use_amp: true
  use_class_weights: true
  scheduler: "cosine"
  torch_compile: false

transfer:
  freeze_epochs: 5
  backbones: [resnet18, resnet50, mobilenet_v3_small, efficientnet_b0]
```

### Environment-Specific Paths

| Environment | `data_root` | `results_root` | `checkpoints_root` |
|-------------|-------------|----------------|-------------------|
| local | `data/` | `results/` | `checkpoints/` |
| colab | `/content/drive/MyDrive/asl/` | `.../asl/results/` | `.../asl/checkpoints/` |
| kaggle | `/kaggle/input/asl-data/` | `/kaggle/working/results/` | `/kaggle/working/checkpoints/` |

---

## 9. Results & Metrics

All experiments trained on the merged dataset (48,183 images, 37 classes).

### Summary Table

Test-set metrics from `results/{exp}/test_results.json`, ordered by Macro-F1:

| Model | Test Acc | Test Macro-F1 | Test Bal Acc | Params (M) | Latency (ms) | Epochs |
| ----- | :------: | :-----------: | :----------: | :--------: | :----------: | :----: |
| **Transfer ResNet50** | **99.61%** | **99.59%** | **99.59%** | 23.58 | 2.49 | 5+25† |
| Transfer EfficientNet-B0 | 99.60% | 99.57% | 99.57% | 4.05 | 3.28 | 5+47 |
| Transfer ResNet18 | 99.49% | 99.46% | 99.45% | 11.20 | 1.03 | 5+38† |
| Transfer MobileNetV3-Small | 98.95% | 98.92% | 98.92% | 1.56 | 2.08 | 5+50 |
| DeepCNN + BatchNorm | 95.71% | 95.76% | 95.75% | 1.45 | 0.36 | 50 |
| DeepCNN + BN + Dropout | 93.33% | 93.49% | 93.43% | 1.45 | 0.35 | 50 |
| DeepCNN + BN + Dropout + Aug | 76.40% | 75.85% | 76.71% | 1.45 | 0.35 | 50 |
| DeepCNN | 74.05% | 73.84% | 74.00% | 1.45 | 0.28 | 50 |
| DeepCNN + Dropout | 69.85% | 69.92% | 69.80% | 1.45 | 0.28 | 50 |
| BaselineCNN | 69.65% | 69.29% | 69.46% | 0.16 | 0.14 | 50 |

† Early stopping triggered. Latency = single-image CPU forward pass (100-run average).

### Key Findings

1. **Transfer learning dominates scratch models** — 99%+ vs ~70% test accuracy.
2. **ResNet50 narrowly beats ResNet18** on the test set (99.59% vs 99.46% F1), but ResNet18 is 23× smaller and 2.4× faster at inference — making it the better practical choice.
3. **MobileNetV3-Small is the most efficient transfer model** — 1.56 M params, 2.08 ms latency, only 0.67% below ResNet50 F1.
4. **BatchNorm is the most impactful regularizer for scratch CNNs** — BatchNorm alone (95.76%) outperforms BatchNorm+Dropout (93.49%), adding Dropout on top slightly hurts.
5. **Augmentation with 50 epochs hurt BN+Dropout** — `deep_bn_dropout_aug` dropped from 93.49% to 75.85% F1; heavy augmentation with this architecture likely requires more epochs to converge.
6. **Dropout alone provides no benefit** without BatchNorm — `deep_dropout` (69.92%) is barely above baseline (69.29%).
7. **Early stopping** prevented unnecessary epochs for ResNet18 (stopped at phase-2 epoch 38) and ResNet50 (stopped at phase-2 epoch 25).

---

## 10. Real-Time Inference

Script: `src/realtime_inference.py`

```bash
python -m src.realtime_inference \
  --checkpoint checkpoints/transfer_mobilenet_v3_small/best.pt \
  --model mobilenet_v3_small \
  --camera 0 \
  --top-k 3 \
  --device cuda
```

### Supported `--model` values

```
Scratch:   baseline | deep | deep_regularized | deep_batchnorm | deep_batchnorm_regularized
Transfer:  resnet18 | resnet50 | mobilenet_v3_small | efficientnet_b0
```

### Features

| Feature | Detail |
|---------|--------|
| Video source | Webcam via OpenCV (`cv2.VideoCapture`) |
| ROI guide box | 55% of frame width, centered |
| Top-K predictions | Configurable (default: 3), shown with softmax probabilities |
| Confidence color | Green ≥ 70% · Yellow ≥ 40% · Red < 40% |
| FPS counter | Displayed bottom-right |
| Screenshot | Press `s` to save current frame with overlay |
| Quit | Press `q` |

### Inference Pipeline

```
Frame capture → Crop ROI → Apply eval transform → Model forward pass
→ Softmax → Top-K selection → Overlay rendering → Display
```

The same eval transform (resize + normalize, no augmentation) used during validation is applied at inference time.

---

## 11. Notebooks

All notebooks are in `experiments/`.

| Notebook | Purpose |
|----------|---------|
| `01_baseline.ipynb` | Trains BaselineCNN; establishes performance floor and validates pipeline |
| `02_deeper.ipynb` | Compares 2-block vs 4-block CNNs; shows capacity effects |
| `03_regularization.ipynb` | Tests BatchNorm, Dropout, augmentation combinations; best scratch model (95.6% val acc) |
| `04_transfer.ipynb` | Trains all four backbones with two-phase strategy; ResNet18 achieves 99.59% F1 |
| `model_comparison.ipynb` | Cross-model analysis: confusion matrices, per-class F1, training curves, timing |

---

## 12. Dependencies

```
# Deep learning
torch>=2.1.0
torchvision>=0.16.0

# Data & ML
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
Pillow>=10.0.0
tqdm>=4.65.0

# Configuration
PyYAML>=6.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# WebDataset (cloud training)
webdataset>=0.3.0

# Real-time inference
opencv-python>=4.8.0
mediapipe>=0.10.0

# Near-duplicate detection
imagehash>=4.3.1

# Notebooks
ipykernel>=6.0.0
ipywidgets>=8.0.0
```
