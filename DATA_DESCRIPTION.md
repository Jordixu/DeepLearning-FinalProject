# Dataset & Preprocessing Description

> [!NOTE] Raw images and the shards are not loaded onto the repository due to size. I uploaded only the mini dataset so it can be tested.
---

## Class Space (37 classes)

| Group | Labels |
| --- | --- |
| Digits | `1` `2` `3` `4` `5` `6` `7` `8` `9` `10` |
| Letters | `A` `B` `C` `D` `E` `F` `G` `H` `I` `J` `K` `L` `M` `N` `O` `P` `Q` `R` `S` `T` `U` `V` `W` `X` `Y` `Z` |
| Special | `blank` |

Excluded during merge: `del`, `space`, `0` (see `scripts/merge_datasets.py` -> `_SKIP_CLASSES`).
`nothing` and `Blank` folder names are normalised to `blank`.
Single-letter folder names are normalised to uppercase.

---

## Source Datasets

All five datasets are merged into a single pool before splitting. The 70/15/15 stratified split is applied **across the entire pool**, every source contributes to every split.

| # | Dataset | Local folder | Native size | Classes contributed |
| --- | ------- | ------------ | ----------- | ------------------- |
| 1 | **ASL Alphabet** | `archive (2)/` | 200x200 px | A-Z, blank |
| 2 | **Combined ASL Dataset** | `combine_asl_dataset/` | 400x400 px | A-Z, 1-9 |
| 3 | **ASL-HG Raw Images** | `ASL_Raw_Images/asl_dataset/` | Variable HD | A-Z, 1-9 |
| 4 | **Synthetic ASL Alphabet** | `synthetic_alphabet/` | Variable | A-Z, blank |
| 5 | **Synthetic ASL Numbers** | `synthetic_numbers/` | Variable | 1-10, blank |

---

## 1. ASL Alphabet

**Source:** Kaggle: [`grassknoted/asl-alphabet`](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

**Local folder:** `archive (2)/`

**Contents:** ASL alphabet hand signs, 200x200 px. Two sub-splits are both used:

- `asl_alphabet_train/asl_alphabet_train/`: 29 class subfolders, ~3,000 images per class
- `asl_alphabet_test/asl_alphabet_test/`: flat folder, one image per class named `{Class}_test.jpg`

**Class folders used:** `A-Z` + `nothing` (-> `blank`).
**Excluded:** `del`, `space` - not present across the other datasets.

### Near-duplicate frame thinning

The train split contains sequential video frames that were thinned before merging.

**Keep rule:** retain index `n` if `n == 1` **or** `n % 20 == 0`

Retained indices: `1, 20, 40, 60, ..., 2980, 3000` -> **~151 images per class**
Removed: ~2,849 per class.

**Rationale:** sequential video frames are near-identical; keeping all would inflate dataset size without diversity and risks train/test leakage.

---

## 2. Combined ASL Dataset

**Source:** Kaggle: [`vignonantoine/combinedasldatasets`](https://www.kaggle.com/datasets/vignonantoine/combinedasldatasets)

**Local folder:** `combine_asl_dataset/`

**Contents:** ASL hand signs, 400x400 px JPEG with uniform background. Class subfolders named `0-9` (digits) and `a-z` (lowercase letters).

**Mapping applied during merge:**

- `a-z` -> `A-Z` (uppercased)
- `1-9` -> `1-9` (kept as-is)
- `0` -> **excluded** (class `0` is in `_SKIP_CLASSES`; ASL zero is visually identical to the letter O, also because we had very few images compared to other classes)

This dataset contributes digits `1-9` only (no `10`).

---

## 3. ASL-HG: American Sign Language Hand Gesture Image Dataset

**Source:** Mendeley Data: [`j4y5w2c8w9/1`](https://data.mendeley.com/datasets/j4y5w2c8w9/1)

**Local folder:** `ASL_Raw_Images/asl_dataset/`

**Contents:** 36,000 high-resolution JPG images across 36 classes (A-Z + 0-9). Collected from 10 volunteers in indoor and outdoor environments. Originally 1,000 images per class.

**Variants available in the original download:**

- `ASL_Raw_Images/asl_dataset/` - original unprocessed images; flat class-folder layout (**used**)
- `ASL_Processed_Images/asl_processed/` - MediaPipe-segmented, pre-made 80/20 split (**not used** - we control train/val/test splits at the dataset level)

**Class folders used:** `A-Z` + `1-9` (`0` excluded by `_SKIP_CLASSES`).

### Near-duplicate frame thinning (ASL-HG)

Source images are sequential video frames that were thinned before merging.

**Keep rule:** retain index `n` if `n % 50 == 0` **or** `n % 100 == 1`

Retained indices: `1, 50, 100, 101, 150, 200, 201, ..., 950, 1000` -> **30 images per class**
Removed: 970 per class.

**Rationale:** sequential frames are near-identical; keeping all would inflate dataset size without diversity and risks train/test leakage.

---

## 4. Synthetic ASL Alphabet

**Source:** Kaggle: [`lexset/synthetic-asl-alphabet`](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet)

**Local folder:** `synthetic_alphabet/`

**Contents:** Computer-rendered ASL alphabet hand images. Both sub-splits are merged into the pool:

- `Train_Alphabet/` - class subfolders A-Z + Blank
- `Test_Alphabet/` - class subfolders A-Z + Blank

No frame thinning required - images are individually rendered, not sequential video frames.

---

## 5. Synthetic ASL Numbers

**Source:** Kaggle: [`lexset/synthetic-asl-numbers`](https://www.kaggle.com/datasets/lexset/synthetic-asl-numbers)

**Local folder:** `synthetic_numbers/`

**Contents:** Computer-rendered ASL number hand images. Both sub-splits are merged into the pool:

- `Train_Nums/` - class subfolders 1-10 + Blank
- `Test_Nums/` - class subfolders 1-10 + Blank

No frame thinning required.

---

## Preprocessing Pipeline

Entry point: `src/prepare_data.py`

```bash
python -m src.prepare_data                # full pipeline (CSVs + shards)
python -m src.prepare_data --mini 10      # mini dataset (10 images/class, local dev)
python -m src.prepare_data --skip_shards  # CSVs only, no shards
python -m src.prepare_data --dry_run      # print stats, no writes
```

`--mini N` samples N images per class from `merged_dataset/` into `data/asl_clean_mini/`, generates stratified split CSVs in `data/splits/asl_clean_mini/`, and skips shard creation. Requires `dataset_variant: mini` in `config.yaml`.

### Pipeline steps

```text
0. (mini only) Sample N images per class
   ├── Read merged_dataset/{class}/ folders
   ├── Random sample without replacement (seed from config)
   └── Copy to data/asl_clean_mini/{class}/

1. Build manifest
   ├── Scan source directory (asl_clean_mini/ for mini, merged_dataset/ for shards)
   ├── Collect valid image files (.jpg, .jpeg, .png, .bmp, .webp)
   └── Create DataFrame: (filepath, class_name, class_id)

2. Image validation
   ├── Open each image with PIL
   ├── Verify dimensions >= 32 px on each side  (min_image_size: 32)
   └── Drop corrupted or unreadable files

3. Stratified 70 / 15 / 15 split
   ├── Per-class shuffle (NumPy PCG64 PRNG, seed=33)
   ├── Split: 70% train / 15% val / 15% test - per class
   ├── Concatenate all classes, global shuffle
   └── Guarantee at least 1 image per class per split

4. Write CSV manifests
   ├── data/splits/merged/train.csv   (33,727 rows)
   ├── data/splits/merged/val.csv     (7,231 rows)
   └── data/splits/merged/test.csv    (7,225 rows)

5. Compute normalization statistics (train split only)
   ├── Sample up to 8,000 training images
   ├── Compute per-channel mean and std at 224x224
   └── Save to data/norm_stats.json

6. Create WebDataset shards  (optional; required for cluster training)
   ├── Group images into ~1,000 images per .tar shard  (shard_size: 1000)
   ├── Write to data/asl_shards/{train,val,test}/
   ├── Store as: {key}.jpg + {key}.cls
   └── Write _info.json per split with shard list, totals, class counts
```

### Split statistics

```json
{
  "total_images": 48183,
  "dropped_invalid": 0,
  "split_counts": {
    "train": 33727,
    "val":    7231,
    "test":   7225
  }
}
```

**Per-class training counts:**

| Class | Train | Val | Test |
| --- | --- | --- | --- |
| 1-9 (each) | 717 | 154 | 153 |
| 10 | 700 | 150 | 150 |
| A | 1007 | 216 | 216 |
| B | 1003 | 215 | 215 |
| C | 1017 | 218 | 218 |
| D | 1046 | 224 | 225 |
| E-F | ~930-946 | ~199-203 | ~200-202 |
| G-H | ~979 | ~210 | ~209-210 |
| I | 1024 | 219 | 220 |
| J | 760 | 163 | 163 |
| K | 945 | 202 | 203 |
| L | 1078 | 231 | 231 |
| M-N | ~928-930 | ~199 | ~199-200 |
| O | 965 | 207 | 207 |
| P-Q | ~926-937 | ~198-201 | ~199-201 |
| R | 1082 | 232 | 232 |
| S | 1025 | 220 | 219 |
| T | 923 | 198 | 198 |
| U | 1016 | 218 | 218 |
| V-W | ~976-983 | ~209-211 | ~210 |
| X | 930 | 199 | 199 |
| Y | 1079 | 231 | 231 |
| Z | 738 | 158 | 159 |
| blank | 1422 | 305 | 304 |

`blank` is over-represented relative to digits; handled via inverse-frequency class weights during training.

### Shard metadata format (`_info.json`)

```json
{
  "shards": ["shard_001.tar", "shard_002.tar", "..."],
  "total": 33727,
  "format": "jpeg",
  "class_counts": {"1": 717, "10": 700, "A": 1007, "...", "blank": 1422}
}
```

---

## Image Sizes

| Dataset | Native size | Pipeline input |
| --- | --- | --- |
| ASL Alphabet | 200x200 px | Resized to 224x224 on-the-fly |
| Combined ASL Dataset | 400x400 px | Resized to 224x224 on-the-fly |
| ASL-HG Raw Images | Variable HD | Resized to 224x224 on-the-fly |
| Synthetic ASL Alphabet | Variable | Resized to 224x224 on-the-fly |
| Synthetic ASL Numbers | Variable | Resized to 224x224 on-the-fly |

All images are resized to **224x224** inside the transform pipelines in `src/augmentation.py`.

---

## Augmentation Pipeline (`src/augmentation.py`)

> [!NOTE]
> The rotation has been kept low since letters such as J in the dataset is an "inclined version of another" (in this case, I) since for J we would have to capture the movement. The low rotation is to avoid confusion between these two.

### Explicitely excluded

- **No horizontal flip** - mirrors left/right hand, breaking orientation-sensitive signs (G, H, P, Q).
- **No vertical flip** - gravity-inverts hand posture, never seen in real signing.
- **No 90 degree rotations** - same reason.

### Training transform (stochastic)

```python
T.Compose([
    T.Resize((224, 224)),
    T.RandomAffine(
        degrees=(-15, 15),           # +-15 degree rotation
        translate=(0.10, 0.10),      # +-10% translation in x and y
        shear=(-10, 10, -10, 10),    # +-10 degree shear on both axes
    ),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
    T.Normalize(mean, std),
])
```

### Evaluation / inference transform (deterministic)

```python
T.Compose([
    T.Resize((224, 224)),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean, std),
])
```

### Normalization statistics

| Model type | `mean` | `std` |
| --- | --- | --- |
| Scratch CNNs | `[0.4876, 0.4538, 0.4196]` (dataset-specific, `data/norm_stats.json`) | `[0.2514, 0.2589, 0.2716]` |
| Transfer models | `[0.485, 0.456, 0.406]` (ImageNet) | `[0.229, 0.224, 0.225]` |

Transfer models use ImageNet statistics because their pretrained backbones expect ImageNet-normalised inputs. Scratch CNNs are trained from random initialization, so they use the dataset-specific stats (`DATASET_MEAN` / `DATASET_STD` in `src/augmentation.py`).

---