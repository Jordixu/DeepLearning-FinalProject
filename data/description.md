# Dataset Descriptions

> [!NOTE] Information extracted from the original dataset sources.

---

## 1. Combined ASL Dataset

**Source:** [Kaggle - vignonantoine/combinedasldatasets](https://www.kaggle.com/datasets/vignonantoine/combinedasldatasets?select=combine_asl_dataset)

**Local folder:** `combine_asl_dataset/`

**Contents:** ASL hand signs for numbers (0–9) and letters (A–Z). Images are 400×400 px JPEG with a uniform background. Data includes augmented and cropped frames from source videos.

**Class folders:** `0–9` (digit strings), `a–z` (lowercase letters) → mapped to canonical uppercase in the pipeline.

### Images excluded

Four cleanup scripts were run to remove non-400×400 images before the dataset was committed to this repo:

| Script | Method | Effect |
|---|---|---|
| `run_cleanup_400.py` | PIL `Image.open` | Deletes any file not exactly 400×400 px |
| `run_cleanup_cv2.py` | OpenCV `imread` | Same check via OpenCV (alternative runner) |
| `run_cleanup_headers.py` | Binary header parsing (no full decode) | Faster; supports JPEG, PNG, GIF, BMP, WEBP |
| `run_cleanup_timeout.py` | Header parsing + `ProcessPoolExecutor` | Handles stalled files with 2 s timeout |

All four scripts apply the same rule: **keep only if `(width, height) == (400, 400)`**. Corrupted or unreadable files are also deleted.

**Result:** The dataset contains only valid 400×400 JPEG images. Any image that was not exactly this size (e.g. uncroppable frames, thumbnails, non-image files) has been permanently removed.

---

## 2. ASL-HG: American Sign Language Hand Gesture Image Dataset

**Source:** [Mendeley Data - j4y5w2c8w9/1](https://data.mendeley.com/datasets/j4y5w2c8w9/1)

**Local folder:** `ASL-HG American Sign Language Hand Gesture Image D/ASL-HG American Sign Language Hand Gesture Image D/ASL_HG_36000/`

**Contents:** 36,000 high-resolution JPG images across 36 ASL classes (A–Z + 0–9). Collected from 10 volunteers in indoor and outdoor environments. Originally 1,000 images per class.

**Variants:**
- `ASL_Raw_Images/asl_dataset/` - original unprocessed images; flat class-folder layout
- `ASL_Processed_Images/asl_processed/` - MediaPipe-segmented, pre-made 80/20 train/test split (**not used in this project**)

**Only the raw variant is used.** The processed variant is excluded because each source dataset is assigned to exactly one split via `prepare_data.py`, so we control train/val/test boundaries at the dataset level rather than relying on pre-made splits.

### Images excluded - `delete_asl_files.py`

The script `delete_asl_files.py` was run on `ASL_Raw_Images/` to thin out near-sequential frames.

**Keep rule:** retain file if its trailing index `n` satisfies  
`n % 50 == 0` **or** `n % 100 == 1`

From a source range of 1–1,000, this keeps indices:  
`1, 50, 100, 101, 150, 200, 201, 250, 300, 301, 350, 400, 401, 450, 500, 501, 550, 600, 601, 650, 700, 701, 750, 800, 801, 850, 900, 901, 950, 1000`  
→ **30 images retained per class** (970 deleted per class).

**Rationale:** Sequential video frames are near-identical; keeping all of them would inflate training set size without adding diversity and would cause leakage if frames from the same shot ended up in both train and test.

---

## 3. ASL Alphabet

**Source:** [Kaggle - grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data?select=asl_alphabet_train)

**Local folder:** `archive (2)/asl_alphabet_train/asl_alphabet_train/`

**Contents:** 87,000 images of ASL alphabet hand signs, 200×200 px, 29 class folders.

**Class folders used:**
- `A–Z` (26 uppercase letters) → mapped to canonical labels
- `nothing` (lowercase) → mapped to canonical class `"nothing"` (class index 36)

**Class folders excluded:**
- `del` - ASL delete gesture; not present in the other two datasets, excluded to keep the label space consistent
- `space` - ASL space gesture; same reason

### Images excluded - `delete_archive_train.py`

The script `delete_archive_train.py` was run to thin out near-sequential frames from this dataset.

**Keep rule:** retain file if its trailing index `n` satisfies  
`n == 1` **or** `n % 20 == 0`

From a source range of 1–3,000, this keeps indices:  
`1, 20, 40, 60, …, 2980, 3000`  
→ **~151 images retained per class** (≈2,849 deleted per class).

**Rationale:** Same as above - removes near-duplicate sequential frames while retaining a well-spaced sample of hand orientations and lighting conditions.

---

## Image sizes across datasets

| Dataset | Native size | After pipeline |
|---|---|---|
| combine_asl | 400×400 px | Resized to 224×224 on-the-fly |
| ASL-HG raw | Variable (smartphone HD) | Resized to 224×224 on-the-fly |
| ASL Alphabet | 200×200 px | Resized to 224×224 on-the-fly |

All images are resized to **224×224** inside `build_train_transform()` / `build_eval_transform()` in `src/augmentation.py`. No images are pre-resized to disk except when `cache_resized: true` is set in `config.yaml` (for slow Colab Drive I/O).

---

## Discarded data (manual review)

> Add notes here for any classes or individual images removed during manual inspection.
