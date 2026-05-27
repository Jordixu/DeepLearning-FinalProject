# ASL Hand Gesture Classification

Multi-source ASL recognition across 37 classes (digits 0-9, letters A-Z, and "nothing").
Three source datasets are unified into a single pipeline where each dataset maps to one split.

## Repository structure

```text
configs/        config.yaml with all hyperparameters and paths
data/           datasets (see below)
experiments/    one Jupyter notebook per experiment
results/        training logs and plots (gitignored)
checkpoints/    saved model weights (gitignored)
src/
  config.py       config loader, environment detection
  dataset.py      DataLoaders for all three dataset variants
  augmentation.py train and eval transforms
  model.py        BaselineCNN, DeepCNN, transfer learning wrappers
  training.py     Trainer class with AMP, early stopping, logging
  prepare_data.py full data preparation pipeline: unify → validate → split → shard (local)
  utils.py        seeding, visualization, model persistence, evaluation
  eda.ipynb       exploratory data analysis
```

## Datasets

There are three variants. Switch between them by changing `dataset_variant` in `configs/config.yaml`.

| Variant | Directory | Splits | Use case | Git |
| ------- | --------- | ------ | -------- | --- |
| `mini` | `data/asl_clean_mini/` | `data/splits/asl_clean_mini/` | Local dev and testing | Yes |
| `full` | `data/asl_clean/` | `data/splits/asl_clean/` | Full local training | No |
| `shards` | `data/asl_shards/` | `_info.json` per split | Cloud training (recommended) | No |

The mini dataset is a small sample committed to the repo. The full dataset and shards are local-only and must be generated from the raw sources using the pipeline below.

## Local data preparation pipeline (no GPU needed)

Run once on a local machine. The single script handles all steps.

```sh
python -m src.prepare_data
```

Outputs:

- `data/manifest_raw.csv` — unified image manifest
- `data/splits/asl_clean/{train,val,test}.csv` — per-split manifests
- `data/preprocessing_report.json` — validation statistics
- `data/asl_shards/{train,val,test}/*.tar` — WebDataset shards for cloud training

**Split strategy:** each source dataset is assigned to exactly one split via
`preprocessing.dataset_splits` in `configs/config.yaml` — no random splitting.
The default assignment is `combine_asl → train`, `asl_hg_raw → val`, `asl_alphabet → test`.

Optional flags:

```sh
python -m src.prepare_data --skip_shards   # stop after writing split CSVs
python -m src.prepare_data --dry_run       # print stats without writing anything
```

## Cloud training (Colab or Kaggle)

1. Run the local pipeline to produce `data/asl_shards/`.
2. Upload `data/` (or just `asl_shards/`) to your cloud storage.
3. Set `dataset_variant: shards` in `configs/config.yaml`.
4. Update `data_root` in the config for your cloud environment (see comments in config.yaml).
5. Run the experiment notebook.

**Colab:** call `config.mount_drive()` before `load_config()`. Upload `data/` to `MyDrive/asl/` so shards are at `MyDrive/asl/asl_shards/`.

**Kaggle:** add your uploaded dataset; the expected layout is `/kaggle/input/<slug>/asl_shards/{train,val,test}/`. Update `data_root` in config.yaml with your slug.

## Running an experiment

All experiments use the same boilerplate regardless of which dataset variant is active.

```python
import sys, pathlib
ROOT = pathlib.Path('.').resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from config import mount_drive, load_config
from augmentation import build_train_transform, build_eval_transform
from dataset import build_dataloaders
from training import Trainer, compute_class_weights
from utils import set_seed

mount_drive()  # no-op unless on Colab
cfg = load_config(ROOT / 'configs/config.yaml')
set_seed(cfg.preprocessing.random_seed, cfg.device)

train_tf = build_train_transform(cfg.preprocessing.image_size)
eval_tf = build_eval_transform(cfg.preprocessing.image_size)

train_loader, val_loader, test_loader = build_dataloaders(
    train_transform=train_tf,
    eval_transform=eval_tf,
    **cfg.dataloader_kwargs(),
)
```

`cfg.dataloader_kwargs()` handles all dataset variant logic. The same call works for mini, full, and shards - just change `dataset_variant` in config.yaml.

For class-weighted loss:

```python
class_weights = compute_class_weights(cfg.class_weight_source, num_classes)
```

## Configuration reference

All settings are in `configs/config.yaml`. The most commonly changed ones:

- `dataset_variant` - which dataset to use (`mini` / `full` / `shards`)
- `training.num_epochs` - how many epochs to train
- `training.learning_rate` - initial learning rate
- `dataloader.local.batch_size` - batch size for local runs

## Notes

See `note.md` for open issues and things to try in future experiments.
