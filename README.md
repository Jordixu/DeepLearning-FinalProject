# ASL Hand Gesture Classification

Multi-source ASL recognition across 37 classes (digits 1-10, letters A-Z, and "blank").
Three source datasets are unified into a single pipeline where each dataset maps to one split.

> A more detailed description of the project / repo can be found at [PROJECT.MD](PROJECT.MD).

## Repository structure

```text
configs/        config.yaml with all hyperparameters and paths
data/           datasets (see below)
notebooks/    model_comparison.ipynb - results analysis
results/        training logs and plots (gitignored)
checkpoints/    saved model weights (gitignored)
slurm/          SLURM job scripts for Pirineus 3 (HPC)
scripts/        utility scripts (data preparation, weight download, dataset filtering)
src/
  config.py       config loader, expands $HOME paths on cluster
  dataset.py      DataLoaders for all three dataset variants
  augmentation.py train and eval transforms
  model.py        BaselineCNN, DeepCNN, EfficientNetV2 wrappers
  training.py     Trainer class with AMP, logging
  prepare_data.py full local preparation pipeline: merge → validate → split → shard
  utils.py        seeding, visualization, model persistence, evaluation
  eda.ipynb       exploratory data analysis
  train.py        CLI entry point for SLURM batch jobs
```

## Datasets

| Variant | Directory | Splits | Use case |
| ------- | --------- | ------ | -------- |
| `mini` | `data/asl_clean_mini/` | `data/splits/asl_clean_mini/` | Local dev and testing |
| `shards` | `data/asl_shards/` | `_info.json` per split | Cluster training |

Set `dataset_variant` in `configs/config.yaml` to switch between variants. See [DATA_DESCRIPTION.md](DATA_DESCRIPTION.md) for full dataset documentation, preprocessing pipeline details, and per-class statistics.

## Local data preparation pipeline

Run once on a local machine (to check everything works) before uploading to the cluster. Requires `data/merged_dataset/` to exist (produced by `scripts/merge_datasets.py`).

**Mini dataset** (local dev, fast, small):

```sh
python -m src.prepare_data --mini 10
```

Samples 10 images per class from `merged_dataset/` into `data/asl_clean_mini/`, then generates stratified split CSVs in `data/splits/asl_clean_mini/`. Change `10` to any number per class.

**Full pipeline** (produces shards for cluster training):

```sh
python -m src.prepare_data
```

Outputs:

- `data/splits/merged/{train,val,test}.csv`: stratified split manifests
- `data/norm_stats.json`: per-channel mean & std (train set only)
- `data/preprocessing_report.json`: validation statistics
- `data/asl_shards/{train,val,test}/*.tar`: WebDataset shards

Optional flags:

```sh
python -m src.prepare_data --skip_shards      # stop after CSVs + norm stats
python -m src.prepare_data --skip_validation  # skip corrupt/size check
python -m src.prepare_data --dry_run          # print stats without writing anything
```

## Cluster setup (Pirineus 3 - CSUC HPC)

> [!WARNING]
> This setup is specific for this cluster, other clusters may need a different configuration. This is also true if using Kaggle or Colab. Take into account also that the batch files are also specific to this cluster, specially regarding resource allocation (header).

Large inputs live under `$DATA` (big-file storage); outputs and logs live under `$HOME`:

```text
$DATA/
  asl/                 <- this project (upload root directory here)
  asl_shards/          <- WebDataset tar shards (train/ val/ test/)
  pretrained_weights/  <- torchvision weight cache (flat *.pth files)
$HOME/
  results/             <- training outputs (created automatically)
  checkpoints/         <- saved checkpoints (created automatically)
  LOGS/                <- SLURM stdout/stderr logs
```

### One-time setup

**1. Download pretrained weights locally** (needs internet access):

> [!TIP]
> This specific cluster does not have internet access, therefore weights have to be downloaded beforehand. If using Kaggle or Colab, this step can be skipped.

```bash
python scripts/download_weights.py --out pretrained_weights/
```

**2. Upload to the cluster** using the file browser. Three things go under `$DATA/`:

| Local path | Destination on cluster |
| ---------- | ---------------------- |
| Project root (all files) | `$DATA/asl/` |
| `pretrained_weights/` | `$DATA/pretrained_weights/` |
| `data/asl_shards/` | `$DATA/asl_shards/` |

**3. Install extra packages** on the cluster:

```bash
module load conda && conda activate (env-name)
pip install -r $DATA/asl/requirements.txt
```

**4. Create the logs directory:**

```bash
mkdir -p $HOME/LOGS
```

## Running experiments

All commands below are run **from `$DATA/asl/` on the cluster**.

```bash
# Submit all jobs at once
bash slurm/run_all.sh

# Or individually:
sbatch slurm/train_baseline.sh
sbatch slurm/train_deeper.sh
sbatch slurm/train_regularization.sh
sbatch slurm/train_transfer_array.sh   # all backbones in parallel

# Monitor
squeue -u $USER
tail -f $HOME/LOGS/<JOBID>.out
```

### train.py CLI reference

`src/train.py` is the entry point each SLURM job calls. It builds the model and
data loaders, runs the training loop (two-phase for transfer models), then
evaluates on the test split and writes results/checkpoints.

```text
python src/train.py --model MODEL --exp-name NAME [options]

  --model        baseline | deep | deep_regularized | deep_batchnorm |
                 resnet18 | resnet50 | mobilenet_v3_small | efficientnet_b0
  --exp-name     experiment name (results/ and checkpoints/ subdirs)
  --config       path to config YAML (default: configs/config.yaml)
  --epochs       override num_epochs
  --lr           override learning_rate
  --batch-size   override batch_size
  --freeze-epochs   frozen backbone epochs (transfer only, default: 5)
  --finetune-epochs fine-tune phase epochs (default: same as --epochs)
  --no-finetune  skip fine-tuning phase
  --dropout      classification head dropout (default: 0.3)
```

## Real-time inference

> [!NOTE]
> I deliberately tracked the checkpoint of the MobileNet so you can try it out.

Run live ASL recognition from a webcam using a trained checkpoint:

```bash
python -m src.realtime_inference \
  --checkpoint checkpoints/transfer_mobilenet_v3_small/best.pt \
  --model mobilenet_v3_small \
  --camera 0 \ # Sometimes camera 0 does not work, if that's the case try 1
  --top-k 3 \
  --device cpu
```

Key options:

```text
--checkpoint   path to best.pt checkpoint (relative to current path)
--model        baseline | deep | deep_regularized | deep_batchnorm |
               resnet18 | resnet50 | mobilenet_v3_small | efficientnet_b0
--camera       camera index (default: 0)
--top-k        number of top predictions to display (default: 3)
--device       cpu | cuda (default: cpu)
```

Controls: press `s` to save a screenshot, `q` to quit.

## Retrieving results

```bash
rsync -avP -e 'ssh -p 2122' USER@pirineus3.csuc.cat:~/results/     results/
rsync -avP -e 'ssh -p 2122' USER@pirineus3.csuc.cat:~/checkpoints/ checkpoints/
```

Then open `experiments/model_comparison.ipynb` to analyse results.

## Kaggle / Colab

The above workflow will not work in these environments, the initialization scripts are specially design for clusters that uses SLURM.

If the project were runned on Colab / Kaggle, notebooks emulating the behaviour of the batch scripts (`*.sh`) shall be created. Nonetheless, since the project was created from a modular approach, this can be easily accomplished by importing the different modules into the notebook.

If that were the case, the following requirements should be met:

1. **Dataset**: upload the WebDataset shards (`asl_shards`) or the small `asl_clean_mini` sample, and set `dataset_variant` in `configs/config.yaml` accordingly (`shards` or `mini`).
2. **Pretrained weights**: Colab and Kaggle have internet access, so `scripts/download_weights.py` is not needed, torchvision downloads the backbone weights automatically on first use.
3. **Paths**: point `paths.local.data_root` at wherever the data is mounted (e.g. `/kaggle/input/...`), or override it at runtime with the `ASL_DATA_ROOT` / `ASL_SHARDS_DIR` environment variables.
4. **Training**: reproduce a `slurm/*.sh` job inside a notebook cell by importing the modules directly instead of calling `src/train.py`:

   ```python
   from src.config import load_config
   from src.dataset import build_dataloaders
   from src.model import get_model
   from src.training import Trainer
   ```

   then build the loaders, model and trainer the same way `src/train.py` does.
