#!/bin/bash
#SBATCH -J asl_transfer
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 64G
#SBATCH -N 1
#SBATCH --time=06:00:00
#SBATCH -o /home/%u/LOGS/%A_%a.out
#SBATCH -e /home/%u/LOGS/%A_%a.err
#SBATCH --array=0-3

# Each array task trains one backbone independently on its own GPU.
# Results land in:
#   $HOME/results/transfer_<backbone>_phase1/   <- frozen-head phase
#   $HOME/results/transfer_<backbone>/          <- full fine-tune + test eval
#
# Submit with:
#   sbatch slurm/train_transfer_array.sh
#
# Single backbone (e.g., task 2 = mobilenet_v3_small):
#   sbatch --array=2 slurm/train_transfer_array.sh

BACKBONES=(
  resnet18
  resnet50
  mobilenet_v3_small
  efficientnet_b0
)

MODEL="${BACKBONES[$SLURM_ARRAY_TASK_ID]}"
EXP_NAME="transfer_${MODEL}"

echo "[array] Task $SLURM_ARRAY_TASK_ID  ->  model=$MODEL  exp=$EXP_NAME"

module load conda
conda activate aabi

cd "$DATA/asl"
source slurm/_copy_shards.sh
source slurm/_setup_weights.sh

python src/train.py \
  --model "$MODEL" \
  --exp-name "$EXP_NAME" \
  --epochs 50 \
  --freeze-epochs 5 \
  --dropout 0.3
