#!/bin/bash
#SBATCH -J asl_reg
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 64G
#SBATCH -N 1
#SBATCH --time=04:00:00
#SBATCH -o /home/%u/LOGS/%A_%a.out
#SBATCH -e /home/%u/LOGS/%A_%a.err
#SBATCH --array=0-3

# Each array task trains one regularization variant independently on its own GPU.
# Results land in $HOME/results/<exp_name>/.
#
# Tasks:
#   0 = deep_batchnorm      (DeepCNNBatchNorm: BatchNorm only)
#   1 = deep_dropout        (DeepCNNRegularized: Dropout only)
#   2 = deep_bn_dropout     (DeepCNNBatchNormRegularized: BN + Dropout)
#   3 = deep_bn_dropout_aug (DeepCNNBatchNormRegularized: BN + Dropout + augmentation)
#
# Submit all 4 in parallel:
#   sbatch slurm/train_regularization.sh
#
# Single variant (e.g., task 1):
#   sbatch --array=1 slurm/train_regularization.sh

MODELS=(
  "deep_batchnorm             deep_batchnorm"
  "deep_regularized           deep_dropout"
  "deep_batchnorm_regularized deep_bn_dropout"
  "deep_batchnorm_regularized deep_bn_dropout_aug"
)

# Only the aug variant uses data augmentation
AUGMENT=(0 0 0 1)

ENTRY="${MODELS[$SLURM_ARRAY_TASK_ID]}"
MODEL=$(echo "$ENTRY" | awk '{print $1}')
EXP_NAME=$(echo "$ENTRY" | awk '{print $2}')
AUG_FLAG=""
[ "${AUGMENT[$SLURM_ARRAY_TASK_ID]}" -eq 1 ] && AUG_FLAG="--augment"

echo "[array] Task $SLURM_ARRAY_TASK_ID  ->  model=$MODEL  exp=$EXP_NAME  augment=${AUGMENT[$SLURM_ARRAY_TASK_ID]}"

module load conda
conda activate aabi

cd "$DATA/asl"
source slurm/_copy_shards.sh

python src/train.py \
  --model "$MODEL" \
  --exp-name "$EXP_NAME" \
  --epochs 50 \
  $AUG_FLAG
