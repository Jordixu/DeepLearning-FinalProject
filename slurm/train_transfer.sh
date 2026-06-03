#!/bin/bash
#SBATCH -J asl_transfer
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 64G
#SBATCH -N 1
#SBATCH --time=06:00:00
#SBATCH -o /home/%u/LOGS/%j.out
#SBATCH -e /home/%u/LOGS/%j.err

# Override MODEL and EXP_NAME via environment variables or edit below.
# Example:
#   MODEL=resnet50 EXP_NAME=resnet50_v1 sbatch slurm/train_transfer.sh

MODEL="${MODEL:-resnet50}"
EXP_NAME="${EXP_NAME:-${MODEL}_v1}"

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
