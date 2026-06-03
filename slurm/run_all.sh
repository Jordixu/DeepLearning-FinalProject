#!/usr/bin/env bash
# Submit training jobs to SLURM.
# Run from the cluster root:  cd $DATA/asl && bash slurm/run_all.sh
#
# Submits four jobs in parallel:
#   - baseline                 (1 job)
#   - deeper scratch models    (1 job: deep_cnn, deep_batchnorm)
#   - regularization variants  (array, 4 tasks: deep_batchnorm, deep_dropout, deep_bn_dropout, deep_bn_dropout_aug)
#   - transfer learning        (array, 4 tasks: resnet18, resnet50, mobilenet_v3_small, efficientnet_b0)

set -euo pipefail

echo "=== Submitting ASL training jobs ==="

# SLURM requires the log directory to exist before job submission.
mkdir -p "$HOME/LOGS"
mkdir -p "$HOME/results"
mkdir -p "$HOME/checkpoints"

JB=$(sbatch slurm/train_baseline.sh)
echo "  baseline:                              $JB"

JD=$(sbatch slurm/train_deeper.sh)
echo "  deeper scratch (deep_cnn, deep_bn):    $JD"

# 4 regularization variants run in parallel, each gets its own GPU.
# Tasks: 0=deep_batchnorm  1=deep_dropout  2=deep_bn_dropout  3=deep_bn_dropout_aug
JR=$(sbatch slurm/train_regularization.sh)
echo "  regularization (4 variants, parallel): $JR"

# 4 backbones run in parallel, each gets its own GPU.
# Tasks: 0=resnet18  1=resnet50  2=mobilenet_v3_small  3=efficientnet_b0
JT=$(sbatch slurm/train_transfer_array.sh)
echo "  transfer (4 backbones, parallel):      $JT"

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     \$HOME/LOGS/<job_id>.{out,err}"
echo "Results:  \$HOME/results/<exp_name>/"
