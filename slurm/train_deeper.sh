#!/bin/bash
#SBATCH -J asl_deeper
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 64G
#SBATCH -N 1
#SBATCH --time=03:00:00
#SBATCH -o /home/%u/LOGS/%j.out
#SBATCH -e /home/%u/LOGS/%j.err

module load conda
conda activate aabi

cd "$DATA/asl"
source slurm/_copy_shards.sh

python src/train.py --model deep          --exp-name deep_cnn       --epochs 50
python src/train.py --model deep_batchnorm --exp-name deep_batchnorm --epochs 50
