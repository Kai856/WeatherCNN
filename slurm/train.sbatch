#!/bin/bash
#SBATCH --job-name=hrrr_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/%x_%j.out
# Train the baseline CNN (best checkpoint by val loss), then score it on the test year.
# Extra arguments are passed to train.py, e.g. sbatch slurm/train.sbatch --epochs 40
set -e
source "$(dirname "$0")/env.sh" 2>/dev/null || source "$HOME/hrrr_cnn/slurm/env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv
python train.py --workers 8 "$@"
python evaluate.py
