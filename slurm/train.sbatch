#!/bin/bash
#SBATCH --job-name=hrrr_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/%x_%j.out
# Train the baseline CNN (best checkpoint by val loss), then score it on the test year.
# Extra arguments are passed to train.py; RUN_NAME names the checkpoint and logs, e.g.
#   RUN_NAME=cnn_v2 sbatch slurm/train.sbatch --pos-weight 8 --lr 3e-4 --weight-decay 1e-3 --pool 4 5
set -e
source "$(dirname "$0")/env.sh" 2>/dev/null || source "$HOME/hrrr_cnn/slurm/env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), f'torch {torch.__version__} cannot use the GPU'"
python train.py --workers 8 --name "${RUN_NAME:-baseline_cnn}" "$@"
python evaluate.py --name "${RUN_NAME:-baseline_cnn}"
