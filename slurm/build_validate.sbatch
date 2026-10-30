#!/bin/bash
#SBATCH --job-name=hrrr_build
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/%x_%j.out
# Stage E + validation after all months are processed.
source "$(dirname "$0")/env.sh" 2>/dev/null || source "$HOME/hrrr_cnn/slurm/env.sh"
python build_dataset.py --stats-stride "${STATS_STRIDE:-1}"
python validate_dataset.py --sample-hours 500
