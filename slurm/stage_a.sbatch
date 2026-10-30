#!/bin/bash
#SBATCH --job-name=hrrr_stageA
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/slurm/%x_%j.out
# Stage A: archive inventory, grid metadata, full-range availability scan.
source "$(dirname "$0")/env.sh" 2>/dev/null || source "$HOME/hrrr_cnn/slurm/env.sh"
python inspect_archive.py --network-check
python inspect_archive.py
