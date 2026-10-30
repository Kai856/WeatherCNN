#!/bin/bash
#SBATCH --job-name=hrrr_month
#SBATCH --partition=batch
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-35%6
#SBATCH --output=logs/slurm/%x_%A_%a.out
# Full run: one calendar month per array task (2023-10 .. 2026-09 = 36 months), at most 6 at once.
# Months are day-aligned, so tasks never write the same Zarr chunk. Re-submitting resumes: hours already
# marked done are skipped. Refuses to run unless metadata/pilot_summary.json has gate_pass = true.
source "$(dirname "$0")/env.sh" 2>/dev/null || source "$HOME/hrrr_cnn/slurm/env.sh"
python process_grib.py --month-index "$SLURM_ARRAY_TASK_ID"
