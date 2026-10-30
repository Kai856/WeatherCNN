#!/bin/bash
#SBATCH --job-name=hrrr_pilot
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/%x_%j.out
# Stage B: 7-day pilot with a float32 comparison copy, then report, build, validate.
source "$(dirname "$0")/env.sh" 2>/dev/null || source "$HOME/hrrr_cnn/slurm/env.sh"
START=${PILOT_START:-2025-07-14T00}
END=${PILOT_END:-2025-07-20T23}
python process_grib.py --init-store
python process_grib.py --start "$START" --end "$END" --compare-float32
python pilot_report.py --start "$START" --end "$END"
python build_dataset.py
python validate_dataset.py
