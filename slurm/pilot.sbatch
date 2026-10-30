#!/bin/bash
#SBATCH --job-name=hrrr_pilot
#SBATCH --partition=batch
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/%x_%j.out
# Stage B: two pilot weeks (summer + winter) with a float32 comparison copy, then report, build, validate.
set -e
source "$(dirname "$0")/env.sh" 2>/dev/null || source "$HOME/hrrr_cnn/slurm/env.sh"
python process_grib.py --init-store
# second pass retries only hours that failed transiently (done hours are skipped)
for pass in 1 2; do
  python process_grib.py --start 2025-07-14T00 --end 2025-07-20T23 --compare-float32
  python process_grib.py --start 2022-01-10T00 --end 2022-01-16T23 --compare-float32
done
python pilot_report.py --window 2025-07-14T00 2025-07-20T23 --window 2022-01-10T00 2022-01-16T23
python build_dataset.py
python validate_dataset.py || true
