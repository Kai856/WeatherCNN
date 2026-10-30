# Sourced by every job script. Edit PROJECT / HRRR_DATA_ROOT if your layout differs.
export PROJECT="${PROJECT:-$HOME/hrrr_cnn}"
export HRRR_DATA_ROOT="${HRRR_DATA_ROOT:-/cluster/scratch/$USER/hrrr_cnn}"
export PYTHONUNBUFFERED=1
source "$PROJECT/.venv/bin/activate"
cd "$PROJECT/scripts"
mkdir -p "$HRRR_DATA_ROOT" "$PROJECT/logs/slurm"
