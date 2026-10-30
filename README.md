# HRRR Northeast CNN dataset (6 km, 3 years)

Hourly NOAA HRRR fields over the Northeast US (38–47°N, 85–66°W) for training a CNN that maps weather maps at
time *t* to conditions at Tufts University at *t + 24 h*. The dataset covers 2023-10-01 00Z – 2026-09-30 23Z
(26,304 hours) and is designed to stay within a 100 GB scratch budget.

## What the dataset is

| | |
|---|---|
| Inputs `X[t]` | 49 channels on a 223 × 291 grid (6 km, 2×2 block mean of HRRR's native 3 km Lambert grid) |
| Static | orography and land fraction on the same grid (stored once) |
| Targets `y[t]` | 7 values at the native 3 km cell nearest Tufts, valid at *t + 24 h* |
| Splits | train 2023-10-01…2025-09-30, val 2025-10-01…2026-03-31, test 2026-04-01…2026-09-30 (by input time; the target must fall in the same split) |

**Task definition.** Inputs are the HRRR analysis valid at *t*. Targets are the HRRR analysis valid at *t + 24 h*
(precipitation: HRRR's 1-hour forecast from the *t + 23 h* initialization, the only source of a true 1-hour
total valid at *t + 24 h*). The model therefore learns to predict **future HRRR analysis conditions**, not
station observations.

**Input channels** (order is fixed; see `metadata/dataset_summary.json`):

- Near surface (14): `t2m d2m rh2m u10 v10 gust sp skt mslp cape cin pwat dswrf apcp1h`
- Pressure levels 1000/925/850/700/500/300/250 hPa (35): `t*`, `dpt*`, `u*`, `v*`, `gh*`

Sources per valid hour *v*: `wrfsfcf00` (init *v*) for everything available there, `wrfprsf00` (init *v*) for
HGT 925/300/250 and TMP/DPT 300/250, and `wrfsfcf01` (init *v − 1 h*) for `APCP 0-1 hour acc`. The `f00`
APCP field is a zero-length accumulation and is never used. Every message's valid time is checked against *v*.

**Targets** (`target_fields`, float32): `t2m` K, `rh2m` %, `u10`/`v10` m s⁻¹ (earth-relative), `gust` m s⁻¹,
`precip_1h` mm, `rain_binary` (1 if `precip_1h` > 2.0 mm). HRRR winds are grid-relative; input winds are kept
grid-relative, target winds are rotated to earth-relative at the Tufts cell (rotation ≈ 16.4°).

**Units.** Inputs stay in GRIB SI units (K, Pa, m s⁻¹, %, J kg⁻¹, kg m⁻², W m⁻², gpm). The only conversions are
APCP kg m⁻² → mm (numerically identical) and the wind rotation above. Both are recorded in the store attributes.

## Storage format

`<data_root>/processed/hrrr_ne_6km.zarr` (Zarr v3):

| Array | Shape | Notes |
|---|---|---|
| `inputs` | (time, 49, 223, 291) int16 | chunk = 1 hour; Delta filter + Blosc zstd/bitshuffle; `value = stored * scale[c] + offset[c]`, `-32768` = missing |
| `input_valid` | (time, 49) bool | channel decoded and passed all checks |
| `target_fields` | (time, 7) float32 | values at Tufts valid at `time` (not shifted) |
| `done`, `record_ok` | (time,) bool | resume marker; all channels + targets valid |
| `static_orography`, `static_land`, `lat`, `lon`, `bbox_mask` | (223, 291) | `bbox_mask` = cell center inside the requested lat/lon box |
| `time`, `y`, `x` | | hours since 1970; projected Lambert x/y (m) |

The int16 step for each channel is the finest GRIB packing step HRRR uses for that variable (e.g. 0.0625 K,
0.0625 m s⁻¹, 0.03125 m, 10 Pa), with the offset aligned to that step. So the stored values sit on HRRR's own
value grid, and the maximum error versus float32 is half of HRRR's packing step. The pilot verifies this per
channel (`logs/pilot_report.md`). Channels listed in `config.yaml → storage.float32_channels` are stored as
float32 in `inputs_f32` instead.

Pilot result (2025-07-14…20, 168 hours, all valid): 1.88 MB per hour, projecting to **49.5 GB** for 3 years
(the gate is 70 GB). The encodings compared in `logs/pilot_report.md` project to 103 GB for compressed float32,
82–100 GB for int16 with a bounds-derived step, and 49 GB for the chosen source-step int16 + Delta encoding. All
49 channels meet the precision rule (max error ≤ half the GRIB packing step, allowing 2 float32 ulps for the
float32 reference itself). The download is about 43 MB per hour, about 1.1 TB in total, streamed through `tmp/`
and deleted per hour.

Reading a sample:

```python
import sys; sys.path.insert(0, "scripts")
import pandas as pd, zarr
from build_dataset import read_inputs
from common import load_config
cfg = load_config()
root = zarr.open_group(str(cfg["store_path"]), mode="r")
samples = pd.read_parquet("metadata/samples.parquet")
s = samples.iloc[0]
x = read_inputs(root, int(s.t_index), root.attrs["channel_order"])   # (49, 223, 291) float32
y = root["target_fields"][int(s.target_index)]                      # (7,)
```

Normalize with `metadata/normalization_stats.json`, which is computed on the training split only. Continuous
targets get mean/std. `rain_binary` stays binary; use a classification loss, for example `BCEWithLogitsLoss`
with the stored `pos_weight`.

## Layout

```
config.yaml                 all settings (dates, bbox, channels, bounds, splits, storage)
scripts/common.py           shared helpers
scripts/inspect_archive.py  Stage A
scripts/download_hrrr.py    byte-range downloader (used by process_grib.py)
scripts/process_grib.py     Stages C/D
scripts/pilot_report.py     Stage B report + storage gate
scripts/build_dataset.py    Stage E
scripts/validate_dataset.py quality checks
slurm/                      job scripts for Pax
metadata/                   inventory, grid, provenance, stats, summaries (small; keep in home)
logs/                       logs and reports
data/                       default data root (set HRRR_DATA_ROOT to use scratch)
```

## Setup on Pax

```bash
# code + metadata in home, data in scratch
rsync -av --exclude .venv --exclude data hrrr_cnn/ kkapli01@login-prod.pax.tufts.edu:~/hrrr_cnn/
ssh kkapli01@login-prod.pax.tufts.edu
cd ~/hrrr_cnn
module load python   # or any Python >= 3.10
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
export HRRR_DATA_ROOT=/cluster/scratch/$USER/hrrr_cnn
mkdir -p logs/slurm   # SLURM opens job output files before the job starts; submit jobs from ~/hrrr_cnn
# confirm compute nodes can reach S3 (otherwise run downloads on a login/transfer node)
srun -p batch -t 00:05:00 python scripts/inspect_archive.py --network-check
```

## Running each stage

All job scripts source `slurm/env.sh`, which sets `HRRR_DATA_ROOT=/cluster/scratch/$USER/hrrr_cnn` and
activates the venv. Run the stages in order:

1. **Stage A** (`sbatch slurm/stage_a.sbatch`, or `python scripts/inspect_archive.py`): writes
   `metadata/variable_inventory.csv`, `grid_metadata.json`, `archive_availability.csv` and
   `missing_timestamps.csv`. Already run: all 26,304 hours have every required file.
2. **Stage B pilot** (`sbatch slurm/pilot.sbatch`): initializes the store, processes 2025-07-14…20 with a
   float32 comparison copy, then writes `logs/pilot_report.md` and `metadata/pilot_summary.json`, and runs
   the build and validation steps. The full run is blocked until `gate_pass` is true (projected total ≤ 70 GB
   and every channel within the precision rule). The pilot has already passed (run locally), and the rsynced
   `metadata/pilot_summary.json` carries that result. Re-running it on Pax is still recommended because it
   creates the scratch store and measures Pax download speed. If you skip it, run
   `python scripts/process_grib.py --init-store` once before the full run.
3. **Full run** (`sbatch slurm/process_months.sbatch`): 36 array tasks, one calendar month each, at most six
   at a time. Afterwards you can delete the pilot-only `hrrr_ne_6km_f32compare.zarr`.
4. **Build + validate** (`sbatch slurm/build_validate.sbatch`): writes `metadata/samples.parquet`,
   `normalization_stats.json`, `missing_timestamps.csv`, `dataset_summary.json` and
   `logs/validation_report.md`. `STATS_STRIDE=6` computes the input statistics from every 6th training hour
   if you want it faster.

## Resuming interrupted jobs

- Re-run the same command or re-submit the same array (or only the failed task IDs, e.g.
  `sbatch --array=7,12 slurm/process_months.sbatch`). Hours whose `done` flag is set are skipped.
- `done[t]` is written last for each hour, after inputs, validity flags and targets. A job killed mid-hour
  leaves that hour undone, so it is redone in full.
- Network or transient errors leave the hour undone and are logged to `logs/process_failures.csv` as
  `processing error (left undone)`. Re-running fixes them.
- Data problems (missing message, wrong valid time, values outside bounds) mark the channel invalid,
  set `done`, and are listed in `metadata/missing_timestamps.csv` by `build_dataset.py`. Samples that need
  invalid data are excluded. Nothing is interpolated.
- Partial downloads in `<data_root>/tmp/` are detected (no complete `manifest.json`) and fetched again.
- Months are day-aligned, so concurrent array tasks never write the same Zarr chunk. Don't run two jobs on
  the same days at once.

## Scratch purge warning

`/cluster/scratch` deletes files that haven't been accessed in 21 days and has no backups. Code, `config.yaml`
and `metadata/` (including provenance) live in home and are enough to rebuild everything. To rebuild, re-run
Stages B–E with the same config: the archive is public and the pipeline is deterministic. Reading the dataset
during training counts as access. Don't `touch` files to dodge the purge; Tufts prohibits it.

## Quality checks

`scripts/validate_dataset.py` checks:
- Array shapes and that every variable uses the same grid.
- Time alignment: init + lead = valid, and each GRIB valid time equals the record time.
- The exact +24 h target offset.
- The 1 h precipitation interval.
- That no input uses data from after its valid time.
- Duplicate timestamps.
- Missing-data fractions.
- Physical ranges.
- The Tufts cell, by brute-force great-circle search.
- That splits don't overlap.
- The 100 GB budget.

The report goes to `logs/validation_report.md`.
