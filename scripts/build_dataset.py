"""Stage E: build the sample index, chronological splits, and train-only normalization statistics.

Sample definition (one per input hour t):
  X[t] = inputs[t]             all 49 channels valid at t (HRRR analysis; precip = 1 h forecast from init t-1)
  y[t] = target_fields[t + 24] 7 values at the Tufts cell valid at t + 24 h
A sample exists only if every input channel at t is valid, all targets at t+24 are finite, and t and t+24
fall in the same split. Nothing is interpolated.

Outputs
  metadata/samples.parquet          t, target_time, split, store indices
  metadata/normalization_stats.json per-channel input stats + target stats (training split only)
  metadata/missing_timestamps.csv   archive gaps + processing problems
  metadata/dataset_summary.json

Usage
  python scripts/build_dataset.py [--stats-stride 1]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import zarr

from common import (INT16_MISSING, LOG_DIR, METADATA_DIR, TARGET_NAMES, TARGET_UNITS, du_bytes, load_config,
                    load_grid_metadata, read_json, setup_logging, write_json)

log = setup_logging("build_dataset")


class RunningStats:
    """Per-channel count/mean/M2 merged across batches (Chan et al. parallel algorithm), plus min/max."""

    def __init__(self, n):
        self.n = np.zeros(n)
        self.mean = np.zeros(n)
        self.m2 = np.zeros(n)
        self.min = np.full(n, np.inf)
        self.max = np.full(n, -np.inf)

    def update(self, x):  # x: (C, N) float64 with NaNs
        cnt = np.isfinite(x).sum(axis=1)
        ok = cnt > 0
        mu = np.where(ok, np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=1), 0)
        m2 = np.where(ok, np.nansum((x - mu[:, None]) ** 2, axis=1), 0)
        tot = self.n + cnt
        delta = mu - self.mean
        with np.errstate(invalid="ignore", divide="ignore"):
            self.mean = np.where(tot > 0, self.mean + delta * cnt / np.maximum(tot, 1), self.mean)
            self.m2 = self.m2 + m2 + delta ** 2 * self.n * cnt / np.maximum(tot, 1)
        self.n = tot
        self.min = np.fmin(self.min, np.nanmin(x, axis=1))
        self.max = np.fmax(self.max, np.nanmax(x, axis=1))

    def std(self):
        return np.sqrt(self.m2 / np.maximum(self.n - 1, 1))


def split_of(cfg, times: pd.DatetimeIndex) -> np.ndarray:
    out = np.full(len(times), "", dtype=object)
    for name, (a, b) in cfg["splits"].items():
        out[(times >= pd.Timestamp(a)) & (times <= pd.Timestamp(b))] = name
    return out


def open_store(cfg):
    root = zarr.open_group(str(cfg["store_path"]), mode="r")
    times = pd.to_datetime(root["time"][:], unit="h", origin="unix")
    return root, pd.DatetimeIndex(times)


def read_inputs(root, t: int, order: list[str]) -> np.ndarray:
    """All channels of record t as float32 (C, y, x) in canonical channel order."""
    loc = root.attrs["channel_location"]
    parts = {}
    if "inputs" in root:
        a = root["inputs"]
        q = a[t]
        sc = np.asarray(a.attrs["scale"], dtype=np.float32)[:, None, None]
        of = np.asarray(a.attrs["offset"], dtype=np.float32)[:, None, None]
        x = q.astype(np.float32) * sc + of
        x[q == INT16_MISSING] = np.nan
        parts["inputs"] = x
    if "inputs_f32" in root:
        parts["inputs_f32"] = root["inputs_f32"][t]
    return np.stack([parts[loc[c][0]][loc[c][1]] for c in order])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-stride", type=int, default=1, help="use every k-th training hour for input stats")
    args = ap.parse_args()
    cfg = load_config()
    grid = load_grid_metadata()
    root, times = open_store(cfg)
    order = root.attrs["channel_order"]
    lead = cfg["lead_hours"]
    T = len(times)

    done = root["done"][:]
    valid = root["input_valid"][:]
    targets = root["target_fields"][:]
    input_ok = done & valid.all(axis=1)
    target_ok = done & np.isfinite(targets).all(axis=1)
    split = split_of(cfg, times)

    ti = np.arange(T - lead)
    yi = ti + lead
    keep = input_ok[ti] & target_ok[yi] & (split[ti] != "") & (split[ti] == split[yi])
    samples = pd.DataFrame({"t": times[ti[keep]], "target_time": times[yi[keep]], "split": split[ti[keep]],
                            "t_index": ti[keep], "target_index": yi[keep]})
    samples.to_parquet(METADATA_DIR / "samples.parquet", index=False)
    counts = samples.split.value_counts().to_dict()
    log.info("samples: %s", counts)

    # ---- normalization (training split only)
    train = samples[samples.split == "train"]
    in_stats = RunningStats(len(order))
    stat_idx = train.t_index.values[:: args.stats_stride]
    for k, t in enumerate(stat_idx):
        x = read_inputs(root, int(t), order).reshape(len(order), -1).astype(np.float64)
        in_stats.update(x)
        if k % 500 == 0:
            log.info("input stats %d/%d", k, len(stat_idx))
    tgt = targets[train.target_index.values].astype(np.float64)
    cont = TARGET_NAMES[:6]
    norm = {
        "fit_on": "train split only",
        "train_period": cfg["splits"]["train"],
        "n_train_samples": int(len(train)),
        "input_hours_used": int(len(stat_idx)),
        "input_stats_stride": args.stats_stride,
        "inputs": {c: {"mean": float(in_stats.mean[k]), "std": float(in_stats.std()[k]),
                       "min": float(in_stats.min[k]), "max": float(in_stats.max[k]), "n_values": int(in_stats.n[k]),
                       "units": u}
                   for k, (c, u) in enumerate(zip(order, [ch["units"] for ch in cfg["channels"]]))},
        "input_static": {},
        "targets": {n: {"mean": float(np.mean(tgt[:, k])) if len(tgt) else None,
                        "std": float(np.std(tgt[:, k], ddof=1)) if len(tgt) > 1 else None,
                        "units": TARGET_UNITS[k]} for k, n in enumerate(cont)},
        "rain_binary": {"positive_rate": float(tgt[:, 6].mean()) if len(tgt) else None,
                        "n_positive": int(tgt[:, 6].sum()) if len(tgt) else 0,
                        "pos_weight_neg_over_pos": float((len(tgt) - tgt[:, 6].sum()) / tgt[:, 6].sum())
                        if len(tgt) and tgt[:, 6].sum() else None,
                        "threshold_mm": cfg["rain_threshold_mm"],
                        "note": "binary target is not normalized; use a classification loss (e.g. BCEWithLogits with pos_weight)"},
        "usage": "x_norm = (x - mean) / std per channel; NaN never occurs in kept samples",
    }
    for s in cfg["static_fields"]:
        a = root[f"static_{s['name']}"][:]
        norm["input_static"][s["name"]] = {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)), "units": s["units"]}
    write_json(METADATA_DIR / "normalization_stats.json", norm)

    # ---- missing timestamps: archive gaps + processing problems
    parts = []
    arch = METADATA_DIR / "missing_timestamps.csv"
    if arch.exists():
        a = pd.read_csv(arch)
        a["valid_time"] = pd.to_datetime(a.valid_time, format="mixed")
        parts.append(a[a.get("stage", "archive_scan") == "archive_scan"] if "stage" in a else a.assign(stage="archive_scan"))
    fail = LOG_DIR / "process_failures.csv"
    if fail.exists():
        f = pd.read_csv(fail)
        f["valid_time"] = pd.to_datetime(f.valid_time, format="mixed")
        # transient errors ("*" rows) are resolved once the hour is processed; channel problems are resolved
        # once a re-run (--redo-invalid) makes the hour fully valid
        if len(f):
            is_done = f.valid_time.isin(times[done])
            still_bad = f.valid_time.isin(times[done & ~(input_ok & target_ok)])
            f = f[((f.channel == "*") & ~is_done) | ((f.channel != "*") & still_bad)]
        if len(f):
            g = f.groupby("valid_time").apply(lambda d: "; ".join(f"{c}: {r}" for c, r in zip(d.channel, d.reason)),
                                              include_groups=False)
            parts.append(pd.DataFrame({"valid_time": g.index, "missing": g.values, "stage": "processing"}))
    bad_done = times[done & ~(input_ok & target_ok)]
    parts.append(pd.DataFrame({"valid_time": bad_done, "missing": "record not fully valid (see input_valid/target_fields)",
                               "stage": "store"}))
    miss = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else \
        pd.DataFrame(columns=["valid_time", "missing", "stage"])
    miss = miss.sort_values("valid_time")
    miss.to_csv(METADATA_DIR / "missing_timestamps.csv", index=False)

    # ---- summary
    pilot = METADATA_DIR / "pilot_summary.json"
    summary = {
        "task": root.attrs["task"],
        "period": cfg["period"], "lead_hours": lead,
        "store": str(cfg["store_path"]), "store_bytes": du_bytes(cfg["store_path"]),
        "hours_in_period": T, "hours_processed": int(done.sum()), "hours_fully_valid": int((input_ok & target_ok).sum()),
        "hours_not_yet_processed": int((~done).sum()),
        "samples_per_split": {k: int(counts.get(k, 0)) for k in cfg["splits"]},
        "splits": cfg["splits"],
        "split_rule": "t and t+24h must both lie in the same split; the last 24 h of each period only serve as targets",
        "input_shape_per_sample": [len(order), grid["coarse_grid"]["ny"], grid["coarse_grid"]["nx"]],
        "input_channels": [{"name": c["name"], "grib": f"{c['var']}:{c['level']}", "units": c["units"],
                            "source": c["source"], "int16_step": c["scale"]} for c in cfg["channels"]],
        "static_fields": [s["name"] for s in cfg["static_fields"]],
        "targets": [{"name": n, "units": u} for n, u in zip(TARGET_NAMES, TARGET_UNITS)],
        "target_cell": grid["tufts"],
        "grid": {"native": "HRRR 3 km Lambert conformal", "coarse": grid["coarse_grid"] | {"x_m": "...", "y_m": "..."},
                 "crop_window_native": grid["crop_window_native"]},
        "unit_conversions": root.attrs["unit_conversions"],
        "storage": {"format": "Zarr v3", "inputs_dtype": "int16 (per-channel scale/offset)",
                    "chunks": "(1 hour, all channels, y, x)", "filters": "Delta (int16) + Blosc zstd5 bitshuffle",
                    "float32_channels": cfg["storage"]["float32_channels"]},
        "normalization_stats": "metadata/normalization_stats.json",
        "pilot": read_json(pilot) if pilot.exists() else None,
    }
    write_json(METADATA_DIR / "dataset_summary.json", summary)
    log.info("wrote samples.parquet, normalization_stats.json, missing_timestamps.csv, dataset_summary.json")


if __name__ == "__main__":
    main()
