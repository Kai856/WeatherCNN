"""Reference forecasts for the Tufts targets 24 h ahead, scored on val (last training year) and test.

  persistence  y(t+24) = y(t)
  climatology  mean over the reference period for the same hour of day and day of year (+-15 days)
               (reference = fit period when scoring val, whole train split when scoring test)

Metrics: RMSE for t2m, rh2m, u10, v10, gust; MAE for precip_1h; Brier, AUC, F1 for rain_binary.

Output: logs/baselines.md

Usage
  python scripts/baselines.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import zarr

from common import LOG_DIR, METADATA_DIR, TARGET_NAMES, load_config, setup_logging

RMSE_TARGETS = ["t2m", "rh2m", "u10", "v10", "gust"]
UNITS = {"t2m": "K", "rh2m": "%", "u10": "m/s", "v10": "m/s", "gust": "m/s", "precip_1h": "mm"}
CLIM_WINDOW_DAYS = 15


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve (rank formulation, ties averaged)."""
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def score(y_true: np.ndarray, y_pred: np.ndarray, rain_prob: np.ndarray) -> dict:
    """y_true, y_pred: (N, 7) in real units (y_pred[:, 6] unused); rain_prob: (N,) probability of rain_binary."""
    out = {}
    for c in RMSE_TARGETS:
        k = TARGET_NAMES.index(c)
        out[f"{c} RMSE"] = float(np.sqrt(np.nanmean((y_pred[:, k] - y_true[:, k]) ** 2)))
    k = TARGET_NAMES.index("precip_1h")
    out["precip_1h MAE"] = float(np.nanmean(np.abs(y_pred[:, k] - y_true[:, k])))
    rb = y_true[:, 6]
    ok = np.isfinite(rain_prob)
    out["rain Brier"] = float(np.mean((rain_prob[ok] - rb[ok]) ** 2))
    out["rain AUC"] = auc(rb[ok], rain_prob[ok])
    hit = rain_prob[ok] >= 0.5
    tp, fp, fn = (hit & (rb[ok] == 1)).sum(), (hit & (rb[ok] == 0)).sum(), (~hit & (rb[ok] == 1)).sum()
    out["rain F1"] = float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else float("nan")
    return out


def climatology(ref_times: pd.DatetimeIndex, ref_y: np.ndarray, query_times: pd.DatetimeIndex) -> np.ndarray:
    """Mean of ref_y over the same hour of day and day of year +- CLIM_WINDOW_DAYS (circular)."""
    table = np.full((367, 24, ref_y.shape[1]), np.nan)
    rdoy, rhour = ref_times.dayofyear.to_numpy(), ref_times.hour.to_numpy()
    for h in range(24):
        yh, dh = ref_y[rhour == h], rdoy[rhour == h]
        for d in range(1, 367):
            diff = np.abs(dh - d)
            win = np.minimum(diff, 366 - diff) <= CLIM_WINDOW_DAYS
            if win.any():
                table[d, h] = np.nanmean(yh[win], axis=0)
    return table[query_times.dayofyear.to_numpy(), query_times.hour.to_numpy()]


def reference_forecasts(root, samples, eval_period, ref_period):
    """Persistence and climatology predictions (real units) and the truth for samples in eval_period."""
    yall = root["target_fields"][:].astype(np.float64)
    tall = pd.DatetimeIndex(pd.to_datetime(root["time"][:], unit="h", origin="unix"))
    ev = samples[(samples.t >= pd.Timestamp(eval_period[0])) & (samples.t <= pd.Timestamp(eval_period[1]))]
    ref = samples[(samples.t >= pd.Timestamp(ref_period[0])) & (samples.t <= pd.Timestamp(ref_period[1]))]
    truth = yall[ev.target_index.values]
    persist = yall[ev.t_index.values]
    clim = climatology(tall[ref.target_index.values], yall[ref.target_index.values], tall[ev.target_index.values])
    return truth, {"persistence": (persist, persist[:, 6]), "climatology": (clim, clim[:, 6])}


def format_table(results: dict[str, dict]) -> list[str]:
    """results: {method: metrics} -> markdown table with one row per metric."""
    methods = list(results)
    metrics = list(next(iter(results.values())))
    lines = ["| Metric | " + " | ".join(methods) + " |", "|---" * (len(methods) + 1) + "|"]
    for m in metrics:
        unit = UNITS.get(m.split()[0], "")
        label = f"{m} [{unit}]" if unit and m.split()[1] in ("RMSE", "MAE") else m
        lines.append(f"| {label} | " + " | ".join(f"{results[k][m]:.3f}" for k in methods) + " |")
    return lines


def main():
    log = setup_logging("baselines")
    cfg = load_config()
    root = zarr.open_group(str(cfg["store_path"]), mode="r")
    samples = pd.read_parquet(METADATA_DIR / "samples.parquet")
    ms, sp = cfg["model_selection"], cfg["splits"]
    lines = ["# Reference forecasts (24 h ahead at Tufts)", "",
             "Persistence: y(t+24) = y(t). Climatology: reference-period mean for the same hour and day of year "
             f"(+-{CLIM_WINDOW_DAYS} days). Rain probability for persistence is 0/1.", ""]
    for name, period, ref in (("val", ms["val"], ms["fit"]), ("test", sp["test"], sp["train"])):
        truth, preds = reference_forecasts(root, samples, period, ref)
        if not len(truth):
            log.warning("%s: no samples in %s .. %s", name, *period)
            continue
        res = {m: score(truth, p, rp) for m, (p, rp) in preds.items()}
        lines += [f"## {name}: {period[0]} .. {period[1]} ({len(truth):,} samples; climatology from {ref[0]} .. {ref[1]})",
                  "", *format_table(res), ""]
        log.info("%s: %s", name, {m: {k: round(v, 3) for k, v in r.items()} for m, r in res.items()})
    (LOG_DIR / "baselines.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
