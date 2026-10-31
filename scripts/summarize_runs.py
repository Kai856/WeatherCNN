"""Collect every training run into one report: settings, training history, test-year results.

Reads checkpoints/*.pt (settings), logs/train_<name>.csv (per-epoch val metrics) and logs/eval_<name>.md
(test tables). Output: logs/experiments_summary.md

Usage
  python scripts/summarize_runs.py
"""
from __future__ import annotations

import pandas as pd
import torch

from common import LOG_DIR, PROJECT_ROOT, load_config

CKPT_DIR = PROJECT_ROOT / "checkpoints"
METRICS = ["t2m RMSE [K]", "rh2m RMSE [%]", "u10 RMSE [m/s]", "v10 RMSE [m/s]", "gust RMSE [m/s]",
           "precip_1h MAE [mm]", "rain Brier", "rain AUC", "rain F1"]

METHODS = """## Methods

**Data.** HRRR analyses over the Northeast, cropped and coarsened 2x2 to a 223 x 291 grid (about 6 km),
hourly, Oct 2021 - Sep 2026. Inputs at time t: 49 weather channels (apcp1h, cape, cin log-transformed),
orography, land mask, sin/cos of hour of day and day of year; all z-scored with training-split statistics.
Targets: the Tufts grid cell 24 h later: t2m, rh2m, u10, v10, gust, precip_1h (log1p), rain_binary.

**Splits.** fit = Oct 2021 - Sep 2024 (training), val = Oct 2024 - Sep 2025 (model selection),
test = Oct 2025 - Sep 2026 (reported only).

**Reference forecasts.** Persistence: y(t+24) = y(t). Climatology: mean of the reference years at the same
hour of day within +-15 days of the date; the rain probability is the fraction of rainy hours.

**Model.** CNN: 3x3 stem, 5 stride-2 blocks of [conv, BatchNorm, ReLU] x 2 (32/64/128/128/128 channels),
final map flattened (optionally average-pooled to 4 x 5 first) so the head keeps location, then dropout 0.3,
dense 256, dense 7 (6 continuous outputs + rain logit). Options: kernel size, previous-day input channels
(t - 24 h, 104 channels total), delta targets (predict the 24 h change).

**Training.** Loss = MSE on the 6 normalized continuous targets + BCEWithLogits(pos_weight) on rain.
AdamW, cosine learning-rate schedule, batch 32, mixed precision, every 3rd hour per epoch (random offset).
Early stopping (patience 5) on either val loss (older runs) or val skill (newer runs).

**Val skill.** Mean over t2m, rh2m, u10, v10, gust of RMSE_model / RMSE_climatology, plus
(1 - AUC_model) / (1 - AUC_climatology) for rain. 1 = climatology, lower is better.

**Evaluation.** Test-year RMSE (MAE for precip), rain Brier, AUC, F1. Newer evaluations Platt-calibrate the
rain probability on val and tune each method's F1 threshold on val; older ones use raw probabilities and 0.5.
"""


def settings(ckpt: dict) -> dict:
    a, kw = ckpt.get("args", {}), ckpt.get("model_kwargs") or {}
    n = sum(v.numel() for v in ckpt["model"].values() if v.dtype.is_floating_point and v.dim() > 0)
    return {"prev_hours": ckpt.get("prev_hours", 0), "delta": ckpt.get("delta", False),
            "pos_weight": round(ckpt.get("pos_weight") or 62.5, 1), "lr": a.get("lr"),
            "weight_decay": a.get("weight_decay"), "pool": kw.get("pool"), "kernel": kw.get("kernel", 3),
            "hidden": kw.get("hidden", 256), "seed": a.get("seed"), "final": a.get("final", False),
            "selection": "val skill" if "val_skill" in ckpt else "val loss", "params": f"{n / 1e6:.2f}M"}


def history(name: str) -> dict:
    path = LOG_DIR / f"train_{name}.csv"
    if not path.exists():
        return {}
    h = pd.read_csv(path)
    col = "val_skill" if "val_skill" in h else "val_loss"
    best = h.loc[h[col].idxmin()]
    return {"epochs run": len(h), "best epoch": int(best.epoch), f"best {col}": round(float(best[col]), 4),
            "val t2m RMSE at best": round(float(best["val t2m RMSE"]), 3)}


def eval_table(name: str) -> dict[str, dict[str, float]] | None:
    path = LOG_DIR / f"eval_{name}.md"
    if not path.exists():
        return None
    rows = [l for l in path.read_text().splitlines() if l.startswith("| ") and not l.startswith("| Metric")]
    head = next(l for l in path.read_text().splitlines() if l.startswith("| Metric"))
    cols = [c.strip() for c in head.strip("|").split("|")][1:]
    table = {}
    for r in rows:
        cells = [c.strip() for c in r.strip("|").split("|")]
        table[cells[0]] = dict(zip(cols, map(float, cells[1:])))
    return table


def cell(v) -> str:
    if isinstance(v, (list, tuple)):
        return "x".join(map(str, v))
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v)


def md_table(df: pd.DataFrame) -> list[str]:
    cols = [df.index.name or "", *map(str, df.columns)]
    lines = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx), *map(cell, row)]) + " |")
    return lines


def main():
    cfg = load_config()
    names = sorted(p.stem for p in CKPT_DIR.glob("*.pt"))
    runs = {}
    for n in names:
        ckpt = torch.load(CKPT_DIR / f"{n}.pt", map_location="cpu", weights_only=False)
        runs[n] = {**settings(ckpt), **history(n)}
    runs_df = pd.DataFrame(runs).T
    runs_df.index.name = "run"

    tables = {n: eval_table(n) for n in names}
    test = pd.DataFrame({n: {m: t[m]["CNN"] for m in METRICS if m in t} for n, t in tables.items() if t})
    ref = next((t for t in tables.values() if t), None)
    if ref:
        for m in ("persistence", "climatology"):
            test[m] = [ref[k][m] for k in test.index]
    test.index.name = "metric"

    lines = ["# HRRR -> Tufts 24 h CNN: experiments summary", "",
             f"Splits: {cfg['model_selection']} (model selection), test {cfg['splits']['test']}.", "",
             METHODS, "## Runs (settings and validation)", "",
             "pos_weight 62.5 = the default neg/pos ratio. Skill and loss are only comparable within a column "
             "type; val loss also depends on pos_weight.", "", *md_table(runs_df), "",
             "## Test-year results (CNN column of each run's evaluation)", "",
             "Lower is better except AUC and F1. Check each run's `selection` and evaluation type above.", "",
             *(md_table(test) if not test.empty else ["(no evaluations found)"]), ""]
    for n, t in tables.items():
        if t:
            lines += [f"### {n}", "", *(LOG_DIR / f"eval_{n}.md").read_text().splitlines()[2:], ""]
    out = LOG_DIR / "experiments_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(names)} runs)")


if __name__ == "__main__":
    main()
