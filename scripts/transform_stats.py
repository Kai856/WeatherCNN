"""Train-only mean/std of the log-transformed skewed channels (and the log1p precipitation target).

Skewed channels (mostly zero, rare large values) are log-transformed before z-scoring:
  apcp1h, cape:  log1p(x)
  cin (<= 0):    -log1p(-x)

Output: metadata/transform_stats.json

Usage
  python scripts/transform_stats.py [--stride 3]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from build_dataset import RunningStats, open_store, read_inputs
from common import METADATA_DIR, TARGET_NAMES, load_config, setup_logging, write_json

def neg_log1p(x):
    return -np.log1p(-x)


def neg_expm1(y):
    return -np.expm1(-y)


TRANSFORMS = {
    "apcp1h": (np.log1p, np.expm1),
    "cape": (np.log1p, np.expm1),
    "cin": (neg_log1p, neg_expm1),
}
TARGET_TRANSFORMS = {"precip_1h": (np.log1p, np.expm1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3, help="use every k-th training hour")
    args = ap.parse_args()
    log = setup_logging("transform_stats")
    cfg = load_config()
    root, _ = open_store(cfg)
    order = list(root.attrs["channel_order"])
    names = list(TRANSFORMS)
    ks = [order.index(c) for c in names]

    samples = pd.read_parquet(METADATA_DIR / "samples.parquet")
    train = samples[samples.split == "train"]
    idx = train.t_index.values[:: args.stride]
    stats = RunningStats(len(names))
    for n, t in enumerate(idx):
        x = read_inputs(root, int(t), order)[ks].reshape(len(names), -1).astype(np.float64)
        stats.update(np.stack([TRANSFORMS[c][0](x[j]) for j, c in enumerate(names)]))
        if n % 1000 == 0:
            log.info("%d/%d hours", n, len(idx))

    y = root["target_fields"][:][train.target_index.values].astype(np.float64)
    out = {
        "fit_on": "train split only",
        "input_hours_used": int(len(idx)),
        "stride": args.stride,
        "transforms": {"apcp1h": "log1p(x)", "cape": "log1p(x)", "cin": "-log1p(-x)", "precip_1h": "log1p(x)"},
        "inputs": {c: {"mean": float(stats.mean[j]), "std": float(stats.std()[j])} for j, c in enumerate(names)},
        "targets": {},
    }
    for c, (f, _) in TARGET_TRANSFORMS.items():
        v = f(y[:, TARGET_NAMES.index(c)])
        out["targets"][c] = {"mean": float(v.mean()), "std": float(v.std(ddof=1))}
    write_json(METADATA_DIR / "transform_stats.json", out)
    log.info("wrote %s", METADATA_DIR / "transform_stats.json")


if __name__ == "__main__":
    main()
