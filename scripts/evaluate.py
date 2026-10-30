"""Score the best checkpoint on the test year against persistence and climatology.

Output: logs/eval_<name>.md

Usage
  python scripts/evaluate.py [--name baseline_cnn]
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader

from baselines import format_table, reference_forecasts, score
from common import LOG_DIR, METADATA_DIR, load_config, setup_logging
from hrrr_dataset import make_datasets
from model import BaselineCNN
from train import CKPT_DIR, predict, real_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline_cnn")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    log = setup_logging(f"evaluate_{args.name}")
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(CKPT_DIR / f"{args.name}.pt", map_location=device)
    model = BaselineCNN(in_channels=ckpt["in_channels"], grid=tuple(ckpt["grid"])).to(device)
    model.load_state_dict(ckpt["model"])
    ds = make_datasets(cfg)[args.split]
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    pos_weight = torch.tensor(ds.pos_weight or 1.0, device=device)
    loss, out, y = predict(model, loader, device, pos_weight, args.max_batches)
    results = {"CNN": real_metrics(ds, out, y)}

    period = cfg["splits"]["test"] if args.split == "test" else cfg["model_selection"]["val"]
    ref = cfg["splits"]["train"] if args.split == "test" else cfg["model_selection"]["fit"]
    samples = pd.read_parquet(METADATA_DIR / "samples.parquet")
    root = zarr.open_group(str(cfg["store_path"]), mode="r")
    truth, preds = reference_forecasts(root, samples, period, ref)
    n = len(out)
    for m, (p, rp) in preds.items():
        results[m] = score(truth[:n], p[:n], rp[:n])

    lines = [f"# {args.name} on {args.split} ({period[0]} .. {period[1]}, {n:,} samples)", "",
             f"Checkpoint: epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.4f}. Lower is better except AUC and F1.",
             "", *format_table(results), ""]
    (LOG_DIR / f"eval_{args.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    log.info("%s loss %.4f", args.split, loss)


if __name__ == "__main__":
    main()
