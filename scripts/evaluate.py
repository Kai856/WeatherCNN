"""Score the best checkpoint on the test year against persistence and climatology.

The CNN's rain probabilities are Platt-calibrated on val: p = sigmoid(a * logit + b), which undoes the
over-prediction caused by pos_weight. Rain F1 thresholds are also tuned on val, per method.

Output: logs/eval_<name>.md

Usage
  python scripts/evaluate.py [--name baseline_cnn]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import zarr
from torch import nn
from torch.utils.data import DataLoader

from baselines import best_threshold, format_table, reference_forecasts, score
from common import LOG_DIR, METADATA_DIR, load_config, setup_logging
from hrrr_dataset import make_datasets
from model import BaselineCNN
from train import CKPT_DIR, predict


def fit_platt(logit: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(a, b) minimizing log loss of sigmoid(a * logit + b) against y."""
    z, t = torch.as_tensor(logit, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
    ab = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([ab], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(ab[0] * z + ab[1], t)
        loss.backward()
        return loss

    opt.step(closure)
    a, b = ab.detach().tolist()
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline_cnn")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    log = setup_logging(f"evaluate_{args.name}")
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(CKPT_DIR / f"{args.name}.pt", map_location=device)
    kwargs = ckpt.get("model_kwargs") or {"in_channels": ckpt["in_channels"], "grid": tuple(ckpt["grid"])}
    model = BaselineCNN(**kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    datasets = make_datasets(cfg, prev_hours=ckpt.get("prev_hours", 0))
    samples = pd.read_parquet(METADATA_DIR / "samples.parquet")
    root = zarr.open_group(str(cfg["store_path"]), mode="r")
    ms, sp = cfg["model_selection"], cfg["splits"]
    periods = {"val": (ms["val"], ms["fit"]), "test": (sp["test"], sp["train"])}

    def forecasts(split):
        """{method: (truth, prediction, rain probability)} in real units for one split (CNN: rain logit)."""
        ds = datasets[split]
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=device.type == "cuda")
        pos_weight = torch.tensor(ckpt.get("pos_weight") or ds.pos_weight or 1.0, device=device)
        loss, out, y = predict(model, loader, device, pos_weight, args.max_batches)
        log.info("%s loss %.4f", split, loss)
        n = len(out)
        truth, refs = reference_forecasts(root, samples, *periods[split], eval_samples=ds.samples)
        res = {"CNN": (ds.unnormalize(y).numpy(), ds.unnormalize(out).numpy(), out[:, 6].numpy().astype(np.float64))}
        res.update({m: (truth[:n], p[:n], rp[:n]) for m, (p, rp) in refs.items()})
        return res

    def calibrate(res):
        t, p, logit = res["CNN"]
        res["CNN"] = (t, p, 1 / (1 + np.exp(-(a * logit + b))))
        return 1 / (1 + np.exp(-logit))

    val = forecasts("val")
    a, b = fit_platt(val["CNN"][2], val["CNN"][0][:, 6])
    calibrate(val)
    thresholds = {m: best_threshold(t[:, 6], rp) for m, (t, _, rp) in val.items()}
    final = val if args.split == "val" else forecasts(args.split)
    raw = calibrate(final) if final is not val else None
    results = {m: score(t, p, rp, thresholds[m]) for m, (t, p, rp) in final.items()}
    period, n = periods[args.split][0], len(final["CNN"][0])
    raw_brier = f"; uncalibrated CNN Brier {np.mean((raw - final['CNN'][0][:, 6]) ** 2):.3f}" if raw is not None else ""

    sel = f"val skill {ckpt['val_skill']:.4f}" if "val_skill" in ckpt else f"val loss {ckpt['val_loss']:.4f}"
    lines = [f"# {args.name} on {args.split} ({period[0]} .. {period[1]}, {n:,} samples)", "",
             f"Checkpoint: epoch {ckpt['epoch']}, {sel}. Lower is better except AUC and F1.",
             f"CNN rain probability Platt-calibrated on val: sigmoid({a:.3f} * logit + {b:.3f}){raw_brier}.",
             "Rain F1 uses each method's threshold tuned on val: "
             + ", ".join(f"{m} {t:.3f}" for m, t in thresholds.items()) + ".",
             "", *format_table(results), ""]
    (LOG_DIR / f"eval_{args.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
