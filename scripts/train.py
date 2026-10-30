"""Train the baseline CNN on the fit period, select on val (last training year).

Loss = MSE on the 6 normalized continuous targets + BCEWithLogits(pos_weight) on rain_binary.
Each epoch uses every k-th fit hour (random offset) because neighboring hours are nearly identical.

Outputs
  checkpoints/<name>.pt          best model by val loss
  logs/train_<name>.csv          per-epoch losses, val metrics in real units, learning rate

Usage
  python scripts/train.py [--epochs 30] [--batch-size 32] [--workers 8]
  python scripts/train.py --epochs 1 --max-batches 5 --workers 0     # smoke test
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from baselines import score
from common import LOG_DIR, PROJECT_ROOT, setup_logging
from hrrr_dataset import make_datasets
from model import BaselineCNN, n_params

CKPT_DIR = PROJECT_ROOT / "checkpoints"


def loss_fn(out: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    mse = nn.functional.mse_loss(out[:, :6], y[:, :6])
    bce = nn.functional.binary_cross_entropy_with_logits(out[:, 6], y[:, 6], pos_weight=pos_weight)
    return mse + bce


def autocast(device: torch.device):
    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


@torch.no_grad()
def predict(model, loader, device, pos_weight, max_batches=None):
    """Returns (mean loss, normalized outputs (N, 7), normalized targets (N, 7))."""
    model.eval()
    outs, ys, total, n = [], [], 0.0, 0
    for b, (x, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device):
            out = model(x)
        out = out.float()
        total += loss_fn(out, y, pos_weight).item() * len(x)
        n += len(x)
        outs.append(out.cpu())
        ys.append(y.cpu())
    return total / max(n, 1), torch.cat(outs), torch.cat(ys)


def real_metrics(ds, out: torch.Tensor, y: torch.Tensor) -> dict:
    pred = ds.unnormalize(out).numpy()
    truth = ds.unnormalize(y).numpy()
    return score(truth, pred, torch.sigmoid(out[:, 6]).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline_cnn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=None, help="rain BCE pos_weight (default: neg/pos ratio)")
    ap.add_argument("--hidden", type=int, default=256, help="dense head width")
    ap.add_argument("--pool", type=int, nargs=2, default=None, metavar=("H", "W"),
                    help="average-pool the final feature map to H x W before flattening")
    ap.add_argument("--stride", type=int, default=3, help="use every k-th fit hour per epoch")
    ap.add_argument("--val-stride", type=int, default=3, help="every k-th val hour during training")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=None, help="limit batches per epoch (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    log = setup_logging(f"train_{args.name}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = make_datasets()
    fit, val = ds["fit"], ds["val"]
    val_sub = Subset(val, range(0, len(val), args.val_stride))
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_sub, shuffle=False, **loader_kw)
    pos_weight = torch.tensor(args.pos_weight or fit.pos_weight or 1.0, device=device)

    model_kwargs = {"in_channels": fit.n_channels, "grid": tuple(fit.shape), "hidden": args.hidden, "pool": args.pool}
    model = BaselineCNN(**model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * (args.max_batches or -(-len(fit) // (args.stride * args.batch_size)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    use_fp16 = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    log.info("device %s, %s params, fit %d / val %d samples (strides %d / %d), pos_weight %.1f",
             device, f"{n_params(model):,}", len(fit), len(val), args.stride, args.val_stride, pos_weight.item())

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_path = CKPT_DIR / f"{args.name}.pt"
    history, best, bad = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        idx = np.arange(rng.integers(args.stride), len(fit), args.stride)
        rng.shuffle(idx)
        train_loader = DataLoader(Subset(fit, idx.tolist()), shuffle=False, **{**loader_kw, "persistent_workers": False})
        model.train()
        total, n = 0.0, 0
        for b, (x, y) in enumerate(train_loader):
            if args.max_batches and b >= args.max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device):
                out = model(x)
            loss = loss_fn(out.float(), y, pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(x)
            n += len(x)
        train_loss = total / max(n, 1)

        val_loss, vout, vy = predict(model, val_loader, device, pos_weight, args.max_batches)
        metrics = real_metrics(val, vout, vy)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": sched.get_last_lr()[0],
               "seconds": time.time() - t0, **{f"val {k}": v for k, v in metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(LOG_DIR / f"train_{args.name}.csv", index=False)
        log.info("epoch %d: train %.4f val %.4f | t2m RMSE %.2f K, gust RMSE %.2f, rain AUC %.3f | %.0fs",
                 epoch, train_loss, val_loss, metrics["t2m RMSE"], metrics["gust RMSE"], metrics["rain AUC"],
                 row["seconds"])

        if val_loss < best:
            best, bad = val_loss, 0
            torch.save({"model": model.state_dict(), "model_kwargs": model_kwargs, "pos_weight": pos_weight.item(),
                        "epoch": epoch, "val_loss": val_loss, "val_metrics": metrics, "args": vars(args)}, ckpt_path)
            log.info("  saved %s", ckpt_path)
        else:
            bad += 1
            if bad >= args.patience:
                log.info("early stopping: no val improvement for %d epochs", args.patience)
                break
    log.info("best val loss %.4f", best)


if __name__ == "__main__":
    main()
