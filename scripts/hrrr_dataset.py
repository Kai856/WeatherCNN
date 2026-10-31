"""PyTorch Dataset: normalized HRRR input maps at t -> normalized Tufts targets at t+24 h.

x: (55, 223, 291) float32
   0-48   the 49 input channels; apcp1h, cape, cin log-transformed; all z-scored with train-only stats
   49-50  orography (z-scored), land (0/1)
   51-54  sin/cos hour of day, sin/cos day of year (constant maps, from the input time t)
   55-103 with prev_hours > 0: the 49 input channels at t - prev_hours, normalized the same way
y: (7,) float32  t2m, rh2m, u10, v10, gust (z-scored), precip_1h (log1p then z-scored), rain_binary (0/1)
   with delta=True: (13,) = the 6 continuous targets as z-scored 24 h changes from t, rain_binary, then the
   6 normalized values at t (needed to rebuild absolute predictions; use to_real)

Requires metadata/samples.parquet, normalization_stats.json and transform_stats.json.

Usage
  from hrrr_dataset import HRRRDataset
  ds = HRRRDataset("train")
  x, y = ds[0]
  y_real = ds.unnormalize(model_output)   # back to K, %, m/s, m/s, m/s, mm, probability-or-logit unchanged
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

from build_dataset import read_inputs
from common import METADATA_DIR, TARGET_NAMES, load_config, read_json
from transform_stats import TARGET_TRANSFORMS, TRANSFORMS

N_EXTRA = 6


class HRRRDataset(Dataset):
    def __init__(self, split: str, cfg: dict | None = None, start: str | None = None, end: str | None = None,
                 prev_hours: int = 0, delta: bool = False):
        """split: "train" or "test". start/end (inclusive) further restrict samples by input time t.
        prev_hours: also feed the input channels from t - prev_hours (samples without a valid one are dropped).
        delta: continuous targets become the 24 h change from the Tufts values at t (see to_real)."""
        self.cfg = cfg or load_config()
        self.prev_hours = prev_hours
        self.delta = delta
        self._root = None
        root = self.root
        norm = read_json(METADATA_DIR / "normalization_stats.json")
        tstats = read_json(METADATA_DIR / "transform_stats.json")

        # targets: 6 continuous (precip in log space) + rain_binary
        self.cont = TARGET_NAMES[:6]
        tsrc = {c: tstats["targets"].get(c, norm["targets"][c]) for c in self.cont}
        self.y_mean = np.array([tsrc[c]["mean"] for c in self.cont], dtype=np.float32)
        self.y_std = np.array([tsrc[c]["std"] for c in self.cont], dtype=np.float32)
        self.pos_weight = norm["rain_binary"]["pos_weight_neg_over_pos"]
        self.targets = root["target_fields"][:].astype(np.float32)

        samples = pd.read_parquet(METADATA_DIR / "samples.parquet")
        if delta:
            # change stats from the whole train split (same scope as the other normalization stats)
            tr = samples[samples.split == "train"]
            d = (self._norm_targets(self.targets[tr.target_index.values])
                 - self._norm_targets(self.targets[tr.t_index.values]))[:, :6]
            d = d[np.isfinite(d).all(axis=1)]
            self.d_mean, self.d_std = d.mean(axis=0).astype(np.float32), d.std(axis=0).astype(np.float32)
        keep = samples.split == split
        if delta:
            keep &= np.isfinite(self.targets[samples.t_index.values]).all(axis=1)
        if start is not None:
            keep &= samples.t >= pd.Timestamp(start)
        if end is not None:
            keep &= samples.t <= pd.Timestamp(end)
        if prev_hours:
            input_ok = root["done"][:] & root["input_valid"][:].all(axis=1)
            prev = samples.t_index.to_numpy() - prev_hours
            keep &= (prev >= 0) & input_ok[np.clip(prev, 0, None)]
        self.samples = samples[keep].reset_index(drop=True)
        self.order = list(root.attrs["channel_order"])
        times = pd.to_datetime(root["time"][:], unit="h", origin="unix")
        self.times = pd.DatetimeIndex(times)

        # input normalization (transformed channels use transform_stats)
        src = {c: tstats["inputs"].get(c, norm["inputs"][c]) for c in self.order}
        self.in_mean = np.array([src[c]["mean"] for c in self.order], dtype=np.float32)[:, None, None]
        self.in_std = np.array([src[c]["std"] for c in self.order], dtype=np.float32)[:, None, None]
        self.tf_idx = {self.order.index(c): f for c, (f, _) in TRANSFORMS.items()}

        # static channels, computed once
        st = norm["input_static"]
        oro = (root["static_orography"][:] - st["orography"]["mean"]) / st["orography"]["std"]
        land = root["static_land"][:]
        self.static = np.stack([oro, land]).astype(np.float32)
        self.shape = self.static.shape[1:]
        self.n_channels = len(self.order) * (2 if prev_hours else 1) + N_EXTRA

    @property
    def root(self):
        # opened lazily so each DataLoader worker gets its own handle
        if self._root is None:
            self._root = zarr.open_group(str(self.cfg["store_path"]), mode="r")
        return self._root

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_root"] = None
        return state

    def __len__(self) -> int:
        return len(self.samples)

    def _inputs(self, t_index: int) -> np.ndarray:
        x = read_inputs(self.root, t_index, self.order)
        for k, f in self.tf_idx.items():
            x[k] = f(x[k])
        return (x - self.in_mean) / self.in_std

    def __getitem__(self, i: int):
        s = self.samples.iloc[i]
        x = self._inputs(int(s.t_index))

        t = self.times[int(s.t_index)]
        h = 2 * np.pi * t.hour / 24
        d = 2 * np.pi * (t.dayofyear - 1) / 365.25
        time_maps = np.array([np.sin(h), np.cos(h), np.sin(d), np.cos(d)], dtype=np.float32)[:, None, None]
        parts = [x, self.static, np.broadcast_to(time_maps, (4, *self.shape))]
        if self.prev_hours:
            parts.append(self._inputs(int(s.t_index) - self.prev_hours))
        x = np.concatenate(parts).astype(np.float32)

        y = self._norm_targets(self.targets[int(s.target_index)])
        if self.delta:
            base = self._norm_targets(self.targets[int(s.t_index)])[:6]
            y[:6] = (y[:6] - base - self.d_mean) / self.d_std
            y = np.concatenate([y, base])  # (13,): model targets, then the normalized values at t
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))

    def _norm_targets(self, raw: np.ndarray) -> np.ndarray:
        """Real (..., 7) target values -> normalized (precip log-transformed; rain_binary unchanged)."""
        y = np.array(raw, dtype=np.float32)
        for c, (f, _) in TARGET_TRANSFORMS.items():
            k = TARGET_NAMES.index(c)
            y[..., k] = f(y[..., k])
        y[..., :6] = (y[..., :6] - self.y_mean) / self.y_std
        return y

    def to_real(self, v, y) -> np.ndarray:
        """Model-space (N, 7) values (outputs or y[:, :7]) -> real units, using the batch targets y for the
        values at t when delta. The rain column is returned unchanged (logit or 0/1)."""
        v = (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)).astype(np.float32)
        y = (y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)).astype(np.float32)
        out = v[..., :7].copy()
        if self.delta:
            out[..., :6] = out[..., :6] * self.d_std + self.d_mean + y[..., 7:13]
        return self.unnormalize(out)

    def unnormalize(self, y):
        """Normalized (..., 7) predictions -> real units. The rain_binary column is returned unchanged."""
        is_torch = isinstance(y, torch.Tensor)
        out = y.detach().cpu().numpy().astype(np.float32) if is_torch else np.array(y, dtype=np.float32)
        out[..., :6] = out[..., :6] * self.y_std + self.y_mean
        for c, (_, inv) in TARGET_TRANSFORMS.items():
            k = TARGET_NAMES.index(c)
            out[..., k] = inv(out[..., k])
        return torch.from_numpy(out) if is_torch else out


def make_datasets(cfg: dict | None = None, prev_hours: int = 0, delta: bool = False) -> dict[str, HRRRDataset]:
    """fit / val (last training year) / test datasets."""
    cfg = cfg or load_config()
    ms = cfg["model_selection"]
    kw = {"prev_hours": prev_hours, "delta": delta}
    return {"fit": HRRRDataset("train", cfg, *ms["fit"], **kw),
            "val": HRRRDataset("train", cfg, *ms["val"], **kw),
            "test": HRRRDataset("test", cfg, **kw)}
