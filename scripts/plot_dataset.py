"""Figures that summarize the processed dataset. Works on whatever part of the store is processed.

Outputs PNGs in plots/:
  1_input_maps.png         all 49 input channels for one hour
  2_static_fields.png      orography and land fraction
  3_tufts_timeseries.png   daily Tufts targets over the whole period, train/test shaded
  4_target_distributions.png  train vs test histograms of each target
  5_tufts_climatology.png  monthly and diurnal cycles at Tufts
  6_seasonal_maps.png      seasonal mean 2 m temperature and precipitable water (18Z)
  7_data_coverage.png      processed / fully valid hours per month and invalid fraction per channel

Usage
  python scripts/plot_dataset.py [--time 2025-07-15T18] [--map-stride-days 5]
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_dataset import open_store, read_inputs, split_of
from common import PROJECT_ROOT, TARGET_NAMES, load_config, setup_logging

OUT = PROJECT_ROOT / "plots"
SPLIT_COLORS = {"train": "tab:blue", "test": "tab:orange"}
CMAPS = {"t": "RdYlBu_r", "d": "BrBG", "r": "BrBG", "u": "RdBu_r", "v": "RdBu_r", "g": "viridis",
         "s": "viridis", "m": "viridis", "c": "magma", "p": "GnBu", "a": "Blues"}
SPECIAL = {"t2m": "RdYlBu_r", "skt": "RdYlBu_r", "cape": "magma", "cin": "magma", "dswrf": "inferno",
           "pwat": "GnBu", "apcp1h": "Blues", "gust": "viridis", "sp": "viridis", "mslp": "viridis"}


def cmap_for(name: str) -> str:
    return SPECIAL.get(name, CMAPS.get(name[0], "viridis"))


def draw_map(ax, lon, lat, field, land, tufts, name, unit, symmetric=False, vlim=None, cmap=None):
    finite = field[np.isfinite(field)]
    if vlim is None:
        lo, hi = (np.percentile(finite, [2, 98]) if finite.size else (0, 1))
        if symmetric:
            m = max(abs(lo), abs(hi))
            lo, hi = -m, m
    else:
        lo, hi = vlim
    if hi <= lo:
        hi = lo + 1
    im = ax.pcolormesh(lon, lat, field, cmap=cmap or cmap_for(name), vmin=lo, vmax=hi, shading="auto")
    ax.contour(lon, lat, land, levels=[0.5], colors="k", linewidths=0.4)
    ax.plot(tufts[1], tufts[0], marker="*", color="red", markersize=6, markeredgecolor="k", markeredgewidth=0.3)
    ax.set_title(f"{name} [{unit}]", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6)


def fig_input_maps(root, times, order, units, lon, lat, land, tufts, ok, when):
    t = int(np.argmin(np.abs((times - when).total_seconds())) if not ok.any() else
            np.flatnonzero(ok)[np.argmin(np.abs((times[ok] - when).total_seconds()))])
    x = read_inputs(root, t, order)
    fig, axes = plt.subplots(7, 7, figsize=(24, 20))
    for k, ax in enumerate(axes.flat):
        c = order[k]
        draw_map(ax, lon, lat, x[k], land, tufts, c, units[c], symmetric=c[0] in "uv")
    fig.suptitle(f"All 49 input channels at {times[t]:%Y-%m-%d %H:%M} UTC (6 km grid, red star = Tufts; "
                 "u/v are grid-relative)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "1_input_maps.png", dpi=110)
    plt.close(fig)


def fig_static(root, lon, lat, land, tufts):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    draw_map(axes[0], lon, lat, root["static_orography"][:], land, tufts, "orography", "m", vlim=(0, 1500),
             cmap="terrain")
    draw_map(axes[1], lon, lat, root["static_land"][:], land, tufts, "land", "fraction", vlim=(0, 1), cmap="Greens")
    fig.suptitle("Static fields (stored once)")
    fig.tight_layout()
    fig.savefig(OUT / "2_static_fields.png", dpi=120)
    plt.close(fig)


def target_frame(root, times, done):
    y = root["target_fields"][:]
    df = pd.DataFrame(y, index=times, columns=TARGET_NAMES)
    df = df[done]
    df["t2m"] -= 273.15
    df["wind10"] = np.hypot(df.u10, df.v10)
    return df


def fig_timeseries(df, cfg):
    daily = df.resample("D")
    cols = [("t2m", "2 m temperature [°C]"), ("rh2m", "2 m relative humidity [%]"),
            ("wind10", "10 m wind speed [m/s]"), ("gust", "gust [m/s]")]
    fig, axes = plt.subplots(len(cols) + 1, 1, figsize=(16, 13), sharex=True)
    for ax, (c, label) in zip(axes, cols):
        mean, lo, hi = daily[c].mean(), daily[c].min(), daily[c].max()
        ax.fill_between(mean.index, lo, hi, color="0.8", lw=0, label="daily min-max")
        ax.plot(mean.index, mean, lw=0.6, color="k", label="daily mean")
        ax.set_ylabel(label)
    p = daily["precip_1h"].sum()
    axes[-1].bar(p.index, p.values, width=1.0, color="tab:blue")
    axes[-1].set_ylabel("daily precip [mm]")
    for ax in axes:
        for name, (a, b) in cfg["splits"].items():
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=SPLIT_COLORS.get(name, "0.5"), alpha=0.07)
    axes[0].legend(loc="upper left", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SPLIT_COLORS[n], alpha=0.3) for n in cfg["splits"] if n in SPLIT_COLORS]
    axes[1].legend(handles, [n for n in cfg["splits"] if n in SPLIT_COLORS], loc="upper left", fontsize=8)
    fig.suptitle("HRRR conditions at the Tufts grid cell (targets), daily summaries")
    fig.tight_layout()
    fig.savefig(OUT / "3_tufts_timeseries.png", dpi=110)
    plt.close(fig)


def fig_distributions(df, cfg, threshold):
    split = pd.Series(split_of(cfg, df.index), index=df.index)
    cols = [("t2m", "°C"), ("rh2m", "%"), ("u10", "m/s, earth-relative"), ("v10", "m/s, earth-relative"),
            ("gust", "m/s"), ("precip_1h", "mm, hours with rain only")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (c, unit) in zip(axes.flat, cols):
        for name in cfg["splits"]:
            v = df.loc[split == name, c].dropna()
            if c == "precip_1h":
                v = v[v > 0.01]
            if len(v):
                ax.hist(v, bins=60, density=True, histtype="step", lw=1.4, color=SPLIT_COLORS.get(name),
                        label=f"{name} (n={len(v):,})")
        if c == "precip_1h":
            ax.set_yscale("log")
            ax.axvline(threshold, color="r", ls="--", lw=1, label=f"rain_binary threshold {threshold} mm")
        ax.set_title(f"{c} [{unit}]")
        ax.legend(fontsize=7)
    rates = {n: float((df.loc[split == n, "precip_1h"] > threshold).mean()) for n in cfg["splits"] if (split == n).any()}
    fig.suptitle("Target distributions, train vs test. rain_binary positive rate: " +
                 ", ".join(f"{n} {r:.2%}" for n, r in rates.items()))
    fig.tight_layout()
    fig.savefig(OUT / "4_target_distributions.png", dpi=110)
    plt.close(fig)


def fig_climatology(df, threshold):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    m = df.groupby(df.index.month)
    axes[0].plot(m.t2m.mean().index, m.t2m.mean(), "o-", color="tab:red", label="mean")
    axes[0].fill_between(m.t2m.mean().index, m.t2m.quantile(0.1), m.t2m.quantile(0.9), color="tab:red", alpha=0.2,
                         label="10-90%")
    axes[0].set(title="2 m temperature by month", xlabel="month", ylabel="°C", xticks=range(1, 13))
    axes[0].legend(fontsize=8)
    h = df.groupby(df.index.hour)
    ax = axes[1]
    ax.plot(h.t2m.mean().index, h.t2m.mean(), "o-", color="tab:red")
    ax.set(title="Diurnal cycle (UTC; Eastern = UTC-4/-5)", xlabel="hour UTC", ylabel="2 m temperature °C",
           xticks=range(0, 24, 3))
    ax2 = ax.twinx()
    ax2.plot(h.rh2m.mean().index, h.rh2m.mean(), "s-", color="tab:green", ms=3)
    ax2.set_ylabel("2 m RH %", color="tab:green")
    rain = df.precip_1h > threshold
    wet = df.precip_1h > 0.1
    months = range(1, 13)
    axes[2].bar(np.arange(1, 13) - 0.2, wet.groupby(df.index.month).mean().reindex(months) * 100, width=0.4,
                label="> 0.1 mm/h")
    axes[2].bar(np.arange(1, 13) + 0.2, rain.groupby(df.index.month).mean().reindex(months) * 100, width=0.4,
                label=f"> {threshold} mm/h (rain_binary)")
    axes[2].set(title="Hours with precipitation by month", xlabel="month", ylabel="% of hours", xticks=range(1, 13))
    axes[2].legend(fontsize=8)
    fig.suptitle("Tufts climatology from the dataset targets")
    fig.tight_layout()
    fig.savefig(OUT / "5_tufts_climatology.png", dpi=110)
    plt.close(fig)


def fig_seasonal_maps(root, times, ok, order, lon, lat, land, tufts, stride_days):
    idx = [t for t in np.flatnonzero(ok & (times.hour == 18)) if (times[t].dayofyear % stride_days) == 0]
    if not idx:
        idx = list(np.flatnonzero(ok & (times.hour == 18)))
    ci = {c: order.index(c) for c in ("t2m", "pwat")}
    seasons = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    acc = {s: {c: [] for c in ci} for s in seasons}
    for t in idx:
        x = read_inputs(root, int(t), order)
        s = next(k for k, months in seasons.items() if times[t].month in months)
        for c, k in ci.items():
            acc[s][c].append(x[k])
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    lims = {"t2m": (250, 310), "pwat": (0, 50)}
    for j, s in enumerate(seasons):
        for i, c in enumerate(ci):
            ax = axes[i, j]
            if acc[s][c]:
                mean = np.nanmean(np.stack(acc[s][c]), axis=0)
                draw_map(ax, lon, lat, mean, land, tufts, c, "K" if c == "t2m" else "kg m-2", vlim=lims[c])
                ax.set_title(f"{s} mean {c} 18Z (n={len(acc[s][c])})", fontsize=9)
            else:
                ax.set_axis_off()
                ax.set_title(f"{s}: no processed hours", fontsize=9)
    fig.suptitle(f"Seasonal means at 18 UTC (every {stride_days}th day)")
    fig.tight_layout()
    fig.savefig(OUT / "6_seasonal_maps.png", dpi=110)
    plt.close(fig)


def fig_coverage(root, times, done, ok, order):
    s = pd.DataFrame({"done": done, "ok": ok}, index=times).resample("MS").mean() * 100
    valid = root["input_valid"][:]
    inv = 100 * (1 - valid[done].mean(axis=0)) if done.any() else np.zeros(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    axes[0].bar(s.index, s.done, width=25, color="0.75", label="processed")
    axes[0].bar(s.index, s.ok, width=25, color="tab:green", label="fully valid")
    axes[0].set(ylabel="% of hours", title="Processed and fully valid hours per month", ylim=(0, 105))
    axes[0].legend(fontsize=8)
    axes[1].bar(range(len(order)), inv, color="tab:red")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(order, rotation=90, fontsize=7)
    axes[1].set(ylabel="% invalid", title="Invalid fraction per input channel (processed hours)",
                ylim=(0, max(1.0, float(inv.max()) * 1.2)))
    fig.suptitle(f"Data coverage: {int(done.sum()):,} processed, {int(ok.sum()):,} fully valid of {len(times):,} hours")
    fig.tight_layout()
    fig.savefig(OUT / "7_data_coverage.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", default="2025-07-15T18", help="hour for the all-channel maps (nearest valid hour)")
    ap.add_argument("--map-stride-days", type=int, default=5)
    args = ap.parse_args()
    log = setup_logging("plot_dataset")
    cfg = load_config()
    root, times = open_store(cfg)
    order = list(root.attrs["channel_order"])
    attrs = root["inputs"].attrs
    units = dict(zip(attrs["channels"], attrs["units"]))
    lon, lat = root["lon"][:], root["lat"][:]
    land = root["static_land"][:]
    tu = root.attrs["tufts"]
    tufts = (tu["cell_lat"], tu["cell_lon"])
    done = root["done"][:].astype(bool)
    ok = root["record_ok"][:].astype(bool)
    OUT.mkdir(exist_ok=True)

    log.info("input maps")
    fig_input_maps(root, times, order, units, lon, lat, land, tufts, ok, pd.Timestamp(args.time))
    fig_static(root, lon, lat, land, tufts)
    df = target_frame(root, times, done)
    log.info("target figures (%d hours)", len(df))
    fig_timeseries(df, cfg)
    fig_distributions(df, cfg, cfg["rain_threshold_mm"])
    fig_climatology(df, cfg["rain_threshold_mm"])
    log.info("seasonal maps")
    fig_seasonal_maps(root, times, ok, order, lon, lat, land, tufts, args.map_stride_days)
    fig_coverage(root, times, done, ok, order)
    log.info("wrote figures to %s", OUT)


if __name__ == "__main__":
    main()
