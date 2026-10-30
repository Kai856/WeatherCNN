"""Automated quality checks (spec section 10). Writes logs/validation_report.md; exits 1 if any check fails.

Usage
  python scripts/validate_dataset.py [--sample-hours 48]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import zarr

from build_dataset import open_store, read_inputs, split_of
from common import (LOG_DIR, METADATA_DIR, du_bytes, haversine_km, load_config, load_grid_metadata, native_xy,
                    period_hours, read_json)

BUDGET_GB = 100.0
TARGET_RANGES = {"t2m": (220, 330), "rh2m": (0, 105), "u10": (-75, 75), "v10": (-75, 75), "gust": (0, 100),
                 "precip_1h": (0, 250), "rain_binary": (0, 1)}


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    @property
    def ok(self):
        return all(r[1] for r in self.rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-hours", type=int, default=48, help="processed hours to scan for input ranges")
    args = ap.parse_args()
    cfg = load_config()
    grid = load_grid_metadata()
    root, times = open_store(cfg)
    rep = Report()
    order = root.attrs["channel_order"]
    C = len(order)
    ny, nx = grid["coarse_grid"]["ny"], grid["coarse_grid"]["nx"]
    done = root["done"][:]
    dt = times[done]
    samples = pd.read_parquet(METADATA_DIR / "samples.parquet")
    prov_files = sorted((METADATA_DIR / "provenance").glob("*.parquet"))
    prov = pd.concat([pd.read_parquet(p) for p in prov_files], ignore_index=True) if prov_files else pd.DataFrame()
    prov = prov[prov.valid_time.isin(dt)] if len(prov) else prov

    # 1. spatial dimensions
    shapes = {a: root[a].shape for a in root.array_keys()}
    exp = {"inputs": (len(times), len(root["inputs"].attrs["channels"]), ny, nx) if "inputs" in root else None,
           "lat": (ny, nx), "lon": (ny, nx), "bbox_mask": (ny, nx),
           **{f"static_{s['name']}": (ny, nx) for s in cfg["static_fields"]}}
    if "inputs_f32" in root:
        exp["inputs_f32"] = (len(times), len(root["inputs_f32"].attrs["channels"]), ny, nx)
    bad = {a: (shapes.get(a), e) for a, e in exp.items() if e is not None and shapes.get(a) != e}
    rep.check("Arrays have expected spatial dimensions", not bad, f"{ny} x {nx}; mismatches: {bad or 'none'}")

    # 2. same grid
    grid_bad = int((~prov.grid_ok.fillna(False).astype(bool)).sum()) if len(prov) else -1
    rep.check("All variables use the same native grid", grid_bad == 0,
              f"{len(prov)} decoded messages checked against grid_metadata.json; {grid_bad} mismatches")

    # 3. timestamps aligned
    t_ok = times.equals(period_hours(cfg))
    vt_ok = bool((prov.grib_valid_time == prov.valid_time).all())
    il_ok = bool((prov.init_time + pd.to_timedelta(prov.lead_h, unit="h") == prov.valid_time).all())
    rep.check("Timestamps aligned", t_ok and vt_ok and il_ok,
              f"store time axis == hourly period: {t_ok}; GRIB valid time == record time: {vt_ok}; init + lead == valid: {il_ok}")

    # every channel of every processed hour has a provenance row
    per_hour = prov.groupby("valid_time").channel.nunique()
    cover = bool(len(per_hour) == done.sum() and (per_hour == C).all())
    rep.check("Provenance covers every channel of every processed hour", cover,
              f"{len(per_hour)} hours with provenance / {int(done.sum())} processed")

    # 4. 24 h offset
    lead = cfg["lead_hours"]
    off_ok = bool(((samples.target_time - samples.t) == pd.Timedelta(hours=lead)).all()
                  and ((samples.target_index - samples.t_index) == lead).all()
                  and (times[samples.t_index.values] == pd.DatetimeIndex(samples.t)).all())
    rep.check("Target offset is exactly +24 h", off_ok, f"{len(samples)} samples")

    # 5. precipitation interval
    ap_ = prov[prov.channel == "apcp1h"]
    p_ok = bool(len(ap_) and (ap_.accum_hours == 1).all() and (ap_.step_range == "0-1").all() and (ap_.lead_h == 1).all()
                and (ap_.init_time == ap_.valid_time - pd.Timedelta(hours=1)).all() and (ap_.step_type == "accum").all())
    rep.check("Precipitation interval is exactly 1 h ending at the valid time", p_ok,
              f"{len(ap_)} APCP messages: accum {sorted(ap_.accum_hours.unique().tolist())} h, step {sorted(ap_.step_range.unique().tolist())}")

    # 6. no input uses data from after its valid time
    future = int((prov.init_time > prov.valid_time).sum() + (prov.grib_valid_time != prov.valid_time).sum())
    rep.check("No input sample uses data from after its valid time", future == 0,
              f"inputs: init <= valid and GRIB valid == t for all messages ({future} violations); "
              "targets come from records valid at t+24 (precip from init t+23)")

    # 7. duplicates
    dup_t = int(times.duplicated().sum())
    dup_p = int(prov.duplicated(["valid_time", "channel"]).sum())
    dup_s = int(samples.t.duplicated().sum())
    rep.check("No duplicated timestamps", dup_t + dup_p + dup_s == 0,
              f"time axis {dup_t}, provenance {dup_p}, samples {dup_s}")

    # 8. missing values quantified
    valid = root["input_valid"][:][done]
    inv_frac = 1 - valid.mean(axis=0)
    worst = sorted(zip(order, inv_frac), key=lambda x: -x[1])[:3]
    rec_ok = root["record_ok"][:][done]
    rep.check("Missing values quantified", True,
              f"{int(done.sum())} processed hours, {int((~rec_ok).sum())} not fully valid "
              f"({100 * (1 - rec_ok.mean()):.2f}%); worst channels: {', '.join(f'{c} {100 * f:.2f}%' for c, f in worst)}; "
              f"unprocessed hours in period: {int((~done).sum())}")

    # 9. plausible ranges
    tg = root["target_fields"][:][done]
    names = root["target_fields"].attrs["targets"]
    msgs, ok_r = [], True
    for k, n in enumerate(names):
        col = tg[:, k][np.isfinite(tg[:, k])]
        lo, hi = TARGET_RANGES[n]
        good = bool(((col >= lo) & (col <= hi)).all())
        ok_r &= good
        msgs.append(f"{n} [{col.min():.2f}, {col.max():.2f}]" if col.size else f"{n} empty")
    rb_ok = bool(np.all((tg[:, 6] == 1) == (tg[:, 5] > cfg["rain_threshold_mm"])))
    pick = np.flatnonzero(done)
    pick = pick[np.linspace(0, len(pick) - 1, min(args.sample_hours, len(pick))).astype(int)] if len(pick) else pick
    chans = {c["name"]: c for c in cfg["channels"]}
    oob = []
    for t in pick:
        x = read_inputs(root, int(t), order)
        v = root["input_valid"][int(t)]
        for k, c in enumerate(order):
            if not v[k]:
                continue
            lo, hi = chans[c]["bounds"]
            if np.isnan(x[k]).any() or x[k].min() < lo or x[k].max() > hi:
                oob.append(f"{times[t]} {c}")
    rep.check("Temperature, humidity, wind, precipitation within plausible ranges", ok_r and rb_ok and not oob,
              f"targets: {'; '.join(msgs)}; rain_binary consistent: {rb_ok}; "
              f"inputs in {len(pick)} sampled hours within channel bounds and NaN-free: {not oob} {oob[:3]}")

    # 10. Tufts cell
    tu = grid["tufts"]
    x, y, proj = native_xy(grid["native_grid"])
    jj, ii = np.meshgrid(np.arange(tu["native_j"] - 5, tu["native_j"] + 6), np.arange(tu["native_i"] - 5, tu["native_i"] + 6),
                         indexing="ij")
    lon, lat = proj(x[ii], y[jj], inverse=True)
    d = haversine_km(tu["requested_lat"], tu["requested_lon"], lat, lon)
    bj, bi = np.unravel_index(np.argmin(d), d.shape)
    cell_ok = bool(jj[bj, bi] == tu["native_j"] and ii[bj, bi] == tu["native_i"] and d.min() < 2.13)
    rep.check("Nearest Tufts grid cell is correct", cell_ok,
              f"native (j={tu['native_j']}, i={tu['native_i']}) at ({tu['cell_lat']:.4f}, {tu['cell_lon']:.4f}), "
              f"{d.min():.3f} km from the requested point (brute-force search over 11x11 neighborhood)")

    # 11. splits
    sp = cfg["splits"]
    bounds = sorted((pd.Timestamp(a), pd.Timestamp(b), n) for n, (a, b) in sp.items())
    no_overlap = all(bounds[k][1] < bounds[k + 1][0] for k in range(len(bounds) - 1))
    same = bool((split_of(cfg, pd.DatetimeIndex(samples.target_time)) == samples.split.values).all()
                and (split_of(cfg, pd.DatetimeIndex(samples.t)) == samples.split.values).all())
    sets = {n: set(samples.t[samples.split == n]) for n in sp}
    disjoint = all(not (sets[a] & sets[b]) for a in sp for b in sp if a < b)
    rep.check("Train, validation, and test periods do not overlap", no_overlap and same and disjoint,
              f"period bounds ordered and disjoint: {no_overlap}; every input and target in its sample's split: {same}; "
              f"sample sets disjoint: {disjoint}; counts {samples.split.value_counts().to_dict()}")

    # 12. budget
    used = du_bytes(cfg["data_root_path"]) / 1e9
    pilot = METADATA_DIR / "pilot_summary.json"
    proj_gb = read_json(pilot)["projected_gb"]["total"] if pilot.exists() else None
    proj_txt = f"; projected 3-year total from pilot {proj_gb:.1f} GB" if proj_gb is not None else ""
    rep.check("Output stays within the 100 GB budget", used <= BUDGET_GB and (proj_gb is None or proj_gb <= BUDGET_GB),
              f"data root currently {used:.2f} GB{proj_txt}")

    lines = ["# Validation report", "", f"Store: {cfg['store_path']}", f"Processed hours: {int(done.sum())} / {len(times)}", "",
             "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n, ok, d in rep.rows]
    lines += ["", f"Overall: **{'PASS' if rep.ok else 'FAIL'}**"]
    (LOG_DIR / "validation_report.md").write_text("\n".join(lines) + "\n")
    sys.exit(0 if rep.ok else 1)


if __name__ == "__main__":
    main()
