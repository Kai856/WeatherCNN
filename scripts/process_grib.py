"""Stages C/D: decode, check, crop, coarsen, quantize, and write HRRR fields into the Zarr store.

One record per valid hour v. Every field in a record is valid at v (checked from GRIB metadata):
  inputs[v]        49 channels on the 6 km grid (int16 + per-channel scale/offset; float32 for listed channels)
  target_fields[v] 7 values at the native 3 km cell nearest Tufts, valid at v
The dataset builder later pairs inputs[t] with target_fields[t + 24 h].

Resumable: the store is pre-allocated for the full period; `done[v]` is set only after all of a record's
arrays are written, and hours already marked done are skipped. Raw GRIB is deleted after a successful write.

Usage
  python scripts/process_grib.py --init-store
  python scripts/process_grib.py --start 2025-07-14T00 --end 2025-07-20T23 [--compare-float32]
  python scripts/process_grib.py --month-index 0      # SLURM array helper: month k of the period
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec
from zarr.codecs.numcodecs import Delta

from common import (APCP_1H_DESC, GRID_KEYS, INT16_MISSING, METADATA_DIR, LOG_DIR, TARGET_NAMES, TARGET_UNITS,
                    coarsen, crop, decode_message, grid_signature, grid_to_earth, hourly_range,
                    load_config, load_grid_metadata, make_session, period_hours, quantize, setup_logging,
                    time_index)
from download_hrrr import download_hour

log = setup_logging("process_grib")
PROV_DIR = METADATA_DIR / "provenance"
TARGET_SOURCE = {"t2m": "t2m", "rh2m": "rh2m", "u10": "u10", "v10": "v10", "gust": "gust", "precip_1h": "apcp1h"}


# --------------------------------------------------------------------------- store

def compressor(cfg):
    c = cfg["storage"]["compressor"]
    return BloscCodec(cname=c["cname"], clevel=c["clevel"], shuffle=c["shuffle"])


def channel_layout(cfg: dict, all_float32: bool = False) -> dict:
    f32 = set(cfg["storage"]["float32_channels"])
    names = [c["name"] for c in cfg["channels"]]
    if all_float32:
        f32 = set(names)
    i16 = [n for n in names if n not in f32]
    f32l = [n for n in names if n in f32]
    loc = {n: ("inputs", i16.index(n)) for n in i16}
    loc.update({n: ("inputs_f32", f32l.index(n)) for n in f32l})
    return {"int16": i16, "float32": f32l, "location": loc, "order": names}


def init_store(cfg: dict, grid: dict, path: Path, all_float32: bool = False, with_static: bool = True) -> None:
    if path.exists():
        log.info("store exists: %s", path)
        return
    hours = period_hours(cfg)
    T, C = len(hours), len(cfg["channels"])
    cg = grid["coarse_grid"]
    ny, nx = cg["ny"], cg["nx"]
    lay = channel_layout(cfg, all_float32)
    comp = [compressor(cfg)]
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w-")

    def arr(name, shape, chunks, dtype, fill, dims, compressors="auto", attrs=None, filters="auto"):
        a = root.create_array(name, shape=shape, chunks=chunks, dtype=dtype, fill_value=fill,
                              dimension_names=dims, compressors=compressors, filters=filters)
        if attrs:
            a.attrs.update(attrs)
        return a

    t = arr("time", (T,), (T,), "int64", 0, ["time"],
            attrs={"units": "hours since 1970-01-01 00:00:00", "calendar": "proleptic_gregorian",
                   "long_name": "valid time (UTC)"})
    t[:] = ((hours - pd.Timestamp("1970-01-01")) / pd.Timedelta(hours=1)).astype("int64")
    arr("y", (ny,), (ny,), "float64", 0, ["y"], attrs={"units": "m", "long_name": "projected y of 6 km cell center"})[:] = cg["y_m"]
    arr("x", (nx,), (nx,), "float64", 0, ["x"], attrs={"units": "m", "long_name": "projected x of 6 km cell center"})[:] = cg["x_m"]

    from common import native_xy
    _, _, proj = native_xy(grid["native_grid"])
    XC, YC = np.meshgrid(cg["x_m"], cg["y_m"])
    lon, lat = proj(XC, YC, inverse=True)
    arr("lat", (ny, nx), (ny, nx), "float32", 0, ["y", "x"], attrs={"units": "degrees_north"})[:] = lat
    arr("lon", (ny, nx), (ny, nx), "float32", 0, ["y", "x"], attrs={"units": "degrees_east"})[:] = lon
    bb = cfg["bbox"]
    inb = (lat >= bb["south"]) & (lat <= bb["north"]) & (lon >= bb["west"]) & (lon <= bb["east"])
    arr("bbox_mask", (ny, nx), (ny, nx), "bool", False, ["y", "x"],
        attrs={"long_name": "6 km cell center inside the requested lat/lon bbox"})[:] = inb

    chans = {c["name"]: c for c in cfg["channels"]}
    for aname, names, dtype, fill in (("inputs", lay["int16"], "int16", INT16_MISSING),
                                      ("inputs_f32", lay["float32"], "float32", np.nan)):
        if not names:
            continue
        filters = [Delta(dtype="int16")] if dtype == "int16" and cfg["storage"].get("delta_filter") else "auto"
        a = arr(aname, (T, len(names), ny, nx), (1, len(names), ny, nx), dtype, fill,
                ["time", "channel_" + aname, "y", "x"], compressors=comp, filters=filters)
        a.attrs.update({
            "step_rule": [chans[n]["step_rule"] for n in names] if dtype == "int16" else None,
            "q_src_max": [chans[n].get("q_src_max") for n in names],
            "channels": names,
            "units": [chans[n]["units"] for n in names],
            "grib_var": [chans[n]["var"] for n in names],
            "grib_level": [chans[n]["level"] for n in names],
            "source": [chans[n]["source"] for n in names],
            "scale": [chans[n]["scale"] for n in names] if dtype == "int16" else None,
            "offset": [chans[n]["offset"] for n in names] if dtype == "int16" else None,
            "bounds": [chans[n]["bounds"] for n in names],
            "missing_value": INT16_MISSING if dtype == "int16" else "NaN",
            "decode": "value = stored * scale[c] + offset[c]; stored == missing_value -> NaN" if dtype == "int16" else "none",
        })
    arr("input_valid", (T, C), (24, C), "bool", False, ["time", "channel"],
        attrs={"channels": lay["order"], "long_name": "channel decoded and passed all checks"})
    arr("target_fields", (T, len(TARGET_NAMES)), (24, len(TARGET_NAMES)), "float32", np.nan, ["time", "target"],
        attrs={"targets": TARGET_NAMES, "units": TARGET_UNITS,
               "long_name": "HRRR values at the native cell nearest Tufts, valid at time[t] (not shifted)"})
    arr("done", (T,), (24,), "bool", False, ["time"], attrs={"long_name": "record processed (resume marker)"})
    arr("record_ok", (T,), (24,), "bool", False, ["time"], attrs={"long_name": "all channels and targets valid"})
    for s in cfg["static_fields"]:
        arr(f"static_{s['name']}", (ny, nx), (ny, nx), "float32", np.nan, ["y", "x"],
            attrs={"units": s["units"], "grib": f"{s['var']}:{s['level']}", "method": "6 km block mean"})

    root.attrs.update({
        "title": "HRRR Northeast 6 km hourly analysis fields for 24 h point forecasting at Tufts",
        "source": "NOAA HRRR via s3://noaa-hrrr-bdp-pds (wrfsfcf00, wrfprsf00, wrfsfcf01)",
        "task": "inputs at t (HRRR analysis valid t) -> targets at t+24h (HRRR analysis valid t+24h; precip = HRRR 1 h forecast "
                "from init t+23). The model predicts future HRRR analysis conditions, not station observations.",
        "channel_order": lay["order"],
        "channel_location": {k: list(v) for k, v in lay["location"].items()},
        "grid": {"crs": grid["native_grid"]["proj4"], "native": grid["native_grid"], "crop_window_native": grid["crop_window_native"],
                 "coarsen_factor": grid["coarsen_factor"]},
        "tufts": grid["tufts"],
        "unit_conversions": {
            "inputs": "none: all inputs in GRIB SI units (K, Pa, m s-1, %, J kg-1, kg m-2, W m-2, gpm)",
            "precip_1h": "APCP kg m-2 -> mm (1 kg m-2 of water = 1 mm, numerically identical)",
            "u10/v10 targets": "rotated from HRRR grid-relative to earth-relative using the Lambert rotation at the Tufts cell",
            "input winds": "kept grid-relative (HRRR native), consistent across the grid",
            "rain_binary": f"1 if precip_1h > {cfg['rain_threshold_mm']} mm else 0 (NaN if precip_1h missing)",
        },
        "wind_components": "inputs grid-relative; targets earth-relative",
    })
    zarr.consolidate_metadata(str(path))
    log.info("initialized store %s (T=%d, C=%d, %dx%d)", path, T, C, ny, nx)

    if with_static:
        write_static(cfg, grid, path)


def write_static(cfg: dict, grid: dict, path: Path) -> None:
    session = make_session(cfg["download"]["retries"])
    v = pd.Timestamp(cfg["period"]["start"])
    d = download_hour(cfg, session, v, include_static=True, inputs=False,
                      out_dir=cfg["tmp_path"] / f"static_{v:%Y%m%d%H}")
    root = zarr.open_group(str(path), mode="r+")
    w, f = grid["crop_window_native"], grid["coarsen_factor"]
    for s in cfg["static_fields"]:
        vals, meta = decode_message((d / f"{s['name']}.grib2").read_bytes())
        check_grid(meta, grid)
        root[f"static_{s['name']}"][:] = coarsen(crop(vals, w), f)
        tj, ti = grid["tufts"]["native_j"], grid["tufts"]["native_i"]
        root[f"static_{s['name']}"].attrs.update({"source_valid_time": str(meta["valid_time"]),
                                                  "native_value_at_tufts": float(vals[tj, ti])})
    shutil.rmtree(d)
    zarr.consolidate_metadata(str(path))
    log.info("wrote static fields")


# --------------------------------------------------------------------------- per-hour processing

class CheckError(Exception):
    pass


def check_grid(meta: dict, grid: dict) -> None:
    sig = grid_signature(meta)
    ref = {k: grid["native_grid"][k] for k in GRID_KEYS}
    if sig != ref:
        bad = {k: (sig[k], ref[k]) for k in GRID_KEYS if sig[k] != ref[k]}
        raise CheckError(f"grid mismatch {bad}")


def check_times(meta: dict, item: dict, v: pd.Timestamp) -> float | None:
    if meta["valid_time"] != v:
        raise CheckError(f"valid time {meta['valid_time']} != {v}")
    if meta["init_time"] != pd.Timestamp(item["init_time"]):
        raise CheckError(f"init time {meta['init_time']} != {item['init_time']}")
    if item["idx_desc"] == APCP_1H_DESC:
        if meta["stepType"] != "accum" or (meta["startStep"], meta["endStep"]) != (0, 1):
            raise CheckError(f"APCP accumulation {meta['stepType']} {meta['stepRange']} is not 0-1 h")
        return float(meta["endStep"] - meta["startStep"])
    if meta["stepType"] != "instant" or str(meta["stepRange"]) != "0":
        raise CheckError(f"expected instantaneous analysis, got {meta['stepType']} {meta['stepRange']}")
    return None


def process_hour(cfg: dict, grid: dict, v: pd.Timestamp, d: Path) -> dict:
    man = json.loads((d / "manifest.json").read_text())
    items = {i["name"]: i for i in man["items"]}
    w, f = grid["crop_window_native"], grid["coarsen_factor"]
    tj, ti = grid["tufts"]["native_j"], grid["tufts"]["native_i"]
    north = grid["tufts"]["north_vector_grid_xy"]
    C = len(cfg["channels"])
    ny, nx = grid["coarse_grid"]["ny"], grid["coarse_grid"]["nx"]
    fields = np.full((C, ny, nx), np.nan, dtype=np.float32)
    valid = np.zeros(C, dtype=bool)
    at_tufts: dict[str, float] = {}
    prov, failures = [], []
    for m in man["missing"]:
        failures.append({"valid_time": v, "channel": m["name"], "reason": f"missing: {m['reason']}"})

    for k, ch in enumerate(cfg["channels"]):
        it = items.get(ch["name"])
        if it is None:
            continue
        row = {"valid_time": v, "channel": ch["name"], "variable": ch["var"], "level": ch["level"],
               "source_file": it["source_file"], "product": it["product"], "idx_desc": it["idx_desc"],
               "n_idx_matches": it["n_matches"], "message_bytes": it.get("bytes"), "used_for": "input"}
        if ch["name"] in TARGET_SOURCE.values():
            row["used_for"] = "input+target"
        try:
            vals, meta = decode_message((d / f"{ch['name']}.grib2").read_bytes())
            row.update({"init_time": meta["init_time"], "lead_h": int(meta["endStep"]),
                        "grib_valid_time": meta["valid_time"], "step_range": str(meta["stepRange"]),
                        "step_type": meta["stepType"], "binary_scale_E": meta["binaryScaleFactor"],
                        "decimal_scale_D": meta["decimalScaleFactor"], "bits_per_value": meta["bitsPerValue"],
                        "q_src": meta["q_src"], "packing": meta["packingType"]})
            check_grid(meta, grid)
            row["grid_ok"] = True
            row["accum_hours"] = check_times(meta, it, v)
            if ch["var"] in ("UGRD", "VGRD") and meta["uvRelativeToGrid"] != 1:
                raise CheckError("wind not grid-relative")
            native = crop(vals, w)
            lo, hi = ch["bounds"]
            nan_frac = float(np.isnan(native).mean())
            oob = int(np.count_nonzero((native < lo) | (native > hi)))
            row.update({"nan_frac": nan_frac, "out_of_bounds": oob,
                        "min": float(np.nanmin(native)), "max": float(np.nanmax(native))})
            if oob:
                raise CheckError(f"{oob} values outside bounds [{lo}, {hi}] (min {row['min']}, max {row['max']})")
            if nan_frac > 0:
                raise CheckError(f"{nan_frac:.4f} of crop is missing")
            fields[k] = coarsen(native, f)
            valid[k] = True
            at_tufts[ch["name"]] = float(vals[tj, ti])
            row["ok"] = True
        except Exception as e:  # noqa: BLE001 - record and continue; one bad channel must not stop the run
            row["ok"] = False
            row["error"] = str(e)
            failures.append({"valid_time": v, "channel": ch["name"], "reason": str(e)})
        prov.append(row)

    targets = np.full(len(TARGET_NAMES), np.nan, dtype=np.float32)
    get = lambda n: at_tufts.get(n, np.nan)  # noqa: E731
    u_e, v_e = grid_to_earth(get("u10"), get("v10"), north)
    targets[:] = [get("t2m"), get("rh2m"), u_e, v_e, get("gust"), get("apcp1h"), np.nan]
    if np.isfinite(targets[5]):
        targets[6] = float(targets[5] > cfg["rain_threshold_mm"])
    return {"valid_time": v, "fields": fields, "valid": valid, "targets": targets, "prov": prov, "failures": failures,
            "download_bytes": man.get("total_bytes", 0)}


# --------------------------------------------------------------------------- writing

class StoreWriter:
    def __init__(self, cfg, path: Path, compare_path: Path | None = None):
        self.cfg = cfg
        self.root = zarr.open_group(str(path), mode="r+")
        self.lay = {k: v for k, v in self.root.attrs["channel_location"].items()}
        self.chans = cfg["channels"]
        self.compare = zarr.open_group(str(compare_path), mode="r+") if compare_path else None
        self.i16_idx = [k for k, c in enumerate(self.chans) if self.lay[c["name"]][0] == "inputs"]
        self.f32_idx = [k for k, c in enumerate(self.chans) if self.lay[c["name"]][0] == "inputs_f32"]
        self.scale = np.array([self.chans[k]["scale"] for k in self.i16_idx], dtype=np.float32)
        self.offset = np.array([self.chans[k]["offset"] for k in self.i16_idx], dtype=np.float32)
        self.scale64 = np.array([self.chans[k]["scale"] for k in self.i16_idx], dtype=np.float64)
        self.offset64 = np.array([self.chans[k]["offset"] for k in self.i16_idx], dtype=np.float64)

    def write(self, t: int, res: dict) -> list[dict]:
        qrows = []
        if self.i16_idx:
            sub = res["fields"][self.i16_idx]
            q = np.empty(sub.shape, dtype=np.int16)
            for j, k in enumerate(self.i16_idx):
                q[j], clipped = quantize(sub[j], self.scale[j], self.offset[j])
                if clipped:
                    res["valid"][k] = False
                    res["failures"].append({"valid_time": res["valid_time"], "channel": self.chans[k]["name"], "reason": f"{clipped} values clipped"})
            self.root["inputs"][t] = q
            if self.compare is not None:
                deq = q.astype(np.float64) * self.scale64[:, None, None] + self.offset64[:, None, None]
                deq[q == INT16_MISSING] = np.nan
                err = np.abs(deq - sub.astype(np.float64))
                for j, k in enumerate(self.i16_idx):
                    e = err[j][np.isfinite(err[j])]
                    qrows.append({"channel": self.chans[k]["name"], "max_abs_err": float(e.max()) if e.size else np.nan,
                                  "rmse": float(np.sqrt((e.astype(np.float64) ** 2).mean())) if e.size else np.nan,
                                  "q16": float(self.scale[j])})
        if self.f32_idx:
            self.root["inputs_f32"][t] = res["fields"][self.f32_idx]
        if self.compare is not None:
            self.compare["inputs_f32"][t] = res["fields"]
        self.root["input_valid"][t] = res["valid"]
        self.root["target_fields"][t] = res["targets"]
        self.root["record_ok"][t] = bool(res["valid"].all() and np.isfinite(res["targets"]).all())
        self.root["done"][t] = True
        return qrows


# --------------------------------------------------------------------------- driver

def flush_provenance(rows: list[dict]) -> None:
    if not rows:
        return
    PROV_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for day, g in df.groupby(df.valid_time.dt.strftime("%Y%m%d")):
        p = PROV_DIR / f"{day}.parquet"
        if p.exists():
            old = pd.read_parquet(p)
            g = pd.concat([old[~old.valid_time.isin(g.valid_time.unique())], g], ignore_index=True)
        g = g.sort_values(["valid_time", "channel"])
        tmp = p.with_suffix(".tmp")
        g.to_parquet(tmp, index=False)
        tmp.replace(p)


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if "valid_time" in df:
        df["valid_time"] = pd.to_datetime(df.valid_time).dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def month_range(cfg: dict, k: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(cfg["period"]["start"]).to_period("M").to_timestamp() + pd.DateOffset(months=k)
    end = min(start + pd.DateOffset(months=1) - pd.Timedelta(hours=1), pd.Timestamp(cfg["period"]["end"]))
    return max(start, pd.Timestamp(cfg["period"]["start"])), end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-store", action="store_true")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--month-index", type=int, help="process calendar month k of the period (SLURM array)")
    ap.add_argument("--compare-float32", action="store_true",
                    help="also write a float32 copy and log int16 quantization errors (pilot only)")
    ap.add_argument("--keep-grib", action="store_true")
    ap.add_argument("--ignore-gate", action="store_true", help="allow >7-day ranges without a passing pilot report")
    args = ap.parse_args()

    cfg = load_config()
    grid = load_grid_metadata()
    store = cfg["store_path"]
    if args.init_store:
        init_store(cfg, grid, store)
        return
    if not store.exists():
        raise SystemExit(f"{store} does not exist: run with --init-store first")
    compare_path = None
    if args.compare_float32:
        compare_path = store.with_name(store.stem + "_f32compare.zarr")
        init_store(cfg, grid, compare_path, all_float32=True, with_static=False)

    if args.month_index is not None:
        start, end = month_range(cfg, args.month_index)
    else:
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    pilot = METADATA_DIR / "pilot_summary.json"
    if end - start > pd.Timedelta(days=7) and not args.ignore_gate:
        if not pilot.exists() or not json.loads(pilot.read_text()).get("gate_pass"):
            raise SystemExit("refusing to process more than 7 days: the pilot gate has not passed "
                             "(run the pilot and scripts/pilot_report.py first, or pass --ignore-gate)")
    if start.hour != 0 or end.hour != 23:
        log.warning("range %s..%s is not day-aligned; do not run another job on the same days concurrently", start, end)

    writer = StoreWriter(cfg, store, compare_path)
    done = writer.root["done"][:]
    todo = [v for v in hourly_range(start, end) if not done[time_index(cfg, v)]]
    log.info("processing %s..%s: %d hours to do (%d already done)", start, end, len(todo),
             len(hourly_range(start, end)) - len(todo))
    session = make_session(cfg["download"]["retries"])
    tag = f"{start:%Y%m%d%H}_{end:%Y%m%d%H}"
    fail_csv = LOG_DIR / "process_failures.csv"
    qcsv = LOG_DIR / f"quantization_check_{tag}.csv"
    prov_buf, t0, nbytes = [], time.time(), 0

    with ThreadPoolExecutor(cfg["download"]["hours_in_flight"]) as ex:
        futures = {}
        queue = list(todo)

        def submit_next():
            if queue:
                v = queue.pop(0)
                futures[v] = ex.submit(download_hour, cfg, session, v)

        for _ in range(cfg["download"]["hours_in_flight"]):
            submit_next()
        for n, v in enumerate(todo):
            fut = futures.pop(v)
            submit_next()
            try:
                d = fut.result()
                res = process_hour(cfg, grid, v, d)
                qrows = writer.write(time_index(cfg, v), res)
            except Exception:  # noqa: BLE001 - transient failures leave the hour undone for the next run
                log.error("hour %s failed, left undone:\n%s", v, traceback.format_exc())
                append_csv(fail_csv, [{"valid_time": v, "channel": "*", "reason": "processing error (left undone)"}])
                continue
            prov_buf += res["prov"]
            append_csv(fail_csv, res["failures"])
            append_csv(qcsv, [{"valid_time": v, **r} for r in qrows])
            nbytes += res["download_bytes"]
            if not args.keep_grib:
                shutil.rmtree(d, ignore_errors=True)
            if res["failures"]:
                log.warning("%s: %d channel problems", v, len(res["failures"]))
            if (n + 1) % 24 == 0 or n + 1 == len(todo):
                flush_provenance(prov_buf)
                prov_buf = []
                el = time.time() - t0
                log.info("%d/%d hours, %.1f s/hour, %.2f GB downloaded", n + 1, len(todo), el / (n + 1), nbytes / 1e9)
    flush_provenance(prov_buf)
    log.info("finished %s..%s", start, end)


if __name__ == "__main__":
    main()
