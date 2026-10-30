"""Stage B report: measure the pilot's storage footprint and int16 precision, project 3 years, apply the gate.

Requires a pilot processed with `process_grib.py --compare-float32`.

Outputs logs/pilot_report.md and metadata/pilot_summary.json.

Usage
  python scripts/pilot_report.py --start 2025-07-14T00 --end 2025-07-20T23
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc, Delta

from common import (INT16_MAX, LOG_DIR, METADATA_DIR, du_bytes, hourly_range, load_config, period_hours,
                    quantize, time_index, write_json)

GATE_GB = 70.0
BUDGET_GB = 100.0
REL_TOL = 1e-3  # decimal steps (0.1, 0.01) are not exactly representable in binary floating point


def chunk_bytes(store_path, array: str, t_indices) -> np.ndarray:
    """Bytes of the chunk files for each time index (chunks are (1, C, y, x), stored under c/<t>/)."""
    return np.array([du_bytes(store_path / array / "c" / str(t)) for t in t_indices], dtype=float)


def encoded_size(a: np.ndarray, delta: bool) -> int:
    codec = Blosc("zstd", 5, Blosc.BITSHUFFLE)
    if delta:
        a = Delta(dtype=a.dtype).encode(a)
    return len(codec.encode(np.ascontiguousarray(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-07-14T00")
    ap.add_argument("--end", default="2025-07-20T23")
    args = ap.parse_args()
    cfg = load_config()
    store = cfg["store_path"]
    comp_path = store.with_name(store.stem + "_f32compare.zarr")
    root = zarr.open_group(str(store), mode="r")
    comp = zarr.open_group(str(comp_path), mode="r")
    hours = hourly_range(args.start, args.end)
    tix = [time_index(cfg, v) for v in hours]
    done = root["done"][:][tix]
    tix = [t for t, d in zip(tix, done) if d]
    n = len(tix)
    T = len(period_hours(cfg))

    # ---- storage
    i16 = chunk_bytes(store, "inputs", tix)
    f32x = chunk_bytes(store, "inputs_f32", tix) if "inputs_f32" in root else np.zeros(n)
    f32c = chunk_bytes(comp_path, "inputs_f32", tix)
    per_hour = i16 + f32x
    small_arrays = sum(du_bytes(store / a) for a in ("input_valid", "target_fields", "done", "record_ok"))
    fixed = sum(du_bytes(store / a) for a in ("lat", "lon", "x", "y", "time", "bbox_mask", "static_orography", "static_land"))
    prov_files = [METADATA_DIR / "provenance" / f"{d:%Y%m%d}.parquet" for d in pd.date_range(hours[0], hours[-1], freq="D")]
    prov_bytes = sum(du_bytes(p) for p in prov_files)
    days_period = T / 24
    proj_inputs = per_hour.mean() * T
    proj_small = small_arrays / max(n, 1) * T  # upper bound: small arrays grow per touched day-chunk
    proj_prov = prov_bytes / len(prov_files) * days_period
    proj_total = proj_inputs + proj_small + fixed + proj_prov
    proj_f32 = f32c.mean() * T

    # ---- download volume
    prov = pd.concat([pd.read_parquet(p) for p in prov_files if p.exists()], ignore_index=True)
    prov = prov[prov.valid_time.isin(hours)]
    dl_per_hour = prov.groupby("valid_time").message_bytes.sum().mean()

    # ---- precision
    qfiles = sorted(glob.glob(str(LOG_DIR / "quantization_check_*.csv")))
    q = pd.concat([pd.read_csv(f) for f in qfiles], ignore_index=True)
    q["valid_time"] = pd.to_datetime(q.valid_time, format="mixed")
    q = q[q.valid_time.isin(hours)]
    attrs = root["inputs"].attrs
    chans = attrs["channels"]
    qsrc = dict(zip(chans, attrs["q_src_max"]))
    step = dict(zip(chans, attrs["scale"]))
    pq = prov[prov.ok.astype(bool)].groupby("channel").q_src.max()
    sample_t = tix[:: max(1, n // 24)]
    stack = np.stack([comp["inputs_f32"][t] for t in sample_t])  # (S, C, y, x)
    names_all = comp["inputs_f32"].attrs["channels"]
    std = {c: float(np.nanstd(stack[:, k])) for k, c in enumerate(names_all)}
    rows = []
    for c in chans:
        g = q[q.channel == c]
        qs = max(qsrc[c] or 0, float(pq.get(c, 0)))
        max_err = float(g.max_abs_err.max())
        rmse = float(np.sqrt((g.rmse ** 2).mean()))
        bound = max(abs(b) for b in next(ch["bounds"] for ch in cfg["channels"] if ch["name"] == c))
        tol = qs / 2 * (1 + REL_TOL) + 2 * float(np.spacing(np.float32(bound)))
        rows.append({"channel": c, "int16_step": step[c], "q_src_max": qs, "max_abs_err": max_err,
                     "limit_q_src_over_2": qs / 2, "tolerance_used": tol, "pass": bool(max_err <= tol),
                     "rmse": rmse, "pilot_std": std[c], "rmse_over_std": rmse / std[c] if std[c] else np.nan,
                     "secondary_pass_1e-4": bool(std[c] and rmse / std[c] < 1e-4)})
    prec = pd.DataFrame(rows)
    prec.to_csv(LOG_DIR / "pilot_precision.csv", index=False)
    failing = prec[~prec["pass"]].channel.tolist()

    # ---- alternative encodings on a subsample (for the record)
    chs = cfg["channels"]
    enc = {"float32 (no filter)": 0, "int16 bounds-step (no filter)": 0, "int16 bounds-step + delta": 0,
           "int16 source-step (no filter)": 0, "int16 source-step + delta [chosen]": 0}
    for s in range(stack.shape[0]):
        enc["float32 (no filter)"] += encoded_size(stack[s], False)
        qb = np.stack([quantize(stack[s, k], (c["bounds"][1] - c["bounds"][0]) / (2 * INT16_MAX),
                                (c["bounds"][1] + c["bounds"][0]) / 2)[0] for k, c in enumerate(chs)])
        qs_ = np.stack([quantize(stack[s, k], c["scale"], c["offset"])[0] for k, c in enumerate(chs)])
        enc["int16 bounds-step (no filter)"] += encoded_size(qb, False)
        enc["int16 bounds-step + delta"] += encoded_size(qb, True)
        enc["int16 source-step (no filter)"] += encoded_size(qs_, False)
        enc["int16 source-step + delta [chosen]"] += encoded_size(qs_, True)
    enc = {k: v / stack.shape[0] for k, v in enc.items()}

    gate_pass = proj_total / 1e9 <= GATE_GB and not failing
    summary = {
        "pilot_start": str(hours[0]), "pilot_end": str(hours[-1]), "hours_processed": n,
        "hours_record_ok": int(root["record_ok"][:][tix].sum()),
        "mb_per_hour_inputs_stored": per_hour.mean() / 1e6, "mb_per_hour_float32_compare": f32c.mean() / 1e6,
        "compression_vs_raw_float32": (len(chans) * 223 * 291 * 4) / per_hour.mean(),
        "projected_gb": {"inputs": proj_inputs / 1e9, "per_hour_small_arrays": proj_small / 1e9,
                         "static_and_coords": fixed / 1e9, "provenance_parquet": proj_prov / 1e9,
                         "total": proj_total / 1e9, "if_all_float32": proj_f32 / 1e9},
        "download_mb_per_hour": dl_per_hour / 1e6, "projected_download_tb": dl_per_hour * T / 1e12,
        "encodings_mb_per_hour_subsample": {k: v / 1e6 for k, v in enc.items()},
        "precision_rule": "max |int16 - float32| <= q_src_max/2, plus tolerance for floating-point representation: "
                          "1e-3 relative (decimal steps) + 2 float32 ulps at the channel bound (float32 reference)",
        "precision_failing_channels": failing,
        "secondary_rmse_over_std_lt_1e-4_failing": prec[~prec["secondary_pass_1e-4"]].channel.tolist(),
        "gate_gb": GATE_GB, "budget_gb": BUDGET_GB, "gate_pass": bool(gate_pass),
    }
    write_json(METADATA_DIR / "pilot_summary.json", summary)

    pg = summary["projected_gb"]
    lines = [
        "# Stage B: 7-day pilot report", "",
        f"Pilot: {hours[0]} .. {hours[-1]} ({n} hours processed, {summary['hours_record_ok']} fully valid)", "",
        "## Storage", "",
        f"- Stored inputs: {summary['mb_per_hour_inputs_stored']:.2f} MB/hour "
        f"(raw float32 would be {len(chans) * 223 * 291 * 4 / 1e6:.1f} MB; {summary['compression_vs_raw_float32']:.1f}x smaller)",
        f"- Same fields as compressed float32: {summary['mb_per_hour_float32_compare']:.2f} MB/hour",
        f"- Projected 3 years ({T} hours): inputs {pg['inputs']:.1f} GB + per-hour arrays {pg['per_hour_small_arrays']:.2f} GB "
        f"+ static/coords {pg['static_and_coords']:.3f} GB + provenance {pg['provenance_parquet']:.2f} GB = **{pg['total']:.1f} GB**",
        f"- If every channel were float32: {pg['if_all_float32']:.1f} GB",
        f"- Download volume: {summary['download_mb_per_hour']:.1f} MB/hour, {summary['projected_download_tb']:.2f} TB over 3 years (streamed, not kept)",
        "", "Alternative encodings measured on a pilot subsample (MB/hour):", "",
        "| Encoding | MB/hour | 3-year GB |", "|---|---|---|",
        *[f"| {k} | {v / 1e6:.2f} | {v * T / 1e9:.0f} |" for k, v in enc.items()],
        "", "## Precision (int16 vs float32 6 km fields)", "",
        "Rule: max abs error <= half the largest GRIB packing step seen (plus floating-point representation tolerance: "
        "0.1% for decimal steps and 2 float32 ulps at the channel bound, since the float32 reference itself carries that error). "
        f"Failing channels: {failing or 'none'}",
        f"Secondary (reported only) RMSE/std < 1e-4 fails for {len(summary['secondary_rmse_over_std_lt_1e-4_failing'])}/{len(chans)} channels; "
        "the GRIB source itself is quantized at a comparable level, see logs/pilot_precision.csv.", "",
        "| Channel | int16 step | GRIB step (max) | max error | RMSE/std |", "|---|---|---|---|---|",
        *[f"| {r.channel} | {r.int16_step:g} | {r.q_src_max:g} | {r.max_abs_err:.4g} | {r.rmse_over_std:.1e} |" for r in prec.itertuples()],
        "", "## Gate", "",
        f"Projected total {pg['total']:.1f} GB vs gate {GATE_GB:.0f} GB (budget {BUDGET_GB:.0f} GB, leaving >= 30 GB for temp files and checkpoints); "
        f"precision failures: {len(failing)}. **{'PASS' if gate_pass else 'FAIL'}**",
    ]
    (LOG_DIR / "pilot_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    print(f"\nGATE: {'PASS' if gate_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
