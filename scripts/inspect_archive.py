"""Stage A: inspect the HRRR archive before downloading anything at scale.

Outputs
  metadata/variable_inventory.csv   per sample hour x channel: presence, source file, sizes, times, packing
  metadata/grid_metadata.json       native grid, crop window, 6 km grid, Tufts cell, wind rotation
  metadata/archive_availability.csv per valid hour: which required files exist (full-range listing scan)
  metadata/missing_timestamps.csv   valid hours with any required file missing
  logs/inspect_archive_report.md    human-readable summary

Usage
  python scripts/inspect_archive.py                 # everything
  python scripts/inspect_archive.py --skip-scan     # samples + grid only
  python scripts/inspect_archive.py --network-check # one request; use on a compute node via srun
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from common import (METADATA_DIR, LOG_DIR, ANL_DESC, APCP_1H_DESC, channel_sources, coarsen, crop,
                    decode_message, fetch_entries, fetch_idx, file_key, file_url, find_entry, grid_signature,
                    haversine_km, hourly_range, load_config, make_session, native_xy, north_vector, period_hours,
                    required_files, setup_logging, write_json)

SAMPLE_DATES = ["2023-10-01", "2025-01-15", "2025-04-01", "2025-07-15", "2026-09-30"]
SAMPLE_HOURS = [0, 18]
S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
KEY_RE = re.compile(r"hrrr\.t(\d\d)z\.wrf(sfc|prs)f(\d\d)\.grib2(\.idx)?$")

log = setup_logging("inspect_archive")


def network_check(cfg) -> None:
    s = make_session(retries=1)
    url = file_url(cfg, pd.Timestamp(cfg["period"]["start"]), "sfc", 0)
    r = s.head(url + ".idx", timeout=20)
    print(f"network check: HEAD {url}.idx -> HTTP {r.status_code}")


# --------------------------------------------------------------------------- sample inventory

def inspect_samples(cfg, session) -> tuple[pd.DataFrame, dict]:
    rows, grid_meta = [], None
    for date in SAMPLE_DATES:
        for hour in SAMPLE_HOURS:
            v = pd.Timestamp(date) + pd.Timedelta(hours=hour)
            log.info("sample %s", v)
            idx_cache, size_cache = {}, {}
            for prod, init, fhr in required_files(v):
                url = file_url(cfg, init, prod, fhr)
                idx_cache[(prod, init, fhr)] = fetch_idx(session, url)
                h = session.head(url, timeout=30)
                size_cache[(prod, init, fhr)] = int(h.headers.get("Content-Length", 0)) if h.ok else None

            items = [(c["name"], c) for c in cfg["channels"]]
            items += [(f"static:{s['name']}", {**s, "source": "f00"}) for s in cfg["static_fields"]]
            items += [("doc:apcp_f00", {"var": "APCP", "level": "surface", "source": "doc_f00"})]
            for name, ch in items:
                if ch["source"] == "doc_f00":
                    cands = [("sfc", v, 0, "0-0 day acc fcst")]
                else:
                    cands = channel_sources(ch, v)
                row = {"sample_valid_time": v, "channel": name, "var": ch["var"], "level": ch["level"], "present": False}
                for prod in ("sfc", "prs"):
                    ents = idx_cache.get((prod, v, 0))
                    row[f"in_wrf{prod}f00"] = bool(ents and [e for e in ents if e.var == ch["var"] and e.level == ch["level"]])
                for prod, init, fhr, desc in cands:
                    ents = find_entry(idx_cache.get((prod, init, fhr)), ch["var"], ch["level"], desc)
                    if not ents:
                        continue
                    e = ents[0]
                    url = file_url(cfg, init, prod, fhr)
                    data = fetch_entries(session, url, [e])[e.num]
                    want_latlon = grid_meta is None
                    vals, meta = decode_message(data, with_latlon=want_latlon)
                    if want_latlon:
                        grid_meta = meta
                    row.update({
                        "present": True, "n_matches": len(ents), "source_product": f"wrf{prod}f{fhr:02d}",
                        "source_file": file_key(init, prod, fhr), "file_bytes": size_cache.get((prod, init, fhr)),
                        "message_bytes": len(data), "idx_desc": desc, "init_time": meta["init_time"],
                        "lead_h": fhr, "valid_time": meta["valid_time"], "step_range": meta["stepRange"],
                        "step_type": meta["stepType"],
                        "accum_hours": (meta["endStep"] - meta["startStep"]) if meta["stepType"] == "accum" else None,
                        "valid_matches_expected": meta["valid_time"] == v, "binary_scale_E": meta["binaryScaleFactor"],
                        "decimal_scale_D": meta["decimalScaleFactor"], "bits_per_value": meta["bitsPerValue"],
                        "q_src": meta["q_src"], "packing": meta["packingType"], "grib_units": meta["units"],
                        "uv_relative_to_grid": meta["uvRelativeToGrid"],
                        "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals)),
                        "nan_frac": float(np.isnan(vals).mean()),
                    })
                    break
                rows.append(row)
    return pd.DataFrame(rows), grid_meta


# --------------------------------------------------------------------------- grid metadata

def build_grid_metadata(cfg, meta) -> dict:
    grid = grid_signature(meta)
    lat, lon = meta["latitudes"], meta["longitudes"]
    bb = cfg["bbox"]
    inside = (lat >= bb["south"]) & (lat <= bb["north"]) & (lon >= bb["west"]) & (lon <= bb["east"])
    rows = np.where(inside.any(axis=1))[0]
    cols = np.where(inside.any(axis=0))[0]
    j0, j1, i0, i1 = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
    f = cfg["coarsen"]

    def make_even(a, b, n):
        if (b - a + 1) % f:
            if b + 1 < n:
                b += 1
            else:
                a -= 1
        return a, b

    j0, j1 = make_even(j0, j1, grid["Ny"])
    i0, i1 = make_even(i0, i1, grid["Nx"])
    window = {"j0": j0, "j1": j1, "i0": i0, "i1": i1, "ny": j1 - j0 + 1, "nx": i1 - i0 + 1}

    x, y, proj = native_xy(grid)
    # Check that the projection reproduces GRIB lat/lon (max error over a sample of points).
    jj, ii = np.meshgrid(np.arange(0, grid["Ny"], 97), np.arange(0, grid["Nx"], 97), indexing="ij")
    plon, plat = proj(x[ii], y[jj], inverse=True)
    proj_err_km = float(haversine_km(plat, plon, lat[jj, ii], lon[jj, ii]).max())

    # Does the bbox extend outside the HRRR domain? (check bbox edges against the native grid)
    edge_lon = np.r_[np.linspace(bb["west"], bb["east"], 400), np.full(400, bb["east"]),
                     np.linspace(bb["east"], bb["west"], 400), np.full(400, bb["west"])]
    edge_lat = np.r_[np.full(400, bb["north"]), np.linspace(bb["north"], bb["south"], 400),
                     np.full(400, bb["south"]), np.linspace(bb["south"], bb["north"], 400)]
    ex, ey = proj(edge_lon, edge_lat)
    ei = (ex - x[0]) / grid["DxInMetres"]
    ej = (ey - y[0]) / grid["DyInMetres"]
    outside = (ei < -0.5) | (ei > grid["Nx"] - 0.5) | (ej < -0.5) | (ej > grid["Ny"] - 0.5)

    # 6 km grid: cell centers are block means of native x/y.
    xc = crop(np.broadcast_to(x, (grid["Ny"], grid["Nx"])), window)[0]
    yc = crop(np.broadcast_to(y[:, None], (grid["Ny"], grid["Nx"])), window)[:, 0]
    xc = xc[: (len(xc) // f) * f].reshape(-1, f).mean(axis=1)
    yc = yc[: (len(yc) // f) * f].reshape(-1, f).mean(axis=1)
    XC, YC = np.meshgrid(xc, yc)
    clon, clat = proj(XC, YC, inverse=True)
    in_bbox = (clat >= bb["south"]) & (clat <= bb["north"]) & (clon >= bb["west"]) & (clon <= bb["east"])

    # Tufts: nearest native cell by projected distance, cross-checked with great-circle distance.
    t = cfg["tufts"]
    tx, ty = proj(t["lon"], t["lat"])
    ti = int(round((tx - x[0]) / grid["DxInMetres"]))
    tj = int(round((ty - y[0]) / grid["DyInMetres"]))
    sl = (slice(tj - 3, tj + 4), slice(ti - 3, ti + 4))
    d = haversine_km(t["lat"], t["lon"], lat[sl], lon[sl])
    bj, bi = np.unravel_index(np.argmin(d), d.shape)
    north = north_vector(proj, float(lon[tj, ti]), float(lat[tj, ti]))
    lov = grid["LoVInDegrees"] - 360 if grid["LoVInDegrees"] > 180 else grid["LoVInDegrees"]
    cone_angle = np.degrees(np.sin(np.radians(grid["Latin1InDegrees"])) * np.radians(float(lon[tj, ti]) - lov))

    return {
        "native_grid": {**grid, "projection": "Lambert conformal conic",
                        "proj4": proj.srs, "earth_radius_m": 6371229.0,
                        "x0_m": float(x[0]), "y0_m": float(y[0]),
                        "projection_vs_grib_latlon_max_error_km": proj_err_km},
        "bbox": bb,
        "bbox_outside_hrrr_domain_fraction_of_edge": float(outside.mean()),
        "crop_window_native": window,
        "coarsen_factor": f,
        "coarse_grid": {
            "ny": int(len(yc)), "nx": int(len(xc)), "dx_m": grid["DxInMetres"] * f, "dy_m": grid["DyInMetres"] * f,
            "x_m": xc.tolist(), "y_m": yc.tolist(),
            "lat_min": float(clat.min()), "lat_max": float(clat.max()),
            "lon_min": float(clon.min()), "lon_max": float(clon.max()),
            "cells_inside_bbox": int(in_bbox.sum()), "cells_total": int(in_bbox.size),
            "method": f"{f}x{f} block mean of native cells; any missing native cell -> missing coarse cell",
        },
        "tufts": {
            "requested_lat": t["lat"], "requested_lon": t["lon"],
            "native_j": tj, "native_i": ti,
            "cell_lat": float(lat[tj, ti]), "cell_lon": float(lon[tj, ti]),
            "distance_km": float(haversine_km(t["lat"], t["lon"], lat[tj, ti], lon[tj, ti])),
            "great_circle_check_agrees": bool(bj == 3 and bi == 3),
            "inside_crop_window": bool(window["j0"] <= tj <= window["j1"] and window["i0"] <= ti <= window["i1"]),
            "north_vector_grid_xy": [float(north[0]), float(north[1])],
            "grid_rotation_deg_projection": float(np.degrees(np.arctan2(north[0], north[1]))),
            "grid_rotation_deg_cone_formula": float(cone_angle),
            "wind_rotation": "u_e = u_g*n_y - v_g*n_x ; v_e = u_g*n_x + v_g*n_y (n = north_vector_grid_xy)",
        },
        "row_order": "row 0 = southernmost (jScansPositively=1)",
    }


# --------------------------------------------------------------------------- availability scan

def list_day(session, cfg, day: pd.Timestamp) -> dict[str, int]:
    prefix = f"hrrr.{day:%Y%m%d}/{cfg['archive']['domain']}/hrrr.t"
    out, token = {}, None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = session.get(cfg["archive"]["base_url"] + "/", params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.findall("s:Contents", S3_NS):
            key = c.findtext("s:Key", namespaces=S3_NS)
            m = KEY_RE.search(key)
            if m and int(m.group(3)) <= 1:
                out[key] = int(c.findtext("s:Size", namespaces=S3_NS))
        token = root.findtext("s:NextContinuationToken", namespaces=S3_NS)
        if not token:
            return out


def availability_scan(cfg, session) -> pd.DataFrame:
    hours = period_hours(cfg)
    days = pd.date_range((hours[0] - pd.Timedelta(hours=1)).normalize(), hours[-1].normalize(), freq="D")
    log.info("listing %d days", len(days))
    keys: dict[str, int] = {}
    with ThreadPoolExecutor(16) as ex:
        for k, res in enumerate(ex.map(lambda d: list_day(session, cfg, d), days)):
            keys.update(res)
            if k % 100 == 0:
                log.info("  listed %d/%d days", k, len(days))
    rows = []
    for v in hours:
        row = {"valid_time": v}
        missing = []
        for prod, init, fhr in required_files(v):
            k = file_key(init, prod, fhr, cfg["archive"]["domain"])
            tag = f"wrf{prod}f{fhr:02d}_init{'' if init == v else '_prev'}"
            row[f"{tag}_bytes"] = keys.get(k)
            if k not in keys:
                missing.append(f"{k} (grib2)")
            if k + ".idx" not in keys:
                missing.append(f"{k}.idx")
        row["missing"] = "; ".join(missing)
        row["all_present"] = not missing
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-scan", action="store_true")
    ap.add_argument("--skip-samples", action="store_true")
    ap.add_argument("--network-check", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.network_check:
        network_check(cfg)
        return
    METADATA_DIR.mkdir(exist_ok=True)
    session = make_session(cfg["download"]["retries"])

    if not args.skip_samples:
        report = []
        inv, meta = inspect_samples(cfg, session)
        inv.to_csv(METADATA_DIR / "variable_inventory.csv", index=False)
        grid = build_grid_metadata(cfg, meta)
        write_json(METADATA_DIR / "grid_metadata.json", grid)
        log.info("wrote variable_inventory.csv (%d rows) and grid_metadata.json", len(inv))

        req = inv[~inv.channel.str.startswith("doc:")]
        pres = req.groupby("channel")["present"].all()
        report += [f"Sample hours: {', '.join(SAMPLE_DATES)} at {SAMPLE_HOURS} UTC", "",
                   f"Channels present in every sample: {int(pres.sum())}/{len(pres)}"]
        if (~pres).any():
            report.append(f"Missing in some sample: {', '.join(pres[~pres].index)}")
        src = req[req.present].groupby("channel")["source_product"].agg(lambda s: "/".join(sorted(set(s))))
        report += ["", "Source product per channel: " + ", ".join(f"{k}={v}" for k, v in src.items()), ""]
        ok_valid = bool(req[req.present]["valid_matches_expected"].all())
        report.append(f"All decoded valid times equal the requested valid hour: {ok_valid}")
        ap1 = req[req.channel == "apcp1h"]
        report.append(f"apcp1h accumulation hours (f01 of previous init): {sorted(ap1.accum_hours.dropna().unique().tolist())}, "
                      f"step ranges: {sorted(ap1.step_range.dropna().unique().tolist())}")
        doc = inv[inv.channel == "doc:apcp_f00"]
        report.append(f"APCP in f00 files: step ranges {sorted(doc.step_range.dropna().unique().tolist())}, "
                      f"max value {doc['max'].max()} (zero-length accumulation, not used)")
        dsw = req[(req.channel == "dswrf")]
        report.append("DSWRF f00 max per sample: " + ", ".join(
            f"{t:%Y-%m-%d %HZ}={m:.0f}" for t, m in zip(dsw.sample_valid_time, dsw["max"])))
        msg_mb = req[req.present & ~req.channel.str.startswith("static")].groupby("sample_valid_time")["message_bytes"].sum() / 1e6
        report.append(f"Bytes downloaded per valid hour (49 channels): {msg_mb.min():.1f}-{msg_mb.max():.1f} MB")
        fsz = inv.dropna(subset=["file_bytes"]).groupby("source_product")["file_bytes"].mean() / 1e6
        report.append("Mean full file sizes: " + ", ".join(f"{k}={v:.0f} MB" for k, v in fsz.items()))
        q = req[req.present].groupby("channel")["q_src"].max()
        report.append("Max GRIB packing step per channel: " + ", ".join(f"{k}={v:g}" for k, v in q.items()))
        w = grid["crop_window_native"]
        cg = grid["coarse_grid"]
        report += ["", f"Crop window (native): rows {w['j0']}-{w['j1']}, cols {w['i0']}-{w['i1']} ({w['ny']} x {w['nx']})",
                   f"6 km grid: {cg['ny']} x {cg['nx']} (lat {cg['lat_min']:.2f}..{cg['lat_max']:.2f}, "
                   f"lon {cg['lon_min']:.2f}..{cg['lon_max']:.2f}); {cg['cells_inside_bbox']}/{cg['cells_total']} cell centers inside bbox",
                   f"Fraction of bbox edge outside HRRR domain: {grid['bbox_outside_hrrr_domain_fraction_of_edge']:.3f}",
                   f"Projection vs GRIB lat/lon max error: {grid['native_grid']['projection_vs_grib_latlon_max_error_km']:.4f} km",
                   f"Tufts cell: j={grid['tufts']['native_j']} i={grid['tufts']['native_i']} "
                   f"({grid['tufts']['cell_lat']:.4f}, {grid['tufts']['cell_lon']:.4f}), {grid['tufts']['distance_km']:.2f} km away; "
                   f"great-circle check agrees: {grid['tufts']['great_circle_check_agrees']}",
                   f"Wind rotation at Tufts: projection {grid['tufts']['grid_rotation_deg_projection']:.3f} deg, "
                   f"cone formula {grid['tufts']['grid_rotation_deg_cone_formula']:.3f} deg "
                   "(same magnitude; the two use opposite sign conventions)", ""]
        (LOG_DIR / "inspect_archive_samples.md").write_text("\n".join(report) + "\n")

    if not args.skip_scan:
        report = []
        av = availability_scan(cfg, session)
        av.to_csv(METADATA_DIR / "archive_availability.csv", index=False)
        miss = av[~av.all_present][["valid_time", "missing"]].assign(stage="archive_scan")
        miss.to_csv(METADATA_DIR / "missing_timestamps.csv", index=False)
        report += [f"Availability scan: {len(av)} valid hours, {int(av.all_present.sum())} complete, "
                   f"{len(miss)} with at least one required file missing (see metadata/missing_timestamps.csv)"]
        for col in [c for c in av.columns if c.endswith("_bytes")]:
            report.append(f"  {col}: mean {av[col].mean() / 1e6:.0f} MB, missing {int(av[col].isna().sum())}")
        log.info("availability: %d/%d complete", av.all_present.sum(), len(av))
        (LOG_DIR / "inspect_archive_scan.md").write_text("\n".join(report) + "\n")

    parts = ["# Stage A: archive inspection", ""]
    for name in ("inspect_archive_samples.md", "inspect_archive_scan.md"):
        if (LOG_DIR / name).exists():
            parts.append((LOG_DIR / name).read_text())
    (LOG_DIR / "inspect_archive_report.md").write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
