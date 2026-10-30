"""Shared helpers for the HRRR Northeast CNN dataset pipeline."""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd
import pyproj
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "metadata"
LOG_DIR = PROJECT_ROOT / "logs"

INT16_MISSING = -32768
INT16_MAX = 32767
EARTH_RADIUS_M = 6371229.0  # GRIB shapeOfTheEarth = 6, used by HRRR

TARGET_NAMES = ["t2m", "rh2m", "u10", "v10", "gust", "precip_1h", "rain_binary"]
TARGET_UNITS = ["K", "%", "m s-1", "m s-1", "m s-1", "mm", "1"]


# --------------------------------------------------------------------------- config

def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(path or PROJECT_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    root = Path(os.environ.get("HRRR_DATA_ROOT", cfg["data_root"]))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    cfg["data_root_path"] = root
    cfg["store_path"] = root / "processed" / cfg["store_name"]
    cfg["tmp_path"] = root / "tmp"
    cfg["channels"] = build_channels(cfg)
    if cfg["storage"].get("int16_step", "bounds") == "source":
        apply_source_steps(cfg["channels"])
    return cfg


def apply_source_steps(channels: list[dict]) -> None:
    """Set each channel's int16 step to the finest GRIB packing step seen in Stage A whose int16 range covers
    the channel bounds, and align the offset to that step so stored integers sit on HRRR's value grid.
    Max quantization error is then step/2 <= (largest GRIB step)/2."""
    inv_path = METADATA_DIR / "variable_inventory.csv"
    if not inv_path.exists():
        return
    inv = pd.read_csv(inv_path)
    inv = inv[inv["present"].astype(bool)]
    for c in channels:
        steps = sorted(inv.loc[inv.channel == c["name"], "q_src"].dropna().unique())
        lo, hi = c["bounds"]
        fit = [s for s in steps if s * (2 * INT16_MAX - 2) >= hi - lo]
        if not fit:
            continue
        s = float(fit[0])
        c["scale"] = s
        c["offset"] = float(np.round((hi + lo) / 2 / s) * s)
        c["step_rule"] = "finest GRIB packing step whose int16 range covers bounds"
        c["q_src_max"] = float(max(steps))


def build_channels(cfg: dict) -> list[dict]:
    channels = []
    for c in cfg["surface_channels"]:
        channels.append({**c, "source": c.get("source", "f00"), "kind": "surface"})
    for p in cfg["pressure_variables"]:
        for lev in cfg["pressure_levels"]:
            bounds = p["bounds_by_level"][lev] if "bounds_by_level" in p else p["bounds"]
            channels.append({
                "name": f"{p['name']}{lev}", "var": p["var"], "level": f"{lev} mb",
                "units": p["units"], "bounds": list(bounds), "source": "f00",
                "kind": "pressure", "pressure_hpa": lev,
            })
    for c in channels:
        lo, hi = c["bounds"]
        c["scale"] = (hi - lo) / (2 * INT16_MAX)
        c["offset"] = (hi + lo) / 2
        c["step_rule"] = "bounds: (hi - lo) / 65534"
    return channels


def setup_logging(name: str) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_DIR / f"{name}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(o))
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


# --------------------------------------------------------------------------- time

def hourly_range(start, end) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="h")


def period_hours(cfg: dict) -> pd.DatetimeIndex:
    return hourly_range(cfg["period"]["start"], cfg["period"]["end"])


def time_index(cfg: dict, t: pd.Timestamp) -> int:
    return int((pd.Timestamp(t) - pd.Timestamp(cfg["period"]["start"])) / pd.Timedelta(hours=1))


# --------------------------------------------------------------------------- archive

def file_key(init: pd.Timestamp, product: str, fhr: int, domain: str = "conus") -> str:
    return f"hrrr.{init:%Y%m%d}/{domain}/hrrr.t{init:%H}z.wrf{product}f{fhr:02d}.grib2"


def file_url(cfg: dict, init: pd.Timestamp, product: str, fhr: int) -> str:
    return f"{cfg['archive']['base_url']}/{file_key(init, product, fhr, cfg['archive']['domain'])}"


def make_session(retries: int = 5, pool: int = 32) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET", "HEAD"))
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool, pool_maxsize=pool)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


ANL_DESC = "anl"
APCP_1H_DESC = "0-1 hour acc fcst"


def channel_sources(ch: dict, valid: pd.Timestamp) -> list[tuple[str, pd.Timestamp, int, str]]:
    """Candidate (product, init_time, forecast_hour, idx_desc) for a channel valid at `valid`, in preference order."""
    if ch["source"] == "f01prev":
        return [("sfc", valid - pd.Timedelta(hours=1), 1, APCP_1H_DESC)]
    return [("sfc", valid, 0, ANL_DESC), ("prs", valid, 0, ANL_DESC)]


def required_files(valid: pd.Timestamp) -> list[tuple[str, pd.Timestamp, int]]:
    return [("sfc", valid, 0), ("prs", valid, 0), ("sfc", valid - pd.Timedelta(hours=1), 1)]


@dataclass
class IdxEntry:
    num: int
    start: int
    end: int | None  # inclusive; None for the last message
    var: str
    level: str
    desc: str

    @property
    def nbytes(self) -> int | None:
        return None if self.end is None else self.end - self.start + 1


def parse_idx(text: str) -> list[IdxEntry]:
    rows = [line.split(":") for line in text.strip().splitlines() if line.strip()]
    entries = []
    for k, r in enumerate(rows):
        start = int(r[1])
        end = int(rows[k + 1][1]) - 1 if k + 1 < len(rows) else None
        entries.append(IdxEntry(int(r[0]), start, end, r[3], r[4], r[5]))
    return entries


def fetch_idx(session: requests.Session, url: str, timeout: float = 60) -> list[IdxEntry] | None:
    r = session.get(url + ".idx", timeout=timeout)
    if r.status_code in (403, 404):
        return None
    r.raise_for_status()
    return parse_idx(r.text)


def find_entry(entries: list[IdxEntry] | None, var: str, level: str, desc: str) -> list[IdxEntry]:
    if not entries:
        return []
    return [e for e in entries if e.var == var and e.level == level and e.desc == desc]


def fetch_entries(session: requests.Session, url: str, entries: list[IdxEntry], timeout: float = 60) -> dict[int, bytes]:
    """Download the given idx entries with HTTP Range requests, merging contiguous messages."""
    out: dict[int, bytes] = {}
    entries = sorted(entries, key=lambda e: e.start)
    groups: list[list[IdxEntry]] = []
    for e in entries:
        if groups and groups[-1][-1].end is not None and groups[-1][-1].end + 1 == e.start:
            groups[-1].append(e)
        else:
            groups.append([e])
    for g in groups:
        end = "" if g[-1].end is None else g[-1].end
        r = session.get(url, headers={"Range": f"bytes={g[0].start}-{end}"}, timeout=timeout)
        r.raise_for_status()
        data = r.content
        for e in g:
            lo = e.start - g[0].start
            hi = None if e.end is None else e.end - g[0].start + 1
            out[e.num] = data[lo:hi]
    return out


# --------------------------------------------------------------------------- GRIB decoding

GRID_KEYS = ["Nx", "Ny", "latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees",
             "LoVInDegrees", "Latin1InDegrees", "Latin2InDegrees", "DxInMetres", "DyInMetres",
             "shapeOfTheEarth", "jScansPositively", "iScansNegatively"]
META_KEYS = ["shortName", "name", "units", "typeOfLevel", "level", "dataDate", "dataTime",
             "validityDate", "validityTime", "stepRange", "stepType", "startStep", "endStep",
             "stepUnits", "binaryScaleFactor", "decimalScaleFactor", "bitsPerValue", "packingType"]


def _get(h, key):
    try:
        return eccodes.codes_get(h, key)
    except eccodes.KeyValueNotFoundError:
        return None


def decode_message(data: bytes, with_latlon: bool = False) -> tuple[np.ndarray, dict]:
    h = eccodes.codes_new_from_message(data)
    try:
        meta = {k: _get(h, k) for k in META_KEYS + GRID_KEYS}
        meta["uvRelativeToGrid"] = _get(h, "uvRelativeToGrid")
        ny, nx = meta["Ny"], meta["Nx"]
        vals = eccodes.codes_get_values(h).astype(np.float32)
        if _get(h, "bitmapPresent"):
            vals[vals == _get(h, "missingValue")] = np.nan
        vals = vals.reshape(ny, nx)
        if with_latlon:
            meta["latitudes"] = eccodes.codes_get_array(h, "latitudes").reshape(ny, nx)
            lons = eccodes.codes_get_array(h, "longitudes").reshape(ny, nx)
            meta["longitudes"] = np.where(lons > 180, lons - 360, lons)
    finally:
        eccodes.codes_release(h)
    meta["init_time"] = pd.Timestamp(f"{meta['dataDate']}{meta['dataTime']:04d}")
    meta["valid_time"] = pd.Timestamp(f"{meta['validityDate']}{meta['validityTime']:04d}")
    meta["q_src"] = 2.0 ** meta["binaryScaleFactor"] / 10.0 ** meta["decimalScaleFactor"]
    return vals, meta


def grid_signature(meta: dict) -> dict:
    return {k: (round(float(meta[k]), 6) if isinstance(meta[k], float) else meta[k]) for k in GRID_KEYS}


# --------------------------------------------------------------------------- grid

def hrrr_proj(grid: dict) -> pyproj.Proj:
    lov = grid["LoVInDegrees"]
    lov = lov - 360 if lov > 180 else lov
    return pyproj.Proj(proj="lcc", lat_1=grid["Latin1InDegrees"], lat_2=grid["Latin2InDegrees"],
                       lat_0=grid["Latin1InDegrees"], lon_0=lov, a=EARTH_RADIUS_M, b=EARTH_RADIUS_M)


def native_xy(grid: dict) -> tuple[np.ndarray, np.ndarray, pyproj.Proj]:
    """Projected x (Nx,) and y (Ny,) coordinates of native cell centers."""
    p = hrrr_proj(grid)
    x0, y0 = p(grid["longitudeOfFirstGridPointInDegrees"], grid["latitudeOfFirstGridPointInDegrees"])
    x = x0 + np.arange(grid["Nx"]) * grid["DxInMetres"]
    y = y0 + np.arange(grid["Ny"]) * grid["DyInMetres"]
    return x, y, p


def crop(a: np.ndarray, window: dict) -> np.ndarray:
    return a[window["j0"]:window["j1"] + 1, window["i0"]:window["i1"] + 1]


def coarsen(a: np.ndarray, f: int) -> np.ndarray:
    """Block mean over f x f cells; any NaN in a block makes that coarse cell NaN."""
    ny, nx = a.shape[0] // f, a.shape[1] // f
    return a[:ny * f, :nx * f].reshape(ny, f, nx, f).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def north_vector(p: pyproj.Proj, lon: float, lat: float) -> tuple[float, float]:
    """Unit vector of true north expressed in grid (x, y) components."""
    x0, y0 = p(lon, lat)
    x1, y1 = p(lon, lat + 1e-4)
    d = np.hypot(x1 - x0, y1 - y0)
    return (x1 - x0) / d, (y1 - y0) / d


def grid_to_earth(u_grid, v_grid, north: tuple[float, float]):
    nx, ny = north
    u_e = u_grid * ny - v_grid * nx
    v_e = u_grid * nx + v_grid * ny
    return u_e, v_e


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.229 * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- quantization

def quantize(x: np.ndarray, scale: float, offset: float) -> tuple[np.ndarray, int]:
    """float -> int16 with NaN -> INT16_MISSING. Returns (array, number of clipped values)."""
    q = np.round((x - offset) / scale)
    finite = np.isfinite(q)
    clipped = int(np.count_nonzero(finite & ((q > INT16_MAX) | (q < -INT16_MAX))))
    q = np.clip(np.where(finite, q, 0), -INT16_MAX, INT16_MAX).astype(np.int16)
    q[~finite] = INT16_MISSING
    return q, clipped


def dequantize(q: np.ndarray, scale, offset) -> np.ndarray:
    """int16 -> float32. Per-channel scale/offset vectors broadcast over (..., C, y, x)."""
    scale = np.asarray(scale, dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32)
    if scale.ndim == 1:
        scale = scale.reshape(-1, 1, 1)
        offset = offset.reshape(-1, 1, 1)
    x = q.astype(np.float32) * scale + offset
    x[q == INT16_MISSING] = np.nan
    return x


# --------------------------------------------------------------------------- store helpers

def load_grid_metadata() -> dict:
    path = METADATA_DIR / "grid_metadata.json"
    if not path.exists():
        raise SystemExit("metadata/grid_metadata.json missing: run scripts/inspect_archive.py first")
    return read_json(path)


def du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
