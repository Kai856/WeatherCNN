"""Download only the required GRIB2 messages for each valid hour (HTTP Range requests, no credentials).

For valid hour v the messages come from:
  wrfsfcf00 init v   (analysis; preferred source for every f00 channel)
  wrfprsf00 init v   (analysis; only for channels absent from wrfsfc)
  wrfsfcf01 init v-1 (APCP 0-1 hour accumulation, valid at v)

Each hour is written to <data_root>/tmp/YYYYMMDDHH/<channel>.grib2 plus manifest.json. The manifest is
written last, so an hour directory without a complete manifest is re-downloaded. Downloads are idempotent.

Usage
  python scripts/download_hrrr.py --start 2025-07-14T00 --end 2025-07-14T23
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from common import (IdxEntry, channel_sources, fetch_entries, fetch_idx, file_key, file_url, hourly_range,
                    load_config, make_session, setup_logging)

log = setup_logging("download_hrrr")


def hour_dir(cfg: dict, v: pd.Timestamp) -> Path:
    return cfg["tmp_path"] / f"{v:%Y%m%d%H}"


def plan_hour(cfg: dict, session, v: pd.Timestamp, include_static: bool = False, inputs: bool = True) -> dict:
    idx_cache: dict[tuple, list | None] = {}

    def entries_for(prod, init, fhr):
        k = (prod, init, fhr)
        if k not in idx_cache:
            idx_cache[k] = fetch_idx(session, file_url(cfg, init, prod, fhr), cfg["download"]["timeout_s"])
        return idx_cache[k]

    wanted = []
    if inputs:
        wanted += [(c["name"], "input", c) for c in cfg["channels"]]
    if include_static:
        wanted += [(s["name"], "static", {**s, "source": "f00"}) for s in cfg["static_fields"]]

    items, missing = [], []
    for name, kind, ch in wanted:
        chosen = None
        for prod, init, fhr, desc in channel_sources(ch, v):
            ents = entries_for(prod, init, fhr)
            match = [e for e in (ents or []) if e.var == ch["var"] and e.level == ch["level"] and e.desc == desc]
            if match:
                e = match[0]
                chosen = {"name": name, "kind": kind, "product": f"wrf{prod}f{fhr:02d}", "init_time": str(init),
                          "lead_h": fhr, "idx_desc": desc, "url": file_url(cfg, init, prod, fhr),
                          "source_file": file_key(init, prod, fhr, cfg["archive"]["domain"]),
                          "msg_num": e.num, "start": e.start, "end": e.end, "n_matches": len(match)}
                break
        if chosen:
            items.append(chosen)
        else:
            reasons = []
            for prod, init, fhr, desc in channel_sources(ch, v):
                ents = idx_cache.get((prod, init, fhr))
                reasons.append(f"{file_key(init, prod, fhr)}: {'idx missing' if ents is None else 'message absent'}")
            missing.append({"name": name, "kind": kind, "reason": "; ".join(reasons)})
    return {"valid_time": str(v), "items": items, "missing": missing}


def download_hour(cfg: dict, session, v: pd.Timestamp, include_static: bool = False, inputs: bool = True,
                  out_dir: Path | None = None) -> Path:
    d = out_dir or hour_dir(cfg, v)
    man_path = d / "manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text())
        if man.get("complete") and (not include_static or any(i["kind"] == "static" for i in man["items"])):
            return d
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    man = plan_hour(cfg, session, v, include_static=include_static, inputs=inputs)

    by_url = defaultdict(list)
    for it in man["items"]:
        by_url[it["url"]].append(it)

    def fetch(url_items):
        url, its = url_items
        ents = [IdxEntry(i["msg_num"], i["start"], i["end"], "", "", "") for i in its]
        data = fetch_entries(session, url, ents, cfg["download"]["timeout_s"])
        for i in its:
            (d / f"{i['name']}.grib2").write_bytes(data[i["msg_num"]])
            i["bytes"] = len(data[i["msg_num"]])

    jobs = []
    for url, its in by_url.items():
        its = sorted(its, key=lambda i: i["start"])
        n = max(1, min(cfg["download"]["threads_per_hour"], len(its)))
        size = -(-len(its) // n)
        jobs += [(url, its[k:k + size]) for k in range(0, len(its), size)]
    with ThreadPoolExecutor(cfg["download"]["threads_per_hour"]) as ex:
        list(ex.map(fetch, jobs))

    man["complete"] = True
    man["total_bytes"] = sum(i.get("bytes", 0) for i in man["items"])
    tmp = man_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=1))
    tmp.replace(man_path)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--static", action="store_true", help="also fetch static fields (orography, land)")
    args = ap.parse_args()
    cfg = load_config()
    session = make_session(cfg["download"]["retries"])
    for v in hourly_range(args.start, args.end):
        d = download_hour(cfg, session, v, include_static=args.static)
        man = json.loads((d / "manifest.json").read_text())
        log.info("%s: %d messages, %.1f MB, %d missing", v, len(man["items"]), man["total_bytes"] / 1e6, len(man["missing"]))


if __name__ == "__main__":
    main()
