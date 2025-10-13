#!/usr/bin/env python3
"""
Aggregate K parquet files (parallel) and produce per-categorical-combo
1-D domains for each continuous covariate by expanding each sample by
per-dimension tolerances and merging resulting intervals.

Output:
- k_combo_domains.json : list of combos with decoded fields, sample_count,
  and per-covariate merged intervals (list of [min,max]).
- combo_counts.csv : counts per combo.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# must match your script
COV_NAMES = ["elevation","slope","access","fcc0_u","fcc0_d","fcc5_u","fcc5_d","fcc10_u","fcc10_d"]

# minimal fallback for luc column selection (or import your project's helper)
def luc_matching_columns(start_year: int, available: set | None):
    abs_names = (f"luc_{start_year}", f"luc_{start_year-5}", f"luc_{start_year-10}")
    rel_names = ("luc_0","luc_-5","luc_-10")
    if available is None:
        return abs_names
    if all(n in available for n in abs_names):
        return abs_names
    if all(n in available for n in rel_names):
        return rel_names
    chosen=[]
    for a,r in zip(abs_names, rel_names):
        chosen.append(a if a in available else (r if r in available else a))
    return tuple(chosen)

# covariate names and default per-dimension tolerances (same order)
DEFAULT_WIDTHS = np.array([200.0, 2.5, 10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)


def build_key(ecoregion: int, country: int, luc0: int, luc5: int, luc10: int) -> int:
    return (int(ecoregion) << 32) | (int(country) << 16) | (int(luc0) << 10) | (int(luc5) << 5) | int(luc10)


def decode_key(k: int) -> Tuple[int, int, int, int, int]:
    ecoregion = (k >> 32) & 0x7FFFFFFF
    country = (k >> 16) & 0xFFFF
    l0 = (k >> 10) & 0x1F
    l5 = (k >> 5) & 0x1F
    l10 = k & 0x1F
    return int(ecoregion), int(country), int(l0), int(l5), int(l10)


def process_file(args) -> Dict[int, List[Tuple[float, ...]]]:
    fpath, start_year, round_dp = args
    perfile: Dict[int, List[Tuple[float, ...]]] = defaultdict(list)
    try:
        # prefer column selection via pyarrow
        try:
            import pyarrow.parquet as pq  # type: ignore
            pqf = pq.ParquetFile(fpath)
            file_cols = set(pqf.schema.names)
            luc0, luc5, luc10 = luc_matching_columns(start_year, file_cols)
            # require all continuous covariates to be present in file
            missing_covs = [c for c in COV_NAMES if c not in file_cols]
            if missing_covs:
                print(f"Skipping {os.path.basename(fpath)} — missing covariates: {missing_covs}")
                return {}
            # require luc columns present
            for lc in (luc0, luc5, luc10):
                if lc not in file_cols:
                    print(f"Skipping {os.path.basename(fpath)} — missing luc column: {lc}")
                    return {}
            cols = ["ecoregion", "country", luc0, luc5, luc10] + [c for c in COV_NAMES if c in file_cols]
            if not cols:
                return {}
            df = pd.read_parquet(fpath, columns=cols, engine="pyarrow")
        except Exception:
            df = pd.read_parquet(fpath)
            file_cols = set(df.columns)
            luc0, luc5, luc10 = luc_matching_columns(start_year, file_cols)
            missing_covs = [c for c in COV_NAMES if c not in file_cols]
            if missing_covs:
                print(f"Skipping {os.path.basename(fpath)} — missing covariates: {missing_covs}")
                return {}
            for lc in (luc0, luc5, luc10):
                if lc not in file_cols:
                    print(f"Skipping {os.path.basename(fpath)} — missing luc column: {lc}")
                    return {}
            cols = ["ecoregion", "country", luc0, luc5, luc10] + [c for c in COV_NAMES if c in file_cols]
            df = df[cols]

        # downcast and round
        present_cov = [c for c in COV_NAMES if c in df.columns]
        for c in present_cov:
            df[c] = df[c].astype(np.float32, copy=False)
        if "ecoregion" in df.columns:
            df["ecoregion"] = df["ecoregion"].astype(np.int32, copy=False)
        if "country" in df.columns:
            df["country"] = df["country"].astype(np.uint16, copy=False)
        for c in [luc0, luc5, luc10]:
            if c in df.columns:
                df[c] = df[c].astype(np.uint8, copy=False)

        if present_cov and round_dp >= 0:
            df[present_cov] = df[present_cov].round(round_dp).astype(np.float32, copy=False)

        dedupe_cols = ["ecoregion", "country", luc0, luc5, luc10] + present_cov
        dedupe_cols = [c for c in dedupe_cols if c in df.columns]
        if dedupe_cols:
            df = df.drop_duplicates(subset=dedupe_cols, ignore_index=True)

        for r in df.itertuples(index=False, name=None):
            row = dict(zip(df.columns, r))
            try:
                ecoregion = int(row["ecoregion"]); country = int(row["country"])
                l0 = int(row[luc0]); l5 = int(row[luc5]); l10 = int(row[luc10])
            except Exception:
                continue
            key = build_key(ecoregion, country, l0, l5, l10)
            cov = []
            skip = False
            for c in COV_NAMES:
                if c in df.columns:
                    v = row[c]
                    if pd.isna(v):
                        skip = True; break
                    cov.append(float(v))
                else:
                    skip = True; break
            if skip:
                continue
            perfile[key].append(tuple(cov))
    except Exception:
        return {}
    return perfile


def merge_maps(maps: List[Dict[int, List[Tuple[float, ...]]]]) -> Dict[int, List[Tuple[float, ...]]]:
    agg: Dict[int, List[Tuple[float, ...]]] = defaultdict(list)
    for m in maps:
        for k, vals in m.items():
            agg[k].extend(vals)
    # dedupe per key
    for k in list(agg.keys()):
        if not agg[k]:
            del agg[k]; continue
        agg[k] = list({tuple(x) for x in agg[k]})
    return agg


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[List[float]]:
    if not intervals:
        return []
    # sort by start
    intervals_sorted = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = intervals_sorted[0]
    for a, b in intervals_sorted[1:]:
        if a <= cur_end:
            cur_end = max(cur_end, b)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = a, b
    merged.append((cur_start, cur_end))
    # convert to lists for JSON
    return [[float(a), float(b)] for a, b in merged]


def build_domains(agg_map: Dict[int, List[Tuple[float, ...]]], widths: np.ndarray) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    # covariates that must be non-negative (lower bound zero)
    _nonneg_covs = {"slope", "access", "fcc0_u", "fcc0_d", "fcc5_u", "fcc5_d", "fcc10_u", "fcc10_d"}
    for key, vals in tqdm(list(agg_map.items()), desc="Building domains"):
        arr = np.asarray(vals, dtype=float)  # shape (N, D)
        if arr.size == 0:
            continue
        sample_count = int(arr.shape[0])
        cov_intervals: Dict[str, List[List[float]]] = {}
        for j, cname in enumerate(COV_NAMES):
            col_vals = arr[:, j]
            intervals = []
            w = float(widths[j])
            for v in col_vals:
                lo = v - w
                hi = v + w
                if cname in _nonneg_covs:
                    lo = max(lo, 0.0)
                intervals.append((lo, hi))
            merged = merge_intervals(intervals)
            cov_intervals[cname] = merged
        out[int(key)] = {"sample_count": sample_count, "intervals": cov_intervals}
    return out


def write_outputs(domains: Dict[int, dict], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    decoded = []
    for k, v in domains.items():
        ecoregion, country, l0, l5, l10 = decode_key(k)
        decoded.append({
            "key": k,
            "ecoregion": ecoregion,
            "country": country,
            "luc0": l0,
            "luc5": l5,
            "luc10": l10,
            "sample_count": v["sample_count"],
            "intervals": v["intervals"],
        })
    out_json = os.path.join(out_dir, "k_combo_1d_domains.json")
    with open(out_json, "w") as fh:
        json.dump(decoded, fh, indent=2)
    rows = [(d["key"], d["ecoregion"], d["country"], d["luc0"], d["luc5"], d["luc10"], d["sample_count"]) for d in decoded]
    df = pd.DataFrame(rows, columns=["key","ecoregion","country","luc0","luc5","luc10","sample_count"])
    df.to_csv(os.path.join(out_dir, "combo_counts.csv"), index=False)
    return out_json


def parse_widths(s: str | None) -> np.ndarray:
    if s is None:
        return DEFAULT_WIDTHS
    parts = [p.strip() for p in s.split(",")]
    vals = [float(p) for p in parts]
    if len(vals) != len(COV_NAMES):
        raise ValueError(f"--widths must have {len(COV_NAMES)} comma-separated values")
    return np.array(vals, dtype=float)


def main():
    p = argparse.ArgumentParser(description="Build 1-D domains per covariate from K samples")
    p.add_argument("--k", required=True, help="Directory with k_*.parquet files")
    p.add_argument("--start-year", required=True, type=int)
    p.add_argument("--out", required=True, help="Output directory for JSON/CSV")
    # round continuous covariates to 4 decimal places before dedupe (as requested)
    p.add_argument("--round-dp", type=int, default=4, help="Round continuous values to this many decimals before dedupe")
    p.add_argument("--widths", type=str, default=None, help="Comma-separated per-cov width values (same order as COV_NAMES) e.g. '200,2.5,10,0.1,...'")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.k, "k_*.parquet")))
    if not files:
        raise SystemExit("No k_*.parquet files found in --k directory")

    widths = parse_widths(args.widths)
    tasks = [(f, args.start_year, args.round_dp) for f in files]
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks), desc="Processing files"))

    agg = merge_maps(results)
    domains = build_domains(agg, widths=widths)
    out_json = write_outputs(domains, args.out)
    print("Wrote 1-D domains JSON:", out_json)


if __name__ == "__main__":
    main()