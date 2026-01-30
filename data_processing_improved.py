"""
Process mining feature builder for XES logs (pm4py):
- Import XES with normalization + caching
- Build three feature types:
  1) Concurrency (3 variants)
  2) Resource utilization (2 variants)
  3) Throughput time (row/span/rolling/none)
- Store intermediate results via pickle cache

Notes:
- Concurrency variants are implemented as separate methods (as requested).
- Plotting is kept outside the builders.
- Uses vectorized pandas operations where possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Literal, Union

import pickle
import numpy as np
import pandas as pd
import pm4py
import matplotlib.pyplot as plt


# -----------------------------
# Configuration + Constants
# -----------------------------

CASE_COL = "case:concept:name"
TIME_COL = "time:timestamp"
RES_COL = "org:resource"


@dataclass(frozen=True)
class Config:
    root: Path
    dataset: str
    cache_dir: Path = Path("tmp/pickle")
    utc: bool = True  # enforce UTC timestamps


def _dataset_path(cfg: Config) -> Path:
    return cfg.root / cfg.dataset


def _cache_path(cfg: Config, name: str) -> Path:
    return cfg.root / cfg.cache_dir / name


# -----------------------------
# Caching helper
# -----------------------------

def load_or_compute(
    cache_path: Path,
    compute_fn: Callable[[], object],
    from_file: bool,
) -> object:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if from_file and cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)
    result = compute_fn()
    with cache_path.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result


# -----------------------------
# Import XES
# -----------------------------

def import_xes(cfg: Config, from_file: bool = False, time_col: str = TIME_COL, case_col: str = CASE_COL) -> pd.DataFrame:
    """
    Imports an XES into a pandas DataFrame via pm4py and normalizes timestamps.
    Caches the resulting DataFrame.

    Improvements vs original:
    - No fragile df.iloc[:, i] overwrites
    - Enforces datetime parsing robustly (+ optional UTC)
    - Centralized caching with directory creation
    """
    cache = _cache_path(cfg, "raw_data.pkl")
    xes_path = _dataset_path(cfg)

    def _compute() -> pd.DataFrame:
        df = pm4py.read_xes(str(xes_path))

        if time_col not in df.columns:
            raise KeyError(f"Expected column '{time_col}' not found in XES dataframe columns: {list(df.columns)}")
        if case_col not in df.columns:
            raise KeyError(f"Expected column '{case_col}' not found in XES dataframe columns: {list(df.columns)}")

        # Normalize timestamp
        df[time_col] = pd.to_datetime(df[time_col], utc=cfg.utc, errors="coerce")
        if df[time_col].isna().any():
            bad = df[df[time_col].isna()].head(5)
            raise ValueError(
                f"Some timestamps could not be parsed (showing up to 5 rows):\n{bad.to_string(index=False)}"
            )

        return df

    return load_or_compute(cache, _compute, from_file=from_file)  # type: ignore


# -----------------------------
# First/Last per case (vectorized)
# -----------------------------

def build_first_last_df(cfg: Config, data: pd.DataFrame, from_file: bool = False, time_col: str = TIME_COL, case_col: str = CASE_COL) -> pd.DataFrame:
    """
    Returns DataFrame with columns: ['name', 'first', 'last'].

    Improvements:
    - Vectorized groupby min/max instead of per-case loops
    - Correct typing and caching
    """
    cache = _cache_path(cfg, "first_last.pkl")

    def _compute() -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame(columns=["name", "first", "last"])

        g = data.groupby(case_col)[time_col]
        out = g.agg(first="min", last="max").reset_index()
        out = out.rename(columns={case_col: "name"})
        return out

    return load_or_compute(cache, _compute, from_file=from_file)  # type: ignore


# -----------------------------
# Plot helper (kept separate)
# -----------------------------

def plot_series(ts: pd.Series, title: str = "") -> None:
    ts.plot(figsize=(10, 5), title=title or ts.name or "Time series")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


# -----------------------------
# Concurrency (3 variants)
# -----------------------------
# Your original code conceptually does:
# active case if ts in [first, last).
#
# Variant A: hourly grid + sweep-line (fast, good for forecasting)
# Variant B: exact change-points at first/last (step function on irregular index)
# Variant C: event-time sampling (compute at each event timestamp, then optionally resample)

ConcurrencyVariant = Literal["hourly_sweepline", "exact_changepoints", "event_sampled"]


def build_concurrency_series(cfg: Config, data: pd.DataFrame, from_file: bool = False, variant: ConcurrencyVariant = "hourly_sweepline", freq: str = "1H", keep_datetime_index: bool = True, time_col: str = TIME_COL, case_col: str = CASE_COL) -> pd.Series:
    cache = _cache_path(cfg, f"concurrency_{variant}_{freq.replace('/', '-')}.pkl")
    #print(data.sort_values(by=["time:timestamp"]))
    def _compute() -> pd.Series:
        #print("COMPUTE")
        fl = build_first_last_df(cfg, data, from_file=from_file, time_col=time_col, case_col=case_col)
        if fl.empty:
            return pd.Series(dtype="float64", name="concurrent_cases")

        if variant == "hourly_sweepline":
            ts = _concurrency_hourly_sweepline(fl, freq=freq)
        elif variant == "exact_changepoints":
            ts = _concurrency_exact_changepoints(fl)
        elif variant == "event_sampled":
            ts = _concurrency_event_sampled(data, fl, time_col=time_col)
        else:
            raise ValueError(f"Unknown concurrency variant: {variant}")

        ts.name = "concurrent_cases"
        
        if not keep_datetime_index:
            ts = ts.reset_index(drop=True)
        print(ts)
        return ts

    return load_or_compute(cache, _compute, from_file=from_file)  # type: ignore


def _concurrency_hourly_sweepline(first_last: pd.DataFrame, freq: str = "1H") -> pd.Series:
    """
    Fast hourly concurrency using start/end counts and cumulative sum.
    Implements active in [first, last) on an hourly grid.

    Steps:
    - Floor starts to hour bucket
    - Ceil ends to hour bucket (so the case stops contributing at 'end' bucket)
    - count(+1) at start bucket, count(-1) at end bucket, cumsum
    """
    fl = first_last.copy()
    fl["first_bucket"] = pd.to_datetime(fl["first"]).dt.floor(freq)
    fl["last_bucket"] = pd.to_datetime(fl["last"]).dt.ceil(freq)

    start = fl["first_bucket"].min()
    end = fl["last_bucket"].max()
    idx = pd.date_range(start=start, end=end, freq=freq)

    starts = fl["first_bucket"].value_counts().reindex(idx, fill_value=0)
    ends = fl["last_bucket"].value_counts().reindex(idx, fill_value=0)

    conc = (starts - ends).cumsum().astype(float)
    return conc


def _concurrency_exact_changepoints(first_last: pd.DataFrame) -> pd.Series:
    """
    Exact concurrency as a step function on irregular timestamps.

    Creates a series at change points (all 'first' and 'last') and returns a
    stepwise constant concurrency level *after* applying deltas at each point,
    matching active in [first, last).

    Usage:
    - Good for accuracy, but the time index is irregular.
    - You can resample afterwards (e.g., .resample('1H').ffill()).
    """
    fl = first_last.copy()
    starts = pd.Series(1, index=pd.to_datetime(fl["first"]))
    ends = pd.Series(-1, index=pd.to_datetime(fl["last"]))

    deltas = pd.concat([starts, ends]).groupby(level=0).sum().sort_index()
    conc = deltas.cumsum().astype(float)
    conc.index = pd.to_datetime(conc.index)
    return conc


def _concurrency_event_sampled(data: pd.DataFrame, first_last: pd.DataFrame, time_col: str = TIME_COL) -> pd.Series:
    """
    Concurrency computed at each *event timestamp* (sampled at event times).
    This matches the spirit of your original approach but avoids O(C*T) loops.

    Approach:
    - Build exact change-point concurrency (fast)
    - Evaluate it at each unique event time (forward-fill)
    """
    event_times = pd.to_datetime(data[time_col]).dropna().drop_duplicates().sort_values()
    if event_times.empty:
        return pd.Series(dtype="float64")

    cp = _concurrency_exact_changepoints(first_last)

    # We want concurrency value at event times. cp is already a step function at change points.
    # Reindex on union -> ffill
    idx = event_times
    out = cp.reindex(cp.index.union(idx)).sort_index().ffill().reindex(idx)
    out = out.fillna(0.0).astype(float)
    return out


# -----------------------------
# Resource utilization (2 variants)
# -----------------------------
# Variant A: "observed_resources_per_bin" (your original: distinct resources that appear in that hour)
# Variant B: "event_rate_per_bin" (events per hour normalized by total resources)

ResourceVariant = Literal["observed_resources_per_bin", "event_rate_per_bin"]


def build_resource_utilization_series(cfg: Config, data: pd.DataFrame, from_file: bool = False, variant: ResourceVariant = "observed_resources_per_bin", freq: str = "1H", smoothing: bool = True, res_col: str = RES_COL, time_col: str = TIME_COL) -> pd.Series:
    cache = _cache_path(cfg, f"resource_{variant}_{freq.replace('/', '-')}_smooth{int(smoothing)}.pkl")

    def _compute() -> pd.Series:
        if data.empty:
            return pd.Series(dtype="float64", name="resource_utilization")

        if res_col not in data.columns:
            raise KeyError(f"Expected column '{res_col}' not found in data columns: {list(data.columns)}")

        total_resources = data[res_col].nunique(dropna=True)
        if total_resources == 0:
            return pd.Series(dtype="float64", name="resource_utilization")

        df = data[[time_col, res_col]].dropna().copy()
        df[time_col] = pd.to_datetime(df[time_col])

        if variant == "observed_resources_per_bin":
            # Count distinct resources observed per time bin, normalize by total resources
            util = (
                df.set_index(time_col)
                  .groupby(pd.Grouper(freq=freq))[res_col]
                  .nunique()
                  .astype(float) / float(total_resources)
            )
        elif variant == "event_rate_per_bin":
            # Events per bin / total_resources (rough workload proxy)
            util = (
                df.set_index(time_col)
                  .groupby(pd.Grouper(freq=freq))[res_col]
                  .count()
                  .astype(float) / float(total_resources)
            )
        else:
            raise ValueError(f"Unknown resource utilization variant: {variant}")

        util.name = "resource_utilization"

        # Fill missing bins
        if smoothing:
            util = util.asfreq(freq).ffill().fillna(0.0)
        else:
            util = util.asfreq(freq, fill_value=0.0)

        return util

    return load_or_compute(cache, _compute, from_file=from_file)  # type: ignore


# -----------------------------
# Throughput time
# -----------------------------

TTMethod = Literal["row", "span", "rolling", "none"]


def build_throughput_time_series(cfg: Config, data: pd.DataFrame, from_file: bool = False, method: TTMethod = "row", method_param: Union[int, str] = 100, freq: str = "1H", smoothing: bool = True, time_col: str = TIME_COL, case_col: str = CASE_COL) -> pd.Series:  # row group size or rolling window or resample freq ,

    cache = _cache_path(cfg, f"throughput_{method}_{str(method_param)}_{freq.replace('/', '-')}_smooth{int(smoothing)}.pkl")

    def _compute() -> pd.Series:
        fl = build_first_last_df(cfg, data, from_file=from_file, time_col=time_col, case_col=case_col)
        if fl.empty:
            return pd.Series(dtype="float64", name="throughput_time")

        fl = fl.copy()
        fl["throughput_time"] = (pd.to_datetime(fl["last"]) - pd.to_datetime(fl["first"])).dt.total_seconds()

        # Sort by completion time
        base = fl[["last", "throughput_time"]].sort_values(by="last").reset_index(drop=True)

        if method == "row":
            ts = _aggregate_tt_row_number(base, group_size=int(method_param))
        elif method == "span":
            ts = _aggregate_tt_timespan(base, time_length=str(method_param))
        elif method == "rolling":
            ts = _aggregate_tt_rolling(base, window=int(method_param))
        elif method == "none":
            ts = base.set_index("last")["throughput_time"].astype(float)
        else:
            raise ValueError(f"Unknown throughput aggregation method: {method}")

        ts.index = pd.to_datetime(ts.index)
        ts.name = "throughput_time"

        # Reindex to regular grid for forecasting
        idx = pd.date_range(
            start=ts.index.min().floor(freq),
            end=ts.index.max().ceil(freq),
            freq=freq
        )

        if smoothing:
            # IMPORTANT: use method='ffill' in reindex so it works even when ts has minutes/seconds
            out = ts.sort_index().reindex(idx, method="ffill").fillna(0.0)
        else:
            # if no smoothing, bins with no exact timestamp stay 0
            out = ts.sort_index().reindex(idx).fillna(0.0)

        return out.astype(float)

    return load_or_compute(cache, _compute, from_file=from_file)  # type: ignore


def _aggregate_tt_row_number(data: pd.DataFrame, group_size: int = 10) -> pd.Series:
    d = data.copy()
    d["group"] = d.index // group_size
    agg = d.groupby("group").agg(last=("last", "max"), throughput_time=("throughput_time", "mean"))
    return agg.set_index("last")["throughput_time"].astype(float)


def _aggregate_tt_timespan(data: pd.DataFrame, time_length: str = "1D") -> pd.Series:
    d = data.copy()
    d = d.set_index("last")
    agg = d["throughput_time"].resample(time_length).mean()
    return agg.astype(float)


def _aggregate_tt_rolling(data: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Rolling mean over last `window` completed cases, indexed by completion time.
    """
    d = data.copy()
    d["throughput_time"] = d["throughput_time"].rolling(window=window, min_periods=1).mean()
    return d.set_index("last")["throughput_time"].astype(float)


# -----------------------------
# Optional: DFG visualization
# -----------------------------

def view_dfg(cfg: Config, data: Optional[pd.DataFrame] = None, from_file: bool = True) -> None:
    if data is None:
        data = import_xes(cfg, from_file=from_file)
    dfg = pm4py.discover_dfg(data)
    pm4py.view_dfg(dfg[0], dfg[1], dfg[2])


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":
    cfg = Config(
        root=Path("/Volumes/Daniel/Thesis/resources"),
        dataset="BPI_2012/BPI_Challenge_2012.xes",
        cache_dir=Path.home() / "tmp" / "pm4py_cache",
        utc=True,
    )
    split_timestamp = "2012-02-18 18:00:00+00:00"
    df = import_xes(cfg, from_file=True)

    # --- Concurrency (choose variant) ---
    conc_hourly = build_concurrency_series(cfg, df, from_file=False, variant="hourly_sweepline", freq="1H")
    # conc_exact = build_concurrency_series(cfg, df, from_file=True, variant="exact_changepoints")
    # conc_event = build_concurrency_series(cfg, df, from_file=True, variant="event_sampled")

    # --- Resource utilization (choose variant) ---
    res_util = build_resource_utilization_series(cfg, df, from_file=False, variant="observed_resources_per_bin", freq="1H", smoothing=True)

    # --- Throughput time ---
    tt = build_throughput_time_series(cfg, df, from_file=False, method="rolling", method_param=100, freq="1H")

    print(conc_hourly.head())
    print(res_util.head())
    print(tt.head())

    # Plot examples
    plot_series(conc_hourly, "Concurrency (hourly sweep-line)")
    plot_series(res_util, "Resource utilization (observed resources / hour)")
    plot_series(tt, "Throughput time (rolling mean, hourly grid)")
