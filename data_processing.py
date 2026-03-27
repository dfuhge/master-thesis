"""Data Processing Pipeline for Creating Process KPIs

This module provides functions to import event logs in XES or CSV format, compute Process KPIs on Concurrent Cases, Resource Utilization, and Throughput Time, and plot the resulting time series.
It uses simple caching to store intermediate results. For the KPIs, different variants are implemented.

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
import argparse


# Default column names
CASE_COL = "case:concept:name"
TIME_COL = "time:timestamp"
RES_COL = "org:resource"
ACT_COL = "concept:name"


@dataclass(frozen=True)
class Config:
    """
    Dataset and Cache configuration.
    
    - dataset: absolute or relative path to XES file from root
    - cache_dir: directory to store intermediate pickled results
    - utc: whether to enforce UTC timestamps when parsing (default: True)
    - case_col, res_col, act_col, time_col: name of columns for case ID, resoruce, activity/task and timestamp
    """
    dataset: Path
    cache_dir: Path = Path("tmp/pickle")
    utc: bool = True
    case_col: str = CASE_COL
    res_col: str = RES_COL
    act_col: str = ACT_COL
    time_col: str = TIME_COL



# ---------- Utility functions ----------

def load_or_compute(cache_path: Path, compute_fn: Callable[[], object], load: bool) -> object:
    """
    Checks if cache should be loaded, otherwise computes result and saves to cache.

    - cache_path: Path to cache file
    - compute_fn: Function that computes the result if cache is not used
    - load: Whether to load from cache if it exists
    """
    # Ensure cache directory exists
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # If load, load the cache file and return contents
    if load and cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)
    # Apply function
    result = compute_fn()
    # Save result to cache
    with cache_path.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result

def import_data(cfg: Config, load: bool = False) -> pd.DataFrame:
    """
    Imports an event log (XES/CSV) into a pandas DataFrame via pm4py and normalizes timestamps.
    Caches the resulting DataFrame.

    - cfg: Config object with dataset path and cache settings
    - load: Whether to load from cache if it exists (default: False)
    """
    # Create cache path based on config
    cache = cfg.cache_dir / "raw_data.pkl"
    # Get dataset path from config
    path = cfg.dataset

    def _compute() -> pd.DataFrame:
        # Load event log according to suffix
        suffix = path.suffix.lower()
        if suffix == ".xes":
            df = pm4py.read_xes(str(path))
        elif suffix == ".csv":
            df = pd.read_csv(path)

        # Check columns exist
        if cfg.time_col not in df.columns:
            raise KeyError(f"Expected column '{cfg.time_col}' not found in XES dataframe columns: {list(df.columns)}")
        if cfg.case_col not in df.columns:
            raise KeyError(f"Expected column '{cfg.case_col}' not found in XES dataframe columns: {list(df.columns)}")
        if cfg.res_col not in df.columns:
            raise KeyError(f"Expected column '{cfg.res_col}' not found in XES dataframe columns: {list(df.columns)}")
        if cfg.act_col not in df.columns:
            raise KeyError(f"Expected column '{cfg.act_col}' not found in XES dataframe columns: {list(df.columns)}")
        
        # Normalize timestamp
        df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], utc=cfg.utc, errors="coerce")
        # Enforce timestamps are not NaN after parsing
        if df[cfg.time_col].isna().any():
            bad = df[df[cfg.time_col].isna()].head(5)
            raise ValueError(
                f"Some timestamps could not be parsed (showing up to 5 rows):\n{bad.to_string(index=False)}"
            )
        return df

    # Load or compute XES file
    return load_or_compute(cache, _compute, load=load)

def build_first_last_df(cfg: Config, data: pd.DataFrame, load: bool = False) -> pd.DataFrame:
    """
    Returns DataFrame with columns: ['name', 'first', 'last'].

     - cfg: Config object with cache settings
     - data: input DataFrame
     - load: Whether to load from cache if it exists
    """

    # Create cache path based on config
    cache = cfg.cache_dir / "first_last.pkl"

    def _compute() -> pd.DataFrame:
        # Check if data empty, then return empty DataFrame with columns
        if data.empty:
            return pd.DataFrame(columns=["name", "first", "last"])
        # Group data by case ID
        g = data.groupby(cfg.case_col)[cfg.time_col]
        # Aggregate to get first and last timestamp per case
        out = g.agg(first="min", last="max").reset_index()
        # Rename case_col to "name"
        out = out.rename(columns={cfg.case_col: "name"})

        return out

    # Load or compute first/last DataFrame
    return load_or_compute(cache, _compute, load=load)

def regularize_series(ts: pd.Series, freq: str = "1D", name: str = "", smoothing: bool = True) -> pd.Series:
    """
    Regularizes a time series to a specified frequency, optionally applying forward-fill smoothing.

    - ts: time series to regularize
    - freq: frequency for regularization
    - name: name for time series
    - smoothing: if true, applies forward-fill for missing values
    """
    ts.index = pd.to_datetime(ts.index)
    ts.name = name

    # Reindex to regular grid for forecasting
    idx = pd.date_range(
        start=ts.index.min().floor(freq),
        end=ts.index.max().ceil(freq),
        freq=freq
    )
    
    if smoothing:
        out = ts.sort_index().reindex(idx, method="ffill").fillna(0.0)
    else:
        # if no smoothing, bins with no exact timestamp stay 0
        out = ts.sort_index().reindex(idx).fillna(0.0)

    return out.astype(float)

def plot_series(ts: pd.Series, title: str = "", y="", x="") -> None:
    """
    Plot series with optional title.

    - ts: pandas Series to plot
    - title: optional title for the plot
    - y: label for y-axis
    - x: label for x-axis
    """
    ts.plot(figsize=(10, 5), title=title or ts.name or "Time series")
    plt.xlabel(x, labelpad=100)
    plt.ylabel(y)
    plt.grid(True)
    plt.show()

def view_dfg(cfg: Config, data: Optional[pd.DataFrame] = None, load: bool = True) -> None:
    """
    Visualizes Directly-Follows Graph of the data using pm4py.

    - cfg: Config object for dataset
    - data: optional pre-loaded dataframe
    - load: if true, will enable caching
    """
    if data is None:
        data = import_data(cfg, load=load)
    dfg = pm4py.discover_dfg(data)
    pm4py.view_dfg(dfg[0], dfg[1], dfg[2])



# ---------- Concurrent Cases KPI ----------
# Computation Variants for Concurrent Cases KPI
Concurrent_Cases_Variant = Literal["sweepline", "exact_changepoints", "event_sampled"]

def build_concurrent_cases_series(cfg: Config, data: pd.DataFrame, load: bool = False, variant: Concurrent_Cases_Variant = "sweepline", freq: str = "1D", smoothing: bool = True) -> pd.Series:
    """
    Creates a Concurrent Cases KPI series based on the specified variant.

    - cfg: Config object with cache settings
    - data: input DataFrame
    - load: Whether to load from cache if it exists
    - variant: method to compute concurrency ("hourly_sweepline", "exact_changepoints", "event_sampled")
    - freq: frequency for hourly_sweepline variant
    - smoothing: whether to apply forward-fill smoothing
    """
    # Create cache path based on config and variant
    cache = cfg.cache_dir / f"concurrency_{variant}_{freq.replace('/', '-')}.pkl"

    def _compute() -> pd.Series:
        # Build first/last DataFrame. If empty, return empty series with name.
        fl = build_first_last_df(cfg, data, load=load)
        if fl.empty:
            return pd.Series(dtype="float64", name="concurrent_cases")

        # Compute concurrency based on variant
        if variant == "sweepline":
            ts = _concurrent_cases_sweepline(fl, freq=freq)
        elif variant == "exact_changepoints":
            ts = _concurrent_cases_exact_changepoints(fl)
        elif variant == "event_sampled":
            ts = _concurrent_cases_event_sampled(data, fl, time_col=cfg.time_col)
        else:
            raise ValueError(f"Unknown concurrency variant: {variant}")

        # Regularize series
        ts = regularize_series(ts, freq=freq, name="concurrent_cases", smoothing=smoothing)
        
        return ts

    return load_or_compute(cache, _compute, load=load)

def _concurrent_cases_sweepline(first_last: pd.DataFrame, freq: str = "1D") -> pd.Series:
    """
    Fast hourly concurrency using start/end counts and cumulative sum.
    Implements active in [first, last) on an hourly grid.

    - first_last: first/last DataFrame
    - freq: frequency for the result series
    """
    fl = first_last.copy()
    # Ceil timestamps to frequency
    fl["first_bucket"] = pd.to_datetime(fl["first"]).dt.ceil(freq)
    fl["last_bucket"] = pd.to_datetime(fl["last"]).dt.ceil(freq)
    # Build date range
    start = fl["first_bucket"].min()
    end = fl["last_bucket"].max()
    idx = pd.date_range(start=start, end=end, freq=freq)
    # Count starts and ends
    starts = fl["first_bucket"].value_counts().reindex(idx, fill_value=0)
    ends = fl["last_bucket"].value_counts().reindex(idx, fill_value=0)
    # Cumulate
    conc = (starts - ends).cumsum().astype(float)
    return conc

def _concurrent_cases_exact_changepoints(first_last: pd.DataFrame) -> pd.Series:
    """
    Exact concurrency as a step function on irregular timestamps.

    Creates a series at change points (all 'first' and 'last') and returns a
    stepwise constant concurrency level *after* applying deltas at each point,
    matching active in [first, last).

    - first_last: firt/last DataFrame
    """
    fl = first_last.copy()
    # Get all starts and ends
    starts = pd.Series(1, index=pd.to_datetime(fl["first"]))
    ends = pd.Series(-1, index=pd.to_datetime(fl["last"]))
    # Get changepoint deltas
    deltas = pd.concat([starts, ends]).groupby(level=0).sum().sort_index()
    conc = deltas.cumsum().astype(float)
    conc.index = pd.to_datetime(conc.index)
    return conc

def _concurrent_cases_event_sampled(data: pd.DataFrame, first_last: pd.DataFrame, time_col: str = TIME_COL) -> pd.Series:
    """
    Concurrency computed at each *event timestamp* (sampled at event times).

    - data: event log DataFrame
    first_last: first/last DataFrame
    time_col: time column name
    """
    event_times = pd.to_datetime(data[time_col]).dropna().drop_duplicates().sort_values()
    if event_times.empty:
        return pd.Series(dtype="float64")

    cp = _concurrent_cases_exact_changepoints(first_last)

    idx = event_times
    out = cp.reindex(cp.index.union(idx)).sort_index().ffill().reindex(idx)
    out = out.fillna(0.0).astype(float)
    return out


# ---------- Resource Utilization KPI ----------

def build_resource_utilization_series(cfg: Config, data: pd.DataFrame, load: bool = False, freq: str = "1D", smoothing: bool = False, total_res: int = None) -> pd.Series:
    """
    Creates a Resource Utilization KPI series.

    - cfg: Config object with cache settings
    - data: input DataFrame
    - load: Whether to load from cache if it exists
    - freq: frequency for resampling
    - smoothing: If true, applies forward-fill for missing values
    - total_res: total number of resources
    """

    # Create cache path based on config and variant
    cache = cfg.cache_dir / f"resource_{freq.replace('/', '-')}.pkl"

    def _compute() -> pd.Series:
        # Check if data empty, then return empty series with name.
        if data.empty:
            return pd.Series(dtype="float64", name="resource_utilization")
        # Check columns exist
        if cfg.res_col not in data.columns:
            raise KeyError(f"Expected column '{cfg.res_col}' not found in data columns: {list(data.columns)}")

        # Compute total number of unique resources (non-NaN). If zero, return zero utilization.
        if total_res != None:
            total_resources = total_res
        else:
            total_resources = data[cfg.res_col].nunique(dropna=True)
        if total_resources == 0:
            return pd.Series(dtype="float64", name="resource_utilization")

        # Filter data to relevant columns and drop rows with NaN in time or resource
        df = data[[cfg.time_col, cfg.res_col]].dropna().copy()
        df[cfg.time_col] = pd.to_datetime(df[cfg.time_col])

        # Group by time bins and count unique resources, then divide by total resources for utilization
        util = (df.set_index(cfg.time_col).groupby(pd.Grouper(freq=freq))[cfg.res_col].nunique().astype(float) / float(total_resources))

        # Regularize series
        util = regularize_series(util, freq=freq, name="resource_utilization", smoothing=smoothing)

        return util

    return load_or_compute(cache, _compute, load=load)

# ---------- Throughput Time KPI ----------
# Computation Variants for Throughput Time KPI
Throughput_Time_Variant = Literal["row", "span", "rolling"]

def build_throughput_time_series(cfg: Config, data: pd.DataFrame, load: bool = False, variant: Throughput_Time_Variant = "span", variant_param: Union[int, str] = 100, freq: str = "1D", smoothing: bool = True) -> pd.Series:
    """
    Creates a Throughput Time KPI series based on the specified variant.

    - cfg: Config object with cache settings
    - data: input DataFrame
    - load: Whether to load from cache if it exists
    - variant: method to compute throughput time ("row", "span", "rolling")
    - variant_param: parameter for the chosen variant
    - freq: frequency for regularization (only for "span" variant, ignored for others)
    - smoothing: whether to apply forward-fill smoothing
    """

    # Create cache path based on config and variant
    cache = cfg.cache_dir / f"throughput_{variant}_{str(variant_param)}_{freq.replace('/', '-')}_smooth{int(smoothing)}.pkl"

    def _compute() -> pd.Series:
        # Build first/last DataFrame. If empty, return empty series with name.
        fl = build_first_last_df(cfg, data, load=load)
        if fl.empty:
            return pd.Series(dtype="float64", name="throughput_time")
        # Make deep copy to avoid modifying cached first/last DataFrame
        fl = fl.copy()
        # Compute throughput time in seconds
        fl["throughput_time"] = (pd.to_datetime(fl["last"]) - pd.to_datetime(fl["first"])).dt.total_seconds()

        # Sort by completion time
        base = fl[["last", "throughput_time"]].sort_values(by="last").reset_index(drop=True)

        if variant == "row":
            ts = _aggregate_tt_row_number(base, group_size=int(variant_param))
        elif variant == "span":
            ts = _aggregate_tt_timespan(base, time_length=str(variant_param))
        elif variant == "rolling":
            ts = _aggregate_tt_rolling(base, window=int(variant_param))
        else:
            raise ValueError(f"Unknown throughput aggregation method: {variant}")

        # Regularize series
        ts = regularize_series(ts, freq=freq, name="throughput_time", smoothing=smoothing)

        return ts.astype(float)

    return load_or_compute(cache, _compute, load=load)

def _aggregate_tt_row_number(data: pd.DataFrame, group_size: int = 10) -> pd.Series:
    """
    Computes series according to the "row" method: aggregates by grouping cases into consecutive groups of `group_size` based on completion time, then averages throughput time within each group. The resulting series is indexed by the maximum completion time in each group.

    - data: DataFrame with columns ['last', 'throughput_time'] sorted by 'last'
    - group_size: number of cases to include in each group for averaging
    """
    # Make deep copy to avoid modifying input DataFrame
    d = data.copy()
    # Assign group numbers based on row index and group size
    d["group"] = d.index // group_size
    # Aggregate by group to get max completion time and mean throughput time
    agg = d.groupby("group").agg(last=("last", "max"), throughput_time=("throughput_time", "mean"))
    # Set index to completion time and return throughput time series
    s = agg.groupby("last")["throughput_time"].mean()
    # Ensure index is datetime and values are float
    s.index = pd.to_datetime(s.index)
    return s.astype(float)

def _aggregate_tt_timespan(data: pd.DataFrame, time_length: str = "1D") -> pd.Series:
    """
    Computes series according to the "span" method: aggregates cases based on their completion time into fixed time intervals, then averages throughput time within each interval.

    - data: DataFrame with columns ['last', 'throughput_time'] sorted by 'last'
    - time_length: length of time intervals for aggregation
    """
    # Make deep copy to avoid modifying input DataFrame
    d = data.copy()
    # Set index to completion time
    d = d.set_index("last")
    # Resample by time intervals and average throughput time within each interval
    agg = d["throughput_time"].resample(time_length).mean().ffill().to_frame("throughput_time")
    # Group intervals that have the same completion time (if any) and average throughput time, then return series
    s = agg.groupby("last")["throughput_time"].mean()
    s.index = pd.to_datetime(s.index)
    return s.astype(float)

def _aggregate_tt_rolling(data: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Computes series according to the "rolling" method: applies a rolling average with a specified window size to the throughput times, indexed by completion time.

    - data: DataFrame with columns ['last', 'throughput_time'] sorted by 'last'
    - window: size of the rolling window (number of cases to average over)
    """

    # Make deep copy to avoid modifying input DataFrame
    d = data.copy()
    # Apply rolling mean to throughput time
    d["throughput_time"] = d["throughput_time"].rolling(window=window, min_periods=1).mean()
    # Group by completion time and average throughput time for cases with the same completion time
    s = d.groupby("last")["throughput_time"].mean()
    # Ensure index is datetime and values are float
    s.index = pd.to_datetime(s.index)
    return s.astype(float)


def main(args):
    """
    Main function to execute the data processing pipeline.

    - args: command-line arguments parsed by argparse
    """
    cfg = Config(
        dataset=Path(args.d),
        cache_dir=Path(args.c) if args.c else Path("tmp/pickle"),
        utc=args.utc,
        case_col=args.c_col,
        res_col=args.r_col,
        time_col=args.t_col,
        act_col=args.a_col
    )

    df = import_data(cfg, load=False)

    if args.series == "concurrent_cases":
        variant = args.v if args.v else "sweepline"
        ts = build_concurrent_cases_series(cfg, df, load=False, variant=variant, freq=args.freq, smoothing=args.smoothing)
    elif args.series == "resource_utilization":
        ts = build_resource_utilization_series(cfg, df, load=False, freq=args.freq, smoothing=args.smoothing)
    elif args.series == "throughput_time":
        variant = args.v if args.v else "span"
        variant_param = args.vp if args.vp else ("1D" if variant == "span" else 100)
        ts = build_throughput_time_series(cfg, df, load=False, variant=variant, variant_param=variant_param, freq=args.freq, smoothing=args.smoothing)
    else:
        raise ValueError(f"Unknown series type: {args.series}")

    plot_series(ts, title=f"{args.series} KPI series", y=f"{args.series} KPI", x="Time")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Processing Pipeline for Process KPIs")
    parser.add_argument("-d", type=str, required=True, help="Path to dataset (XES or CSV)")
    parser.add_argument("-c", type=str, required=False, help="Optional caching directory")
    parser.add_argument("-utc", action="store_true", help="Store timestamps in UTC")
    parser.add_argument("-c_col", type=str, required=False, default=CASE_COL, help=f"name of the case ID column (default: {CASE_COL})")
    parser.add_argument("-r_col", type=str, required=False, default=RES_COL, help=f"name of the resource column (default: {RES_COL})")
    parser.add_argument("-t_col", type=str, required=False, default=TIME_COL, help=f"name of the time column (default: {TIME_COL})")
    parser.add_argument("-a_col", type=str, required=False, default=ACT_COL, help=f"name of the activity name column (default: {ACT_COL})")
    parser.add_argument("-series", type=str, required=True, help="Type of KPI series to compute")
    parser.add_argument("-v", type=str, required=False, help="Variant for KPI computation")
    parser.add_argument("-vp", type=str, required=False, help="Variant parameter for KPI computation")
    parser.add_argument("-freq", type=str, required=False, default="1D", help="Frequency of final KPI series (default: 1D)")
    parser.add_argument("-smoothing", action="store_true", help="Enable Smoothing of KPI series (default: True)")
    args = parser.parse_args()
    main(args=args)

    # -----------------------------
    # Example usage
    # -----------------------------

    #cfg = Config(
    #    dataset=Path("/Users/dfuhge/Downloads/Sepsis Cases - Event Log.xes"),
    #    cache_dir=Path.home() / "tmp" / "pm4py_cache",
    #    utc=True,
    #    case_col = "case:concept:name",
    #    res_col = "org:group",
    #    time_col = "time:timestamp",
    #    act_col = "concept:name"
    #)
    
    #df = import_data(cfg, load=False)

    # --- Concurrency (choose variant) ---
    #conc_sweepline = build_concurrent_cases_series(cfg, df, load=False, variant="sweepline", freq="1D")
    #conc_exact = build_concurrent_cases_series(cfg, df, load=False, variant="exact_changepoints", freq="1D")
    #conc_event = build_concurrent_cases_series(cfg, df, load=False, variant="event_sampled", freq="1D")

    # --- Resource utilization ---
    #res_util = build_resource_utilization_series(cfg, df, load=False, freq="1D", smoothing=False)

    # --- Throughput time (choose variant) ---
    #tt_rolling = build_throughput_time_series(cfg, df, load=False, variant="rolling", variant_param=100, freq="1D")
    #tt_row = build_throughput_time_series(cfg, df, load=False, variant="row", variant_param=100, freq="1D")
    #tt_span = build_throughput_time_series(cfg, df, load=False, variant="span", variant_param="1D", freq="1D")


    # Plot examples
    #plot_series(conc_sweepline, title="Concurrent Cases KPI series for Sepsis", y="Concurrent Cases KPI", x="Time")
    #plot_series(conc_exact, "Concurrent Cases KPI (exact changepoints, freq=1D)")
    #plot_series(conc_event, "Concurrent Cases KPI (event sampled, freq=1D)")

    #plot_series(res_util, "Resource Utilization KPI series for Sepsis", y="Resource Utilization KPI", x="Time")

    #plot_series(tt_rolling, "Throughput Time KPI (rolling, window_size=100, freq=1D)")
    #plot_series(tt_row, "Throughput Time KPI (row, group_size=100, freq=1D)")
    #plot_series(tt_span, "Throughput Time KPI series for Sepsis", y="Throughput Time KPI", x="Time")
