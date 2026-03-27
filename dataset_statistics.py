"""
This module provides functions to compute comprehensive statistics and exploratory data analysis (EDA) on process event logs represented as pandas DataFrames. 
It includes checks for data quality issues such as missing values, duplicates, temporal ordering violations, and attribute stability. 
The main function `compute_stats` generates a detailed dictionary of statistics that can be printed in a human-readable format using the `print_stats` helper function. 
The module is designed to be flexible with respect to column names and can handle both XES and CSV input formats.

"""


from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import argparse

import pm4py
from pm4py.objects.log.util import dataframe_utils

# Default column names
DEFAULT_CASE_COL = "case:concept:name"
DEFAULT_ACT_COL = "concept:name"
DEFAULT_TIME_COL = "time:timestamp"
DEFAULT_RES_COL = "org:resource"


def _safe_dt(x: pd.Series) -> pd.Series:
    """
    Ensure datetime dtype of the series.
    
    - x: Series to analyze

    Returns:
    - converted series with datetime values
    """
    if not pd.api.types.is_datetime64_any_dtype(x):
        return pd.to_datetime(x, errors="coerce")
    return x


def _fmt_ts(ts: Optional[pd.Timestamp]) -> str:
    """
    Ensures ISO time format.

    - ts: timestamp to check

    Returns:
    - iso-formatted timestamp
    """
    if ts is None or pd.isna(ts):
        return "N/A"
    return ts.isoformat()


def _float_or_nan(x) -> float:
    """
    Tries to convert value to float, else returns NaN

    - x: value to convert

    Returns:
    - converted value, or NaN if error
    """
    try:
        return float(x)
    except Exception:
        return float("nan")


def compute_stats(df: pd.DataFrame, case_col: str = DEFAULT_CASE_COL, act_col: str = DEFAULT_ACT_COL, time_col: str = DEFAULT_TIME_COL, res_col: str = DEFAULT_RES_COL, topk: int = 15, key_attrs: Optional[list[str]] = None) -> dict:
    """
    Computes core statistics and EDA measurements for event log

    - df: event log DataFrame
    - case_col: name of case ID column
    - act_col: name of activity name column
    - time_col: name of timestamp column
    - res_col: name of resource column
    - topk: number of top items to include in frequency lists
    - key_attrs: list of key attributes to check for stability (default: case, activity
    """
    stats: dict = {}

    # Get key attributes for later statistics (they are especially checked for stability
    if key_attrs is None:
        key_attrs = [case_col, act_col, time_col, res_col]

    # Check if all necessary columns are present
    has_act = act_col in df.columns
    has_res = res_col in df.columns
    has_time = time_col in df.columns
    has_case = case_col in df.columns
    if not has_case:
        raise ValueError(f"Missing required case column '{case_col}'")
    if not has_time:
        raise ValueError(f"Missing required timestamp column '{time_col}'")
    if not has_res:
        raise ValueError(f"Missing required resource column '{res_col}'")
    if not has_act:
        raise ValueError(f"Missing required activity column '{act_col}'")

    # Basic counts
    stats["Number of events"] = int(len(df))
    stats["Number of traces (cases)"] = int(df[case_col].nunique(dropna=True))

    # Try to parse timestamps, count failures
    t = _safe_dt(df[time_col])
    stats["Timestamp parse failures"] = int(t.isna().sum())
    stats["Timestamp parse failure rate"] = float(t.isna().mean()) if len(df) else 0.0
    stats["First event date"] = _fmt_ts(t.min())
    stats["Last event date"] = _fmt_ts(t.max())

    # Get distinct activities and resources
    stats["# distinct activities"] = int(df[act_col].nunique(dropna=True)) if has_act else 0
    stats["# distinct resources"] = int(df[res_col].nunique(dropna=True)) if has_res else 0

    # Check columns for missing values
    completeness = []
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_rate = float(df[col].isna().mean()) if len(df) else 0.0
        completeness.append(
            {
                "column": str(col),
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "n_unique": int(df[col].nunique(dropna=True)),
            }
        )
    completeness = sorted(completeness, key=lambda x: (-x["missing_rate"], x["column"]))
    stats["Attribute completeness"] = completeness
    stats["Total missing value rate (top 10 columns)"] = completeness[:10]

    # Check for duplicates
    dup_rows = int(df.duplicated().sum())
    stats["Duplicate rows (exact duplicates)"] = dup_rows
    stats["Duplicate row rate"] = float(dup_rows / len(df)) if len(df) else 0.0

    dup_subset = [c for c in [case_col, act_col, time_col, res_col] if c in df.columns]
    dup_sig = int(df.duplicated(subset=dup_subset).sum()) if dup_subset else 0
    stats["Duplicate event signatures"] = dup_sig
    stats["Duplicate event signature rate"] = float(dup_sig / len(df)) if len(df) else 0.0

    # Check case length distribution
    case_lengths = df.groupby(case_col, dropna=False).size().astype(int)
    stats["Case length (events per trace) - mean"] = float(case_lengths.mean()) if len(case_lengths) else float("nan")
    stats["Case length - median"] = float(case_lengths.median()) if len(case_lengths) else float("nan")
    stats["Case length - min"] = int(case_lengths.min()) if len(case_lengths) else 0
    stats["Case length - max"] = int(case_lengths.max()) if len(case_lengths) else 0
    # Check different quantiles
    for q in (0.25, 0.75, 0.90, 0.95, 0.99):
        stats[f"Case length - p{int(q * 100)}"] = float(case_lengths.quantile(q)) if len(case_lengths) else float("nan")
    case_length_freq = case_lengths.value_counts().sort_index()
    stats["Case length frequency (top 20 lengths)"] = [
        {"case_length": int(length), "count": int(count)}
        for length, count in case_length_freq.head(20).items()
    ]
    stats["Top longest cases"] = [
        {"case_id": str(case_id), "case_length": int(length)}
        for case_id, length in case_lengths.sort_values(ascending=False).head(topk).items()
    ]

    # Check intra-trace temporal ordering
    df_sorted = df.copy()
    df_sorted["_t"] = t
    sort_cols = [case_col, "_t"]
    if has_act:
        sort_cols.append(act_col)
    df_sorted = df_sorted.sort_values(sort_cols, na_position="last")

    # Temporal ordering violations in original order
    df_orig = df.copy()
    df_orig["_t"] = t
    prev_orig = df_orig.groupby(case_col)["_t"].shift(1)
    violation_mask = df_orig["_t"] < prev_orig
    n_violations = int(violation_mask.sum(skipna=True))
    cases_with_violations = int(df_orig.loc[violation_mask, case_col].nunique()) if n_violations > 0 else 0

    stats["Intra-trace timestamp order violations (#events)"] = n_violations
    stats["Intra-trace timestamp order violation rate"] = float(n_violations / len(df_orig)) if len(df_orig) else 0.0
    stats["Cases with temporal-order violations"] = cases_with_violations
    stats["Cases with temporal-order violation rate"] = (
        float(cases_with_violations / df[case_col].nunique(dropna=True))
        if df[case_col].nunique(dropna=True) > 0 else 0.0
    )

    # Equal timestamps within same case
    same_case_ts = (
        df_sorted.groupby([case_col, "_t"], dropna=False).size().reset_index(name="count")
    )
    same_case_ts = same_case_ts[same_case_ts["count"] > 1]
    stats["Case-timestamp combinations with multiple events"] = int(len(same_case_ts))

    # Inter-event gaps
    df_sorted["_prev_t"] = df_sorted.groupby(case_col)["_t"].shift(1)
    delta = df_sorted["_t"] - df_sorted["_prev_t"]
    stats["Zero inter-event gaps"] = int((delta == pd.Timedelta(0)).sum(skipna=True))
    stats["Negative inter-event gaps after sorting"] = int((delta < pd.Timedelta(0)).sum(skipna=True))

    # Trace durations with quantiles
    start = df_sorted.groupby(case_col)["_t"].min()
    end = df_sorted.groupby(case_col)["_t"].max()
    durations = (end - start).dt.total_seconds()

    stats["Trace duration (seconds) - mean"] = float(durations.mean()) if len(durations) else float("nan")
    stats["Trace duration - median"] = float(durations.median()) if len(durations) else float("nan")
    stats["Trace duration - min"] = float(durations.min()) if len(durations) else float("nan")
    stats["Trace duration - max"] = float(durations.max()) if len(durations) else float("nan")
    for q in (0.25, 0.75, 0.90, 0.95, 0.99):
        stats[f"Trace duration - p{int(q * 100)}"] = float(durations.quantile(q)) if len(durations) else float("nan")

    # Distinct Trace variants
    if has_act:
        seq = df_sorted.groupby(case_col)[act_col].apply(tuple)
        stats["# trace variants (unique activity sequences)"] = int(seq.nunique(dropna=True))
        top_variants = seq.value_counts().head(topk)
        stats["Top variants (count)"] = [
            {"variant": " -> ".join(map(str, v)), "count": int(c)}
            for v, c in top_variants.items()
        ]
    else:
        stats["# trace variants (unique activity sequences)"] = 0
        stats["Top variants (count)"] = []

    # Activity frequency
    if has_act:
        top_acts = df[act_col].value_counts(dropna=True).head(topk)
        stats["Top activities (count)"] = [
            {"activity": str(a), "count": int(c)}
            for a, c in top_acts.items()
        ]
    else:
        stats["Top activities (count)"] = []

    # Resource coverage and values
    if has_res:
        missing_resource = int(df[res_col].isna().sum())
        stats["Missing resource values"] = missing_resource
        stats["Missing resource rate"] = float(missing_resource / len(df)) if len(df) else 0.0

        top_res = df[res_col].value_counts(dropna=True).head(topk)
        stats["Top resources (count)"] = [
            {"resource": str(r), "count": int(c)}
            for r, c in top_res.items()
        ]

        res_case_cov = (
            df.dropna(subset=[res_col])
            .groupby(res_col)[case_col]
            .nunique()
            .sort_values(ascending=False)
            .head(topk)
        )
        stats["Top resources by covered cases"] = [
            {"resource": str(r), "covered_cases": int(c)}
            for r, c in res_case_cov.items()
        ]
    else:
        stats["Missing resource values"] = 0
        stats["Missing resource rate"] = 0.0
        stats["Top resources (count)"] = []
        stats["Top resources by covered cases"] = []

    # Temporal pattern diagnostics
    valid_t = df_sorted["_t"].dropna()
    if len(valid_t):
        weekday_counts = valid_t.dt.day_name().value_counts()
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        stats["Events by weekday"] = [
            {"weekday": day, "count": int(weekday_counts.get(day, 0))}
            for day in weekday_order
        ]

        hour_counts = valid_t.dt.hour.value_counts().sort_index()
        stats["Events by hour"] = [
            {"hour": int(h), "count": int(c)}
            for h, c in hour_counts.items()
        ]

        daily_counts = valid_t.dt.floor("D").value_counts().sort_index()
        stats["Daily event count summary"] = {
            "mean": _float_or_nan(daily_counts.mean()),
            "median": _float_or_nan(daily_counts.median()),
            "min": _float_or_nan(daily_counts.min()),
            "max": _float_or_nan(daily_counts.max()),
            "std": _float_or_nan(daily_counts.std()),
        }

        if len(daily_counts) > 1:
            q1 = daily_counts.quantile(0.25)
            q3 = daily_counts.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_days = daily_counts[(daily_counts < lower) | (daily_counts > upper)]
            stats["Potential anomalous days (IQR rule)"] = [
                {"date": str(idx.date()), "count": int(val)}
                for idx, val in outlier_days.head(topk).items()
            ]
        else:
            stats["Potential anomalous days (IQR rule)"] = []
    else:
        stats["Events by weekday"] = []
        stats["Events by hour"] = []
        stats["Daily event count summary"] = {}
        stats["Potential anomalous days (IQR rule)"] = []

    # Attribute stability for key attributes
    attr_stability = []
    for attr in key_attrs:
        if attr not in df.columns:
            continue

        missing = int(df[attr].isna().sum())
        n_unique = int(df[attr].nunique(dropna=True))
        case_nunique = df.groupby(case_col)[attr].nunique(dropna=True)

        attr_stability.append(
            {
                "attribute": str(attr),
                "missing_count": missing,
                "missing_rate": float(missing / len(df)) if len(df) else 0.0,
                "n_unique": n_unique,
                "case_level_distinct_mean": _float_or_nan(case_nunique.mean()),
                "case_level_distinct_median": _float_or_nan(case_nunique.median()),
                "case_level_distinct_p95": _float_or_nan(case_nunique.quantile(0.95)),
            }
        )

    stats["KPI-relevant attribute stability"] = attr_stability

    # Summary
    stats["Data quality summary"] = {
        "n_events": int(len(df)),
        "n_cases": int(df[case_col].nunique(dropna=True)),
        "timestamp_parse_failures": int(t.isna().sum()),
        "duplicate_rows": dup_rows,
        "duplicate_event_signatures": dup_sig,
        "cases_with_temporal_order_violations": cases_with_violations,
        "missing_resource_values": int(df[res_col].isna().sum()) if has_res else 0,
        "distinct_resources": int(df[res_col].nunique(dropna=True)) if has_res else 0,
        "median_case_length": float(case_lengths.median()) if len(case_lengths) else float("nan"),
        "mean_case_length": float(case_lengths.mean()) if len(case_lengths) else float("nan"),
    }

    return stats


def print_stats(stats: dict) -> None:
    
    """
    Helper function to print statistics and EDA

    - stats: statistics to print
    """
    print("\n=== Dataset Statistics + EDA ===\n")

    # Scalars / dicts first
    for k, v in stats.items():
        if isinstance(v, list):
            continue
        if isinstance(v, dict):
            print(f"{k}:")
            if len(v) == 0:
                print("  (none)")
            else:
                for kk, vv in v.items():
                    print(f"  {kk}: {vv}")
            print()
            continue
        print(f"{k}: {v}")

    # Lists afterwards
    for k, v in stats.items():
        if not isinstance(v, list):
            continue
        print(f"\n{k}:")
        if len(v) == 0:
            print("  (none)")
            continue
        for row in v:
            if isinstance(row, dict):
                items = ", ".join(f"{kk}={row[kk]}" for kk in row.keys())
                print(f"  - {items}")
            else:
                print(f"  - {row}")
    print()


def main(args) -> None:
    """
    Main function to compute dataset statistics and EDA.
    """
    path = Path(args.dataset)
    if not path.exists():
        raise FileNotFoundError(f"XES file not found: {path}")

    # Read XES into pm4py log, convert to DataFrame
    suffix = path.suffix.lower()

    if suffix == ".xes":
        log = pm4py.read_xes(str(path))
        df = pm4py.convert_to_dataframe(log)
    elif suffix == ".csv":
        df = pd.read_csv(path)

    # Standardize dataframe types
    df = dataframe_utils.convert_timestamp_columns_in_df(df)

    stats = compute_stats(df=df, topk=15, case_col=args.case_col, act_col=args.activity_col, time_col=args.time_col, res_col=args.resource_col)
    print_stats(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute dataset statistics and EDA for process event logs")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to dataset (XES or CSV)")
    parser.add_argument("-c_col", "--case_col", type=str, required=False, default=DEFAULT_CASE_COL, help=f"name of the case ID column (default: {DEFAULT_CASE_COL})")
    parser.add_argument("-r_col", "--resource_col", type=str, required=False, default=DEFAULT_RES_COL, help=f"name of the resource column (default: {DEFAULT_RES_COL})")
    parser.add_argument("-t_col", "--time_col", type=str, required=False, default=DEFAULT_TIME_COL, help=f"name of the time column (default: {DEFAULT_TIME_COL})")
    parser.add_argument("-a_col", "--activity_col", type=str, required=False, default=DEFAULT_ACT_COL, help=f"name of the activity name column (default: {DEFAULT_ACT_COL})")
    args = parser.parse_args()
    main(args=args)