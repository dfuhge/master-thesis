"""
Evaluation Script for Process KPI Forecasting
"""

from __future__ import annotations

from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import pm4py
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import data_processing as data
import argparse


def _load_pred_npy(path: Path) -> np.ndarray:
    """
    Load prediction from .npy file and reshape to 1D array.

    - path: Path to .npy file containing predictions.
    """
    arr = np.load(path, allow_pickle=False)
    return np.asarray(arr).reshape(-1)

def _load_baseline_generative_lstm(path: str, test_path: str):
    """
    Load baseline predictions from GenerativeLSTM model.

    - path: Path to .csv file containing GenerativeLSTM predictions.
    - test_path: Path to test CSV file.
    """
    # Read in test csv
    test = pd.read_csv(test_path)
    min_timestamp = test.iloc[:, 0].min()
    max_timestamp = test.iloc[:, 0].max()
    df = pd.read_csv(path, delimiter=",", index_col=0, low_memory=False)
    # Filter only artificial event resources
    df = df[df["user"].str.contains("Role", na=False)]
    df["end_timestamp"] = pd.to_datetime(df["end_timestamp"], format="mixed", utc=True)
    #print(df['end_timestamp'])
    cfg = data.Config(dataset="", time_col="end_timestamp", case_col="caseid", res_col="user")
    ts_cc = data.build_concurrent_cases_series(cfg, df, freq="1D")
    ts_ru = data.build_resource_utilization_series(cfg, df, freq="1D")
    ts_tt = data.build_throughput_time_series(cfg, df, variant="span", variant_param="1D")

    # Truncate Series to timestamp
    full_index = pd.date_range(start=min_timestamp, end=max_timestamp, freq="1D")
    ts_cc_truncated = ts_cc.reindex(full_index, method="ffill")
    ts_ru_truncated = ts_ru.reindex(full_index, method="ffill")
    ts_tt_truncated = ts_tt.reindex(full_index, method="ffill")

    return (ts_cc_truncated, ts_ru_truncated, ts_tt_truncated)

def _load_baseline_processtransformer(path: str, test_path: str):
    """
    Load baseline predictions from ProcessTransformer model.

    - path: Path to .csv file containing ProcessTransformer predictions.
    - test_path: Path to test CSV file.
    """
    df = pd.read_csv(path, delimiter=",", index_col=0, low_memory=False)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], format="mixed")
    cfg = data.Config("")
    ts_cc = data.build_concurrent_cases_series(cfg, df)
    ts_ru = None
    ts_tt = data.build_throughput_time_series(cfg, df, variant="span", variant_param="1D")
    # Read in test csv
    test = pd.read_csv(test_path)
    min_timestamp = test.iloc[:, 0].min()
    max_timestamp = test.iloc[:, 0].max()
    # Truncate Series to timestamp
    full_index = pd.date_range(start=min_timestamp, end=max_timestamp, freq="1D")
    ts_cc_truncated = ts_cc.reindex(full_index, method="ffill")
    ts_tt_truncated = ts_tt.reindex(full_index, method="ffill")

    return (ts_cc_truncated, ts_ru, ts_tt_truncated)

def _evaluate_one(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate one prediction series against the ground truth and return a dictionary of metrics.

    - y_test: Ground truth values as a 1D numpy array.
    - y_pred: Predicted values as a 1D numpy array.
    """

    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "n_eval": len(y_test),
    }

def _build_prediction_dataframe_all(df_test: pd.DataFrame, preds: list[dict[str, np.ndarray]]) -> pd.DataFrame:
    """
    Combine test series and all prediction files into one DataFrame.

    - df_test: DataFrame containing the test series (ground truth) with timestamps as index and the first column as values.
    - preds: List of dictionaries, where each dictionary has a model name as key and the corresponding prediction series as a 1D numpy array as value.
    """

    y_true = pd.to_numeric(df_test.iloc[:, 0], errors="coerce").to_numpy()

    df = pd.DataFrame(index=df_test.index)
    df["y_true"] = y_true
    for i, p in preds.items():
        df[i] = p

    return df

def _plot_all_predictions(df: pd.DataFrame, title: str, series_type: str, path: str) -> None:
    """
    Plot ground truth and all prediction series with a clean time axis.

    - df: DataFrame containing the test series and all prediction series, with timestamps as index and each series as a column.
    - title: Title for the plot.
    - series_type: Type of KPI series (e.g., "cc", "ru", "tt") for labeling the y-axis.
    - path: Path to save the plot image.
    """

    # Ensure datetime index
    df.index = pd.to_datetime(df.index)

    fig, ax = plt.subplots(figsize=(14,6))

    # True series as scatter
    # thin line
    ax.plot(df.index, df["y_true"], linewidth=0.7, alpha=0.7, color="black")
    ax.scatter(df.index, df["y_true"], s=8, label="True", color="black")

    labels = {
        "naive": "Naive",
        "seasonal_naive": "S. Naive",
        "ets": "ETS",
        "sarimax": "SARIMA",
        "ridge_lags_sktime": "Ridge",
        "gru": "GRU",
        "nbeats": "N-BEATS",
        "GenerativeLSTMall": "G.LSTM all",
        "GenerativeLSTMtr": "G.LSTM tr",
        "ProcessTransformerall": "ProcessT. all",
        "ProcessTransformertr": "ProcessT. tr"

    }

    # Predictions as scatter
    for col in df.columns:
        if col == "y_true":
            continue
        # thin line
        ax.plot(
            df.index,
            df[col],
            linewidth=0.7,
            alpha=0.7
        )
        ax.scatter(
            df.index,
            df[col],
            s=8,
            alpha=0.6,
            label=labels[col]
        )

    # ---- Clean time axis ----
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    fig.autofmt_xdate()

    ax.set_xlabel("Time")
    if series_type == "cc":
        ax.set_ylabel("Concurrent Cases KPI")
    elif series_type == "ru":
        ax.set_ylabel("Resource Utilization KPI")
    elif series_type == "tt":
        ax.set_ylabel("Throughput Time KPI")
    ax.set_title(title)

    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()





def main(args) -> None:
    """
    Main function to evaluate predictions and plot results.
    """
    pred_dir = Path(args.predictions)
    test_dir = Path(args.test)
    generative_lstm_dir = Path(args.glstm)
    pt_dir = Path(args.pt)
    file_name = args.file
    save = Path(args.save)
    s_types = {
        "throughput_time": "tt",
        "resource_utilization": "ru",
        "concurrent_cases": "cc"
    }
    series_type = s_types[str(pred_dir).split("/")[-1].split("_truncated")[0]]
    y_test = pd.read_csv(test_dir, delimiter=",", index_col=0)

    pred_files = sorted(pred_dir.glob("*.npy"))
    if not pred_files:
        raise FileNotFoundError(f"No .npy files found in directory: {pred_dir}")

    preds = {}
    for pred_file in pred_files:
        try:
            y_pred = _load_pred_npy(pred_file).astype(float)
            name = str(pred_file).split("/")[-1][7:-4]
            preds[name]= y_pred
            result = _evaluate_one(y_test, y_pred)

            print(f"\n=== {pred_file} ===")
            print(f"MSE   : {result['mse']:.6f}")
            print(f"MAE   : {result['mae']:.6f}")
            print(f"RMSE  : {result['rmse']:.6f}")
            print(f"n_eval: {result['n_eval']}")
        except Exception as e:
            print(f"\n=== {pred_file.name} ===")
            print(f"ERROR : {e}")

    # Load Baselines
    # GenerativeLSTM - all data
    p = generative_lstm_dir / "all_data" / file_name
    cc, ru, tt = _load_baseline_generative_lstm(p, test_dir)
    if series_type == "cc":
        preds["GenerativeLSTMall"] = cc
    elif series_type == "ru":
        preds["GenerativeLSTMall"] = ru
    elif series_type == "tt":
        preds["GenerativeLSTMall"] = tt
    try:
        result = _evaluate_one(y_test, preds["GenerativeLSTMall"])

        print(f"\n=== GenerativeLSTM all data ===")
        print(f"MSE   : {result['mse']:.6f}")
        print(f"MAE   : {result['mae']:.6f}")
        print(f"RMSE  : {result['rmse']:.6f}")
        print(f"n_eval: {result['n_eval']}")
    except Exception as e:
        print(f"\n=== GenerativeLSTM all data ===")
        print(f"ERROR : {e}")
        print(traceback.print_exc())

    # GenerativeLSTM - full traces
    p = generative_lstm_dir / "full_traces" / file_name
    cc, ru, tt = _load_baseline_generative_lstm(p, test_dir)
    if series_type == "cc":
        preds["GenerativeLSTMtr"] = cc
    elif series_type == "ru":
        preds["GenerativeLSTMtr"] = ru
    elif series_type == "tt":
        preds["GenerativeLSTMtr"] = tt
    try:
        result = _evaluate_one(y_test, preds["GenerativeLSTMtr"])

        print(f"\n=== GenerativeLSTM full traces ===")
        print(f"MSE   : {result['mse']:.6f}")
        print(f"MAE   : {result['mae']:.6f}")
        print(f"RMSE  : {result['rmse']:.6f}")
        print(f"n_eval: {result['n_eval']}")
    except Exception as e:
        print(f"\n=== GenerativeLSTM full traces ===")
        print(f"ERROR : {e}")
        print(traceback.print_exc())

    # ProcessTransformer only for CC, TT
    if series_type != "ru":
        # Load ProcessTransformer all data
        p = pt_dir / "all_data" / file_name
        print(p)
        cc, __, tt = _load_baseline_processtransformer(p, test_dir)

        if series_type == "cc":
            preds["ProcessTransformerall"] = cc
        elif series_type == "tt":
            preds["ProcessTransformerall"] = tt
        try:
            result = _evaluate_one(y_test, preds["ProcessTransformerall"])

            print(f"\n=== ProcessTransformer all data ===")
            print(f"MSE   : {result['mse']:.6f}")
            print(f"MAE   : {result['mae']:.6f}")
            print(f"RMSE  : {result['rmse']:.6f}")
            print(f"n_eval: {result['n_eval']}")
        except Exception as e:
            print(f"\n=== ProcessTransformer all data ===")
            print(f"ERROR : {e}")
            print(traceback.print_exc())

        # Load ProcessTransformer full traces
        p = pt_dir / "full_traces" / file_name
        print(p)
        cc, __, tt = _load_baseline_processtransformer(p, test_dir)

        if series_type == "cc":
            preds["ProcessTransformertr"] = cc
        elif series_type == "tt":
            preds["ProcessTransformertr"] = tt
        try:
            result = _evaluate_one(y_test, preds["ProcessTransformertr"])

            print(f"\n=== ProcessTransformer full traces ===")
            print(f"MSE   : {result['mse']:.6f}")
            print(f"MAE   : {result['mae']:.6f}")
            print(f"RMSE  : {result['rmse']:.6f}")
            print(f"n_eval: {result['n_eval']}")
        except Exception as e:
            print(f"\n=== ProcessTransformer full traces ===")
            print(f"ERROR : {e}")
            print(traceback.print_exc())
    # Construct Title
    s_labels = {
        "tt": "Throughput Time",
        "ru": "Resource Utilization",
        "cc": "Concurrent Cases"
    }
    dataset = file_name.split("_split_")[0]
    d_labels = {
        "bpic2012": "BPIC 2012",
        "bpic2020_domestic_declarations": "BPIC 2020/Domestic Declarations",
        "bpic2020_international_declarations": "BPIC 2020/International Declarations",
        "bpic2020_prepaid_travel": "BPIC 2020/Prepaid Travel Cost",
        "bpic2020_request_payment": "BPIC 2020/Request for Payment",
        "bpic2020_travel_permit": "BPIC 2020/Travel Permit",
        "helpdesk": "Helpdesk",
        "sepsis": "Sepsis"
    }
    truncate = "(truncated to 75%)" if file_name.split("_last_")[1] != "None.csv" else ""
    title = s_labels[series_type] + " KPI – " + d_labels[dataset] + " " + truncate + " – Predictions and True Values"
    trun = " Truncated" if truncate != "" else ""
    trun_label = "_truncated" if truncate != "" else ""
    save_path = save / (series_type.upper() + trun) / (series_type + trun_label + "_" + dataset)
    print(save_path)
    # Plot
    df_plot = _build_prediction_dataframe_all(y_test, preds)
    _plot_all_predictions(df_plot, title=title, series_type=series_type, path=save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create forecasts for process KPIs using various models and hyperparameter tuning.")
    parser.add_argument("-pred", "--predictions", type=str, required=True, help="Path to predictions folder")
    parser.add_argument("-test", "--test", type=str, required=True, help="Path to test data")
    parser.add_argument("-glstm", "--glstm", type=str, required=True, help="Path to GenerativeLSTM repository")
    parser.add_argument("-pt", "--pt", type=str, required=True, help="Path to ProcessTransformer repository")
    parser.add_argument("-file", "--file", type=str, required=True, help="Filename for baselines")
    parser.add_argument("-save", "--save", type=str, required=True, help="Save path for plots<")
    
    args = parser.parse_args()
    main(args=args)