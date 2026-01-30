import pandas as pd

from typing import Optional
import numpy as np
import pandas as pd
import traceback
from sklearn.metrics import mean_squared_error
from pathlib import Path
import os

import data_processing_improved as data
import forecasting_improved as forecast
from forecasting_improved import Split3WayConfig

def convert_baseline_to_time_series(path: str):
    df = pd.read_csv(path)
    ts_concurrency = data.build_concurrency_data(df)
    ts_ru = data.build_resource_utilization_data(df)
    ts_tt = data.build_throughput_time_data(df)

    return (ts_concurrency, ts_ru, ts_tt)

def _fingerprint(s: pd.Series, name: str):
    import pandas as pd, numpy as np, hashlib
    arr = np.asarray(s.values, dtype=float)
    idx = s.index.astype("int64") if isinstance(s.index, pd.DatetimeIndex) else np.arange(len(s))
    h = hashlib.md5(np.concatenate([idx.astype("int64"), np.nan_to_num(arr, nan=1e308).view("int64")]).tobytes()).hexdigest()
    print(f"{name}: len={len(s)} start={s.index[0]} end={s.index[-1]} md5={h}")


def _execute_compare_one_model(cfg_dataset: data.Config, cfg_baseline: data.Config, kpi: str = "concurrency",split_cfg: Split3WayConfig = Split3WayConfig(0.7, 0.1, 0.2), freq: str = "1s", model_key: str = "ridge", model_params: Optional[dict] = None, metric_fn = None, plot: bool = True):
    """
    Run ONE model with provided hyperparameters and compare vs baseline on the test split.

    Returns dict with:
      - train, val, test
      - yhat_model_test (np.ndarray)
      - yhat_base_test (pd.Series)
      - metric_model, metric_baseline
      - plot_payload (dict label->preds) if plot=True
    """

    # ----------------------------
    # Defaults
    # ----------------------------
    if metric_fn is None:
        # Default metric: MSE
        metric_fn = forecast.mse

    # Available Models
    models = {
            "naive": ("Naive", forecast.model_naive),
            "seasonal_naive": ("SeasonalNaive", forecast.model_seasonal_naive),
            "ses": ("SES", forecast.model_ses),
            "holt": ("Holt", forecast.model_holt),
            "ets": ("ETS", forecast.model_ets),
            "sarimax": ("SARIMAX", forecast.model_sarimax),
            "ridge": ("Ridge", forecast.model_ridge_lags_sktime),
            "gru": ("GRU", forecast.model_gru),
            "nbeats": ("N-Beats", forecast.model_nbeats),
        }
    if model_key not in models:
        raise ValueError(f"Unknown model_key={model_key}. Allowed: {list(models.keys())}")

    # Import data
    df_org = data.import_xes(cfg_dataset, from_file=False)
    df_baseline = pd.read_csv(os.path.join(cfg_baseline.root, cfg_baseline.dataset))
    # Build series
    series_org = None
    series_baseline = None

    if kpi == "concurrency":
        series_org = data.build_concurrency_series(
            cfg_dataset, 
            df_org,
            freq=freq,
            variant="hourly_sweepline",     # or whatever you used
            keep_datetime_index=True,
            from_file=False,                # if you used caching
        )
        series_baseline = data.build_concurrency_series(
            cfg_baseline,
            df_baseline,
            freq=freq,
            variant="hourly_sweepline",
            time_col="tm_real", 
            case_col="caseid",
            from_file=False)
    elif kpi == "resource_utilization":
        series_org = data.build_resource_utilization_series(
            cfg_dataset, df_org,
            from_file=False,
            variant="observed_resources_per_bin",
            freq=freq,
            smoothing=True,
        )
        series_baseline = data.build_resource_utilization_series(cfg_baseline, df_baseline, res_col="resource", time_col="tm_real")
    elif kpi == "throughput_time":
        series_org = data.build_throughput_time_series(cfg_dataset, df_org,
            from_file=False,
            method="rolling",
            method_param=100,
            freq=freq,
            smoothing=True,)
        series_baseline = data.build_throughput_time_series(cfg_baseline, df_baseline, time_col="tm_real", case_col="caseid")
    else:
        raise ValueError(f"Unknown kpi={kpi}.")
    

    # ----------------------------
    # Prepare series
    # ----------------------------

    # Set frequency
    if freq is not None:
        try:
            series_org = series_org.asfreq(freq, method="ffill")
        except Exception:
            pass
    
    preprocessed_series = forecast.preprocess_series(series_org)

    train, val, test = forecast.split_train_val_test(preprocessed_series, split_cfg,)
    train_plus_val = pd.concat([train, val])

    _fingerprint(preprocessed_series, "SERIES")
    #_fingerprint(train, "TRAIN")
    #_fingerprint(val, "VALIDATION")
    #_fingerprint(test, "TEST")

    # Align baseline to test index (critical!)
    if freq is not None:
        try:
            series_baseline = series_baseline.asfreq(freq, method="ffill")
        except Exception as e:
            print(e)
            pass

    series_baseline = series_baseline.reindex(test.index)
    series_baseline = series_baseline.ffill()

    # ----------------------------
    # Run model (no tuning)
    # ----------------------------
    name, fn = models[model_key]

    try:
        yhat_model_test = forecast.safe_predict(fn, train_plus_val, len(test), model_params)
    except Exception:
        traceback.print_exc()
        yhat_model_test = np.full(len(test), np.nan, dtype=float)

    # Truncate both series
    test = test[:-300]
    yhat_model_test = yhat_model_test[:-300]
    series_baseline = series_baseline[:-300]

    # ----------------------------
    # Compute metrics (baseline + model)
    # ----------------------------
    metric_model = metric_fn(test, yhat_model_test)
    metric_baseline = metric_fn(test, series_baseline.values)

    # ----------------------------
    # Plot
    # ----------------------------
    plot_payload = None
    if plot:
        # Build a readable label for the model
        label = name
        if model_params:
            if model_key == "seasonal_naive":
                label = f"SeasonalNaive(season={model_params.get('season')})"
            elif model_key == "ets":
                label = f"ETS(sp={model_params.get('seasonal_periods')}, trend={model_params.get('trend')}, seas={model_params.get('seasonal')})"
            elif model_key == "sarimax":
                label = f"SARIMAX(order={model_params.get('order')}, seas={model_params.get('seasonal_order')})"
            elif model_key == "ridge":
                label = f"Ridge(lags={model_params.get('lags')}, alpha={model_params.get('alpha')})"
            elif model_key == "gru":
                label = f"GRU(n_steps={model_params.get('n_steps')}, hidden={model_params.get('hidden_size')})"

        plot_payload = {
            "Baseline": np.asarray(series_baseline.values, dtype=float),
            label: np.asarray(yhat_model_test, dtype=float),
        }

        print("Metric: " + metric_fn.__name__)
        print("Model: " + model_key)
        print("MSE Model: " + str(metric_model))
        print("MSE Baseline: " + str(metric_baseline))

        forecast.plot_forecasts(
            test,
            plot_payload,
            title=f"Compare on test (target={getattr(series_org, 'name', kpi)})",
        )

    return {
        "series": series_org,
        "train": train,
        "val": val,
        "test": test,
        "baseline_test": series_baseline,              # pd.Series aligned to test index
        "model_test": yhat_model_test,          # np.ndarray length len(test)
        "metric_baseline": metric_baseline,
        "metric_model": metric_model,
        "plot_preds": plot_payload,
        "model_key": model_key,
        "model_params": model_params,
    }

if __name__ == "__main__":
    cfg_dataset = data.Config(
        root=Path("/Volumes/Daniel/Thesis/resources"),
        dataset="BPI_2012/BPI_Challenge_2012.xes",
        cache_dir=Path.home() / "tmp" / "pm4py_cache",
        utc=True,
    )
    cfg_baseline = data.Config(
        root=Path("/Users/dfuhge/Documents/Studium/Uni Mannheim/Master/Masterarbeit/Masterarbeit Wifo/Code/master-thesis/master-thesis/src/compare/GenerativeLSTM-master/output_files/20260128_F6024322_477F_4F21_9065_EA718099E2DC/results"),
        dataset="gen_training_1_absolute_time.csv",
        cache_dir=Path.home() / "tmp" / "pm4py_cache",
        utc=True,
    )


    _execute_compare_one_model(cfg_dataset, cfg_baseline, kpi="concurrency", freq="1H", model_key="ridge", model_params={'lags': 168, 'alpha': 10.0, 'add_time_features': True})