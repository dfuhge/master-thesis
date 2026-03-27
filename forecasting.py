
from __future__ import annotations

# General packages
import math
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal, Iterable, Sequence
import multiprocessing as mp
import traceback
from pathlib import Path
import os
import random

# Data processing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Models and metrics
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.base import ForecastingHorizon

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

import torch
import torch.nn as nn
from torch.nn import GRU

from darts.models import NBEATSModel
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

os.environ["TRANSFORMERS_NO_TF"] = "1"
#from chronos import Chronos2Pipeline, ChronosBoltPipeline

# Warnings
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import argparse

# Own data processing
from data_processing import (
    Config, import_data,
    build_concurrent_cases_series,
    build_resource_utilization_series,
    build_throughput_time_series,
    )

# Types
ArrayLike = Union[np.ndarray, Sequence[float]]
ForecastFn = Callable[[pd.Series, int, Optional[dict]], np.ndarray]
# Model keys
ModelKey = Literal["naive", "seasonal_naive", "ets", "sarimax", "ridge", "gru", "nbeats"]


# Default column names
DEFAULT_CASE_COL = "case:concept:name"
DEFAULT_ACT_COL = "concept:name"
DEFAULT_TIME_COL = "time:timestamp"
DEFAULT_RES_COL = "org:resource"

# ===================================
# (1) Basic utilities
# ===================================

def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """
    Sets all seeds to the parameter value for reproducibility.

    - seed: integer seed value to set for all libraries
    - deterministic_torch: if True, also set PyTorch to deterministic mode (may impact performance)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        # cuDNN / CUDA determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Force deterministic algorithms (may raise if op has no deterministic impl)
        torch.use_deterministic_algorithms(True)


# ===================================
# (2) Data preprocessing and splitting 
# ===================================

class Split3WayConfig:
    """
    Three-way split configuration for train/validation/test set.

    """
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15

    def __init__(self, train_frac, val_frac, test_frac):
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        s = self.train_frac + self.val_frac + self.test_frac
        if s != 1.0:
            raise ValueError(f"train_frac+val_frac+test_frac must sum to 1.0; got {s}")
        if min(self.train_frac, self.val_frac, self.test_frac) <= 0:
            raise ValueError("All split fractions must be positive.")

def preprocess_series(s: pd.Series) -> pd.Series:
    """
    Preprocess a time series to enforce consistency and regularity. Performs the following steps:
    1. Check if input series is a valid, non-empty pandas Series.
    2. Create deep copy to avoid modifying original data.
    3. Sorts series by index to ensure time order.
    4. Enforces numeric dtype, coercing non-numeric values to NaN.
    5. Replaces inf/-inf with NaN.
    6. Checks if result series is too short

    - s: input series
    """
    # Check if series is None or not a pd.Series
    if s is None:
        raise ValueError("Input series is None.")
    if not isinstance(s, pd.Series):
        raise TypeError(f"Expected pd.Series, got {type(s)}")
    # Deep copy to avoid modifying original
    out = s.copy()
    # Sort by index to ensure time order
    out = out.sort_index()
    # Enforce numeric dtype
    out = pd.to_numeric(out, errors="coerce")
    # Replace inf with NaN to handle any blow-ups gracefully
    out = out.replace([np.inf, -np.inf], np.nan)
    # Check if series is too short after preprocessing
    if len(out) < 20:
        raise ValueError(f"Series too short after preprocessing. length={len(out)}")

    return out

def split_train_val_test(s: pd.Series, cfg: Split3WayConfig, split_timestamp: str=None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Time-ordered train/val/test split. If split_timestamp is provided, use it to split train and test and calculate corresponding validation fraction. 
    Otherwise, split according to cfg fractions.

    - s: input series
    - cfg: Split3WayConfig with train/val/test fractions
    - split_timestamp: optional string timestamp to split train/test
    """
    # If split_timestamp is provided, use it to split train/test and calculate corresponding validation fraction.
    if not split_timestamp == None:
        # Convert split_timestamp to pandas Timestamp
        split_timestamp = pd.to_datetime(split_timestamp)
        # Split the series into train/val and test
        train_val = s[s.index <= split_timestamp]
        test = s[s.index > split_timestamp]
        # Calculate validation fraction relative to train_val
        frac_train = cfg.train_frac / (cfg.train_frac + cfg.val_frac)
        # Calculate number of training samples
        n_train = int(len(train_val) * frac_train)
        # Split train_val into train and val
        train = train_val.iloc[:n_train]
        val = train_val.iloc[n_train:]
        # Print last timestamps for sanity check
        print('Last train timestamp: ', train.index[-1])
        print('Last validation timestamp: ', val.index[-1])
        print('Last test timestamp: ', test.index[-1])
        return train, val, test

    # Calculate split indices based on cfg fractions
    n = len(s)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)

    # Ensure each split has at least 1 observation
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError("Not enough data to create non-empty test split.")

    # Perform the splits and print last timestamps for sanity check
    train = s.iloc[:n_train]
    print('Last train timestamp: ', train.index[-1])
    val = s.iloc[n_train:n_train + n_val]
    print('Last validation timestamp: ', val.index[-1])
    test = s.iloc[n_train + n_val:]
    print('Last test timestamp: ', test.index[-1])
    return train, val, test


# ===================================
# (3) Tuning and prediction utilities
# ===================================

def _build_default_param_grid(model_key: ModelKey, seed: int, season_candidates: List[int], ets_candidates: dict, ridge_alpha_candidates: List[float], gru_candidates: dict, n_beats_candidates: dict, chronos_candidates: List[str]) -> List[Optional[dict]]:
    """
    Build a parameter grid per model.

    - model_key: model to build the grid for
    - seed: random seed for any stochastic models
    - season_candidates: list of seasonal periods to try for seasonal models
    - ets_candidates: dict of candidate lists for ETS hyperparameters
    - ridge_alpha_candidates: list of alpha values to try for Ridge regression
    - gru_nbeats_candidates: dict of candidate lists for GRU/N-BEATS hyperparameters
    - chronos_candidates: list of Chronos model configs to try
    """

    # Models without hyperparameters to tune
    if model_key in ("naive", "ses", "holt"):
        return [None]
    # Seasonal naive
    if model_key == "seasonal_naive":
        return [{"season": s} for s in season_candidates]
    # Holt-Winters (ETS)
    if model_key == "ets":
        grid: List[dict] = []
        trend = ets_candidates.get("ets_trend_candidates", [None])
        seasonal = ets_candidates.get("ets_seasonal_candidates", [None])
        error = ets_candidates.get("ets_error_candidates", "add")
        damped = ets_candidates.get("ets_damped_trend_candidates", False)
        for t in trend:
            for d in damped:
                for s in seasonal:
                    for e in error:
                        for period in season_candidates:
                            grid.append({"trend": t, "seasonal": s, "error": e, "seasonal_periods": period, "damped_trend": d})
        return grid
    # SARIMAX
    if model_key == "sarimax":
        # Order candidates
        orders = [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
        # Seasonal order candidates
        seasonal_base = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
        grid: List[dict] = []
        for order in orders:
            for (P, D, Q) in seasonal_base:
                for s in season_candidates:
                    grid.append({"order": order, "seasonal_order": (P, D, Q, s)})
        return grid
    # Ridge Regression
    if model_key == "ridge":
        grid: List[dict] = []
        for lags in season_candidates:
            for alpha in ridge_alpha_candidates:
                grid.append({"lags": lags, "alpha": alpha, "add_time_features": True})
        return grid
    # GRU
    if model_key == "gru":
        grid: List[dict] = []
        batch_size_candidates = gru_candidates.get("batch_size_candidates", [128, 256])
        hidden_size_candidates = gru_candidates.get("hidden_size_candidates", [32, 64])
        num_layers_candidates = gru_candidates.get("num_layers_candidates", [1, 2])
        lr = gru_candidates.get("learning_rate_candidates", [0.001])
        for n_steps in season_candidates:
            for n_future in season_candidates:
                for hidden_size in hidden_size_candidates:
                    for num_layers in num_layers_candidates:
                        for lr_val in lr:
                            for batch_size in batch_size_candidates:
                                grid.append({
                                    "n_steps": n_steps,
                                    "n_future": n_future,
                                    "hidden_size": hidden_size,
                                    "num_layers": num_layers, 
                                    "lr": lr_val,
                                    "batch_size": batch_size,
                                    "seed": seed,
                                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                                })
        return grid
    # N-Beats  
    if model_key == "nbeats":
        grid: List[dict] = []
        n_stacks_candidates = n_beats_candidates.get("n_stacks_candidates", [2, 3])
        n_blocks_candidates = n_beats_candidates.get("n_blocks_candidates", [2, 4])
        num_layers_candidates = n_beats_candidates.get("num_layers_candidates", [2, 4])
        layer_width_candidates = n_beats_candidates.get("layer_width_candidates", [32, 64, 128, 256])
        expansion_coefficient_candidates = n_beats_candidates.get("expansion_coefficient_candidates", [16, 32])
        dropout_candidates = n_beats_candidates.get("dropout_candidates", [0.0, 0.1, 0.2])
        batch_size_candidates = n_beats_candidates.get("batch_size_candidates", [32, 64, 128, 256])
        learning_rate_candidates = n_beats_candidates.get("learning_rate_candidates", [0.001, 0.0005])
        for input_chunk_length in season_candidates:
            for n_stacks in n_stacks_candidates:
                for n_blocks in n_blocks_candidates:
                    for num_layers in num_layers_candidates:
                        for layer_width in layer_width_candidates:
                            for expansion_coefficient in expansion_coefficient_candidates:
                                for dropout in dropout_candidates:
                                    for batch_size in batch_size_candidates:
                                        for lr_val in learning_rate_candidates:
                                            grid.append({
                                                "input_chunk_length": input_chunk_length,
                                                "n_stacks": n_stacks,
                                                "n_blocks": n_blocks,
                                                "n_layers": num_layers,
                                                "layer_width": layer_width,
                                                "expansion_coefficient": expansion_coefficient,
                                                "dropout": dropout,
                                                "batch_size": batch_size,
                                                "lr": lr_val,
                                                "seed": seed
                                            })
        return grid
    # Chronos
    if model_key == "chronos":
        grid: List[dict] = []
        all_chronos_models = ["amazon/chronos-2"]
        if chronos_candidates is None:
            chronos_candidates = all_chronos_models
        for model in chronos_candidates:
            if model in all_chronos_models:
                grid.append({"chronos_config": model})
        return grid

    raise ValueError(f"No grid builder for model_key={model_key}")

def tune_on_validation(model_fn: ForecastFn, metric: Callable[[np.ndarray, np.ndarray], float], train: pd.Series, val: pd.Series, param_grid: List[Optional[dict]]) -> Tuple[Optional[dict], float]:
    """
    Tune model hyperparameters on the validation set.

    - model_fn: forecasting function
    - metric: function to compute evaluation metric
    - train: training series
    - val: validation series
    - param_grid: list of parameter dicts to try
    Returns:
      - best_params: parameter dict with lowest validation metric (or None if all failed)
      - best_score: best validation metric achieved (or inf if all failed)
    """
    horizon = len(val)

    best_params: Optional[dict] = None
    best_score = float("inf")

    for i, params in enumerate(param_grid):
        try:
            yhat = safe_predict(model_fn, train, horizon, params)
            score = metric(val.to_numpy(copy=True), yhat)
        except Exception:
            score = float("inf")

        print(f"  - candidate {i+1}/{len(param_grid)} params={params}  val_MSE={score:.6f}")

        if score < best_score:
            best_score = score
            best_params = params

    return best_params, best_score

def _predict_worker(model_key: str, train: pd.Series, horizon: int, params: dict | None, seed: int, out_q):
    """
    Subprocess worker function to run a single prediction. Catches exceptions and sends back results via a queue.

    - model_key: which model to run
    - train: training series
    - horizon: forecast horizon
    - params: model-specific parameters
    - seed: random seed for reproducibility
    - out_q: multiprocessing.Queue to send back results (status, payload)
    """
    try:
        # Re-seed inside the child for reproducibility
        try:
            set_global_seed(seed, deterministic_torch=True)
        except Exception:
            pass
        # Get model function
        fn = MODEL_FUNCTIONS[model_key]
        # Apply model
        yhat = fn(train, horizon, params)
        # Reshape and validate output
        yhat = np.asarray(yhat, dtype=float).reshape(-1)
        if yhat.shape[0] != horizon or not np.all(np.isfinite(yhat)):
            raise ValueError("Non-finite predictions or wrong horizon length")
        # Send back result
        out_q.put(("ok", yhat))
    except Exception as e:
        out_q.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))

def safe_predict_subprocess(model_key: str, train: pd.Series, horizon: int, params: dict | None, *, seed: int) -> np.ndarray:
    """
    Runs prediction in a subprocess to safely handle timeouts and memory issues. Returns the prediction or raises an error if something goes wrong.

    - model_key: which model to run
    - train: training series
    - horizon: forecast horizon
    - params: model-specific parameters
    - seed: random seed for reproducibility
    """
    timeout_s = 900  # Timeout value in seconds
    # Use "spawn" context for better isolation
    ctx = mp.get_context("spawn")
    # Create a Queue for communication
    q = ctx.Queue()
    # Start subprocess with worker function
    p = ctx.Process(target=_predict_worker, args=(model_key, train, horizon, params, seed, q))
    p.start()
    p.join(timeout_s)
    # Check if process is still alive (timeout) and terminate if so
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"{model_key} timed out after {timeout_s}s")

    # If killed (OOM etc.), exitcode is non-zero and queue may be empty
    if q.empty():
        raise RuntimeError(f"{model_key} subprocess died (exitcode={p.exitcode}). Likely OOM/kill.")
    # Get result from queue
    status, payload = q.get()
    if status != "ok":
        raise RuntimeError(f"{model_key} failed in subprocess:\n{payload}")
    return payload

def safe_predict(model_fn: ForecastFn, train: pd.Series, horizon: int, params: Optional[dict]) -> np.ndarray:
    """
    Calls model and enforces consistent output format and checks for common issues.

    - model_fn: forecasting function to call
    - train: training series
    - horizon: forecast horizon (steps)
    - params: optional parameters to pass to model_fn
    Returns:
      - yhat: predictions as np.ndarray of shape (horizon,) with finite float values
    """
    yhat = model_fn(train, horizon, params)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)

    if len(yhat) != horizon:
        raise ValueError("Wrong horizon length.")

    # discard non-finite predictions (NaN/inf)
    if not np.all(np.isfinite(yhat)):
        raise ValueError("Non-finite predictions (NaN/inf).")
    
    return yhat


# ===============================
# (4) Plotting utilities
# ===============================

def plot_forecasts(test: pd.Series, preds: Dict[str, np.ndarray], title: str) -> None:
    """
    Plot test series and forecasts

    - test: true test series
    - preds: forecasts per model
    - title: title of plot
    """
    horizon = len(test)
    x = np.arange(horizon)

    plt.figure(figsize=(10, 5))
    plt.plot(x, test.values, label="True (test)")

    for name, yhat in preds.items():
        yhat = np.asarray(yhat, dtype=float).reshape(-1)
        if len(yhat) != horizon:
            print(f"[WARN] {name}: prediction length {len(yhat)} != horizon {horizon} (skipping plot).")
            continue
        plt.plot(x, yhat, linestyle="--", label=name)

    plt.title(title)
    plt.xlabel("Forecast step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_train_test_with_forecasts(
    train: pd.Series,
    val: Optional[pd.Series],
    test: pd.Series,
    preds: Dict[str, np.ndarray],
    title: str,
    *,
    show_val: bool = True,
) -> None:
    """
    Plot train (+ optional val) + test on the original time axis,
    and overlay forecasts aligned to the test index.
    """
    plt.figure(figsize=(12, 5))

    # Plot observed history
    plt.plot(train.index, train.values, label="Train (true)")
    if val is not None and show_val and len(val) > 0:
        plt.plot(val.index, val.values, label="Val (true)")
    plt.plot(test.index, test.values, label="Test (true)")

    # Overlay forecasts on the test time index
    horizon = len(test)
    for name, yhat in preds.items():
        yhat = np.asarray(yhat, dtype=float).reshape(-1)
        if len(yhat) != horizon:
            print(f"[WARN] {name}: prediction length {len(yhat)} != horizon {horizon} (skipping).")
            continue
        pred_series = pd.Series(yhat, index=test.index)
        plt.plot(pred_series.index, pred_series.values, linestyle="--", label=f"{name} (forecast)")

    # Visual split markers
    plt.axvline(train.index[-1], linestyle=":", linewidth=1)
    if val is not None and show_val and len(val) > 0:
        plt.axvline(val.index[-1], linestyle=":", linewidth=1)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Concurrent Cases")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===============================
# (5) Models and helper functions
# ===============================

def model_naive(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    """
    Persistence baseline: forecast equals last observed value.

    - train: training series
    - horizon: number of steps to forecast
    - params: not used for this model (ignored if provided)
    """
    last = float(train.iloc[-1])
    return np.full(horizon, last, dtype=float)

def model_seasonal_naive(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    """
    Seasonal naive baseline: repeats values from the previous season.

    - train: training series
    - horizon: number of steps to forecast
    - params:
      - season: int number of steps in a season (e.g., 3600 for 1 hour at 1-second sampling)
    """
    if params is None or "season" not in params:
        raise ValueError("Seasonal naive requires params={'season': int}")
    s = int(params["season"])
    if s <= 0:
        raise ValueError("season must be positive")

    if len(train) < s:
        return model_naive(train, horizon)

    last_season = train.iloc[-s:].to_numpy(dtype=float)
    return np.resize(last_season, horizon).astype(float)

def model_ets(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    """
    Exponential Smoothing (ETS) model with optional error/trend/seasonal components and damping.

    - train: training series
    - horizon: number of steps to forecast
    - params:
      - error: "add" or "mul" (default "add")
      - trend: "add", "mul", or None (default None)
      - seasonal: "add", "mul", or None (default None)
      - seasonal_periods: int number of steps in a season (default 24)
      - damped_trend: bool whether to use damped trend (default False)
    """
    p = params or {}

    error = p.get("error", "add")
    trend = p.get("trend", None)
    seasonal = p.get("seasonal", None)
    seasonal_periods = int(p.get("seasonal_periods", 24))
    damped_trend = bool(p.get("damped_trend", False))

    ts = train.copy()

    scaler = StandardScaler()

    ts = pd.Series(
        scaler.fit_transform(ts.to_frame()).ravel(),
        index=ts.index,
        name=ts.name,
    )

    shift = 0.0
    # Shift for multiplicative components
    if trend == 'mul' or seasonal == 'mul' or error == 'mul':
        mn = float(ts.min())
        if mn <= 0:
            shift = abs(mn) + 1.0
            ts = ts + shift
    # Create model on train series
    try:
        model = ETSModel(ts,
                    error=error,
                    trend=trend,
                    damped_trend=damped_trend,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods)
        # Fit model with iterations
        res = model.fit(maxiter=1000)
    
        fc = res.forecast(horizon) 
    except Exception as error:
        traceback.print_exc()

    # shift back if we shifted for positivity
    if shift != 0.0:
        fc = fc - shift
    
    # scale back

    fc = pd.Series(
        scaler.inverse_transform(fc.to_frame()).ravel(),
        index=fc.index,
        name=train.name,
    )

    return fc





def model_sarimax(train: pd.Series, horizon: int, params: Dict[str, Any]) -> np.ndarray:
    """
    SARIMAX model for time series forecasting.

    - train: training series
    - horizon: number of steps to forecast
    - params:
      - order: tuple (p, d, q) ARIMA order (default (1, 0, 0))
      - seasonal_order: tuple (P, D, Q, s) seasonal order (default (0, 0, 0, 0))
      - trend: "n", "c", "t", or "ct" (default "c" for constant term)
      - enforce_stationarity: bool (default True)
      - enforce_invertibility: bool (default True)
      - simple_differencing: bool (default False)
      - measurement_error: bool (default False)
      - concentrate_scale: bool (default False)
      - fit_method: str optimization method for fitting (default "lbfgs")
      - maxiter: int max iterations for fitting (default 200)
      - cov_type: str covariance type for parameter estimates (default "opg")
      - optim_score: str or None optimization score for fitting (default None for exact)
      - optim_hessian: str or None optimization Hessian for fitting (default None for exact)
      - silence_startup_warnings: bool whether to silence common startup warnings (default True)
      - fail_on_convergence_warning: bool whether to raise an error if a convergence warning occurs (default False)
    """
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(train, pd.Series):
        raise TypeError("train must be a pandas Series")
    if len(train) < 3:
        raise ValueError("train is too short to fit SARIMAX robustly")

    p = params or {}
    order = p.get("order", (1, 0, 0))
    seasonal_order = p.get("seasonal_order", (0, 0, 0, 0))

    # Model options
    trend = p.get("trend", "c")  # allow intercept by default for best accuracy
    enforce_stationarity = bool(p.get("enforce_stationarity", True))
    enforce_invertibility = bool(p.get("enforce_invertibility", True))
    simple_differencing = bool(p.get("simple_differencing", False))
    measurement_error = bool(p.get("measurement_error", False))
    concentrate_scale = bool(p.get("concentrate_scale", False))  # keep full likelihood by default

    # Fit options (favor reliability over speed)
    fit_method = p.get("fit_method", "lbfgs")
    maxiter = int(p.get("maxiter", 200))
    cov_type = p.get("cov_type", "opg")
    optim_score = p.get("optim_score", None)      # exact score
    optim_hessian = p.get("optim_hessian", None)  # exact Hessian

    silence_startup_warnings = bool(p.get("silence_startup_warnings", True))
    fail_on_convergence_warning = bool(p.get("fail_on_convergence_warning", False))

    model = None
    res = None
    try:
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            simple_differencing=simple_differencing,
            measurement_error=measurement_error,
            concentrate_scale=concentrate_scale,
        )

        with warnings.catch_warnings():
            if fail_on_convergence_warning:
                warnings.filterwarnings("error", category=ConvergenceWarning)

            if silence_startup_warnings:
                warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters*")
                warnings.filterwarnings("ignore", message="Non-stationary starting seasonal autoregressive*")
                warnings.filterwarnings("ignore", message="Non-invertible starting seasonal moving average*")
                warnings.filterwarnings("ignore", message="Non-invertible starting moving average*")

            res = model.fit(
                disp=False,
                method=fit_method,
                maxiter=maxiter,
                optim_score=optim_score,
                optim_hessian=optim_hessian,
                cov_type=cov_type,
            )

        fc = res.get_forecast(steps=horizon)
        

        # For "best possible" point forecasts, use the mean
        yhat = np.asarray(fc.predicted_mean, dtype=float).reshape(-1)
        if yhat.shape != (horizon,):
            print(f"Warning: SARIMAX forecast shape mismatch: expected {(horizon,)}, got {yhat.shape}. Returning NaNs.")
            raise RuntimeError(f"Forecast shape mismatch: expected {(horizon,)}, got {yhat.shape}")
        if not np.all(np.isfinite(yhat)):
            print("Warning: SARIMAX forecast contains non-finite values (inf/nan). Returning NaNs.")
            raise FloatingPointError("Forecast contains non-finite values (inf/nan).")
        return yhat

    finally:
        # No error catching; just cleanup
        try:
            if res is not None:
                res.remove_data()
        except Exception:
            pass
        try:
            del res
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        import gc
        gc.collect()

def _make_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Cyclical time features from a DatetimeIndex.
    
    - index: Time index on which to build the features
    """
    hour = index.hour # list of the hours of the index
    dow = index.dayofweek # list of the days in the week (as int)

    # Map hour and day of the week on circle to get seasonal features
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        },
        index=index,
    )

def model_ridge_lags_sktime(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    """
    Ridge regression on lag features with optional time features, implemented using sktime's reduction forecaster.

    - train: training series
    - horizon: number of steps to forecast
    - params:
        - lags: number of lag features to use (default 60)
        - alpha: regularization strength for Ridge regression (default 1.0)
        - add_time_features: whether to add cyclical time features (hour of day, day of week) if datetime index is available (default True)
    """
    # Read hyperparameters
    p = params or {}
    lags = int(p.get("lags", 60))
    alpha = float(p.get("alpha", 1.0))
    add_time_features = bool(p.get("add_time_features", True))

    # Deep copy and enforce float type
    ts = train.copy()
    y = ts.astype(float)

    # Build exogenous time features
    X_train = None
    X_pred = None

    if add_time_features and isinstance(y.index, pd.DatetimeIndex):
        # training time features always possible
        X_train = _make_time_features(y.index)

        # on given frequency, build future time features
        if y.index.freq is not None:
            future_index = pd.date_range(
                start=y.index[-1] + y.index.freq,
                periods=horizon,
                freq=y.index.freq,
            )
            X_pred = _make_time_features(future_index)
        else:
            # if frequency unknown, drop
            X_train = None
            X_pred = None

    # Create Ridge Regression pipeline with scaler
    regressor = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=0)),
        ]
    )
    # Build forecaster on regression pipeline
    forecaster = make_reduction(
        estimator=regressor,
        window_length=lags,
        strategy="recursive", 
    )

    # Forecast horizon: 1..horizon steps ahead (relative)
    fh = ForecastingHorizon(np.arange(1, horizon + 1), is_relative=True)

    # Fit & predict
    forecaster.fit(y=y, X=X_train)
    y_pred = forecaster.predict(fh=fh, X=X_pred)

    # Return as numpy array like your current function
    return np.asarray(y_pred, dtype=float)

class GRUForecaster(nn.Module):
    """
    GRU mapping past window -> future vector.

    Input:  (batch, seq_len, 1)
    Output: (batch, n_future)
    """
    def __init__(self, hidden_size: int, n_future: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_future)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last)

def _create_window_dataset(series: np.ndarray, n_steps: int, n_future: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create windowed supervised dataset for multi-step training.

    X[i] = series[i : i+n_steps]
    y[i] = series[i+n_steps : i+n_steps+n_future]
    """
    X, y = [], []
    n = len(series)
    end = n - n_steps - n_future + 1
    for i in range(end):
        X.append(series[i:i + n_steps])
        y.append(series[i + n_steps:i + n_steps + n_future])
    return np.array(X, dtype=float), np.array(y, dtype=float)

def model_gru(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    """
    GRU forecaster trained on the provided train series only.

    - train: training series
    - horizon: number of steps to forecast
    - params:
        - n_steps: number of past steps to use as input (default 600)
        - n_future: number of future steps to predict at once during training (default 600)
        - hidden_size: GRU hidden size (default 64)
        - num_layers: number of GRU layers (default 1)
        - epochs: number of training epochs (default 50)
        - lr: learning rate (default 1e-3)
        - batch_size: training batch size (default 256)
        - seed: random seed for reproducibility (default 0)
        - device: "cuda" or "cpu" for training (default "cuda" if available else "cpu") 
    """
    p = params or {}
    n_steps = int(p.get("n_steps", 600))
    n_future = int(p.get("n_future", 600))
    hidden_size = int(p.get("hidden_size", 64))
    num_layers = int(p.get("num_layers", 1))
    epochs = int(p.get("epochs", 50))
    lr = float(p.get("lr", 1e-3))
    batch_size = int(p.get("batch_size", 256))
    seed = int(p.get("seed", 0))
    device = p.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    g = torch.Generator()
    g.manual_seed(seed)
    np.random.seed(seed)

    values = train.astype(float).to_numpy().reshape(-1, 1)
    scaler = StandardScaler()
    values_s = scaler.fit_transform(values).reshape(-1)

    X, y = _create_window_dataset(values_s, n_steps=n_steps, n_future=n_future)
    if len(X) < 50:
        return model_naive(train, horizon)

    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (samples, n_steps, 1)
    y_t = torch.tensor(y, dtype=torch.float32)                # (samples, n_future)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=g,
        num_workers=0,  # keep 0 for easiest determinism
    )

    model = GRUForecaster(hidden_size=hidden_size, n_future=n_future, num_layers=num_layers).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * len(xb)

        print(f"[GRU] epoch {epoch+1}/{epochs}  loss={total_loss/len(dataset):.6f}", flush=True)

    # Recursive prediction in chunks of n_future
    model.eval()
    hist = values_s.tolist()
    inp = torch.tensor(hist[-n_steps:], dtype=torch.float32).view(1, n_steps, 1).to(device)

    preds_s: List[float] = []
    chunks = math.ceil(horizon / n_future)

    with torch.no_grad():
        for _ in range(chunks):
            out = model(inp).cpu().numpy().reshape(-1)  # n_future
            preds_s.extend(out.tolist())
            hist.extend(out.tolist())
            inp = torch.tensor(hist[-n_steps:], dtype=torch.float32).view(1, n_steps, 1).to(device)

    preds_s = preds_s[:horizon]
    preds = scaler.inverse_transform(np.array(preds_s, dtype=float).reshape(-1, 1)).reshape(-1)
    return preds.astype(float)

def model_nbeats(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    """
    N-BEATS model for time series forecasting using the darts library.

    - train: training series
    - horizon: number of steps to forecast
    - params:
        - input_chunk_length: number of past steps to use as input (default 168, i.e., 1 week for hourly data)
        - n_stacks: number of stacks in the N-BEATS architecture (default 2)
        - n_blocks: number of blocks per stack (default 2)
        - n_layers: number of fully connected layers per block (default 2)
        - layer_width: width of fully connected layers (default 64)
        - expansion_coefficient: dimension of expansion coefficients in N-BEATS (default 32)
        - dropout: dropout rate (default 0.1)
        - lr: learning rate (default 1e-3)
        - batch_size: training batch size (default 64)
        - epochs: number of training epochs (default 50)
        - seed: random seed for reproducibility (default 0)
    """
    epochs = params.get('epochs', 50)
    try:
        model = NBEATSModel(
            input_chunk_length=params['input_chunk_length'], 
            output_chunk_length=horizon, 
            num_stacks=params['n_stacks'],
            num_blocks=params['n_blocks'],
            num_layers=params['n_layers'],
            layer_widths=params['layer_width'],
            expansion_coefficient_dim=params['expansion_coefficient'],
            dropout=params['dropout'],
            optimizer_kwargs={"lr": params['lr']},
            batch_size=params['batch_size'],
            n_epochs=epochs, 
            random_state=params["seed"])
        # Convert to darts timeseries
        ts_train = TimeSeries.from_series(train.astype("float32"))
        # Scale
        scaler = Scaler()
        ts_train = scaler.fit_transform(ts_train)
        # Fit the model
        model.fit(ts_train)
        # Predict and inverse scale
        pred = scaler.inverse_transform(model.predict(horizon))
        return pred.values().squeeze(-1)  # (horizon,)
    except Exception as error:
        print("An exception occurred:", error) 

def model_chronos(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    """
    Placeholder for a potential future Chronos implementation.
    """
    try:
        config = params.get("chronos_config", "amazon/chronos-small")
                            
        return None
    except Exception as error:
        print(error)

# Model functions
MODEL_FUNCTIONS = {"naive": model_naive, "seasonal_naive": model_seasonal_naive, "ets": model_ets, "sarimax": model_sarimax, "ridge": model_ridge_lags_sktime, "gru": model_gru, "nbeats": model_nbeats}



# ===============================
# (6) Main execution pipeline
# ===============================

def _execute(general_params: dict, model_params: dict) -> Dict[str, object]:
    """
    Main execution function for forecasting pipeline.

    general_params: general setting params in dict
    model_params: model-specific params in dict
    """
    

    # Set seed according to param set
    seed = general_params.get("seed", 42)
    set_global_seed(seed=seed, deterministic_torch=True)

    # Read model params
    season_candidates = model_params.get("season_candidates", [24, 168, 336])  # daily, weekly, biweekly for hourly data
    ridge_alpha_candidates = model_params.get("ridge_alpha_candidates", [0.1, 1.0, 10.0])
    
    # Dataset and Cache settings
    cache_dir = general_params.get("cache_dir", None)
    cache_dir_path = Path(cache_dir) if cache_dir is not None else (Path.home() / "tmp" / "pm4py_cache")

    cfg = Config(dataset=Path(general_params.get("dataset_path", "path/to/dataset.xes")), cache_dir=cache_dir_path, utc=True,
                 case_col=general_params['case_col'], act_col=general_params['act_col'], res_col=general_params['res_col'], time_col=general_params['time_col'])

    load = not general_params.get("recompute", True)

    # Import dataset
    df = import_data(cfg, load=load)
    
    # Set series parameters
    target_series = general_params.get("target_series", "concurrent_cases")
    target_series_variant = general_params.get("target_series_variant", "sweepline")
    freq = general_params.get("freq", "1D")
    # Generate series based on target_series and variant
    if target_series == "concurrent_cases":
        # Generate Concurrent Cases KPI series
        series = build_concurrent_cases_series(cfg, df,load=load, variant=target_series_variant, freq=freq, smoothing=True)
    elif target_series == "resource_utilization":
        # Generate Resource Utilization KPI series
        series = build_resource_utilization_series(cfg, df, load=load, freq=freq, smoothing=False)
    elif target_series == "throughput_time":
        # Generate Throughput Time KPI series
        series = build_throughput_time_series(cfg, df, load=load, variant=target_series_variant, variant_param=general_params.get("target_variant_param", None), freq=freq, smoothing=True)
    else:
        raise ValueError(f"Unknown target_series: {target_series}")

    # Preprocess values
    series = preprocess_series(series)

    # Truncate series 
    truncate = general_params.get("truncate", None)
    if truncate != None:
        new_length = int(len(series) * truncate)
        series = series[:new_length]

    # Split into train/val/test
    train, val, test = split_train_val_test(series, general_params.get("split_cfg", Split3WayConfig(train_frac=0.7, val_frac=0.1, test_frac=0.2)), general_params.get("split_timestamp", None))

    # Model selection
    # If None, run all models
    models_to_run = model_params.get("models_to_run", None)
    if models_to_run is None:
        models_to_run = list(MODEL_FUNCTIONS.keys())
    else:
        # Validate selection
        unknown = [m for m in models_to_run if m not in MODEL_FUNCTIONS.keys()]
        if unknown:
            raise ValueError(f"Unknown model keys: {unknown}. Allowed: {list(MODEL_FUNCTIONS.keys())}")
        models_to_run = [m for m in models_to_run if m in MODEL_FUNCTIONS]

    # Hyperparameter tuning
    metric = general_params.get("metric", mean_squared_error)
    best_params: Dict[str, Optional[dict]] = {}
    val_metrics: Dict[str, float] = {}
    test_metrics: Dict[str, float] = {}
    test_preds_for_plot: Dict[str, np.ndarray] = {}

    # Concatenate train+val for final refit
    train_plus_val = pd.concat([train, val])

    # Read Hyperparameter tuning settings
    season_candidates = model_params.get("season_candidates", None)
    ets_candidates = {
        "ets_trend_candidates": model_params.get("ets_trend_candidates", None),
        "ets_seasonal_candidates": model_params.get("ets_seasonal_candidates", None),
        "ets_error_candidates": model_params.get("ets_error_candidates", None),
        "ets_damped_trend_candidates": model_params.get("ets_damped_trend_candidates", None)
    }
    ridge_alpha_candidates = model_params.get("ridge_alpha_candidates", None)
    gru_candidates = {
        "hidden_size_candidates": model_params.get("gru_hidden_size_candidates", None),
        "num_layers_candidates": model_params.get("num_layers_candidates", None),
        "learning_rate_candidates": model_params.get("learning_rate_candidates", None),
        "batch_size_candidates": model_params.get("batch_size_candidates", None),
        "dropout_candidates": model_params.get("dropout_candidates", None),
    }
    n_beats_candidates = {
        "input_chunk_length_candidates": model_params.get("n_steps_candidates", None),
        "n_stacks_candidates": model_params.get("n_stacks_candidates", None),
        "n_blocks_candidates": model_params.get("n_blocks_candidates", None),
        "n_layers_candidates": model_params.get("num_layers_candidates", None),
        "layer_width_candidates": model_params.get("layer_width_candidates", None),
        "expansion_coefficient_candidates": model_params.get("expansion_coefficient_candidates", None),
        "dropout_candidates": model_params.get("dropout_candidates", None),
        "batch_size_candidates": model_params.get("batch_size_candidates", None),
        "learning_rate_candidates": model_params.get("learning_rate_candidates", None),
    }
    chronos_candidates = model_params.get("chronos_candidates", None)
    # Iterate over selected models and tune each on validation set, then evaluate on test set
    for key in models_to_run:
        # Get the model function
        model_function = MODEL_FUNCTIONS[key]

        # Build a parameter grid for this model
        grid = _build_default_param_grid(key, seed=seed, season_candidates=season_candidates, ets_candidates=ets_candidates, ridge_alpha_candidates=ridge_alpha_candidates, gru_candidates=gru_candidates, n_beats_candidates=n_beats_candidates, chronos_candidates=chronos_candidates,)

        print(f"\n[TUNE] Model: {model_function.__name__} ({key})  grid_size={len(grid)}")

        # --- subprocess-backed model function (closure used only in parent) ---
        def _subproc_model_fn(train_series: pd.Series, h: int, params: Optional[dict]) -> np.ndarray:
            return safe_predict_subprocess(model_key=key, train=train_series, horizon=h, params=params, seed=seed
            )

        # Tune on validation via subprocess
        bp, bscore = tune_on_validation(_subproc_model_fn, metric, train, val, grid)
        best_params[key] = bp
        val_metrics[key] = bscore

        # Refit on train+val and evaluate on test (also in subprocess)
        try:
            yhat_test = safe_predict(_subproc_model_fn, train_plus_val, len(test), bp)
            test_metrics[key] = metric(test, yhat_test)
        except Exception:
            traceback.print_exc()
            yhat_test = np.full(len(test), np.nan, dtype=float)
            test_metrics[key] = float("nan")


        # Build a readable label
        label = model_function.__name__
        test_preds_for_plot[label] = yhat_test

        print(f"[RESULT] {model_function.__name__} best_val_metric={val_metrics[key]:.6f}  test_metric={test_metrics[key]:.6f}  best_params={bp}")

    # Store results to csv
    results_dir = Path(general_params.get("results_path", "./results"))
    results_dir = results_dir / (target_series + ("_truncated" if truncate else ""))
    results_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(results_dir / "train.csv", index=True, header=True)
    val.to_csv(results_dir / "val.csv", index=True, header=True)
    test.to_csv(results_dir / "test.csv", index=True, header=True)
    i = 0
    for label, pred in test_preds_for_plot.items():
        np.save(results_dir / (str(i) + label.replace("{", "").replace("}", "").replace("=", "") + ".npy"), pred)
        i += 1

    #plot_forecasts(test=test, preds=test_preds_for_plot, title=f"KPI: {target_series} - test+predictions")
    #plot_train_test_with_forecasts(train=train, val=val, test=test, preds=test_preds_for_plot, title=f"KPI: {target_series} - train/val/test+predictions", show_val=True)

    return {"series": series, "train": train, "val": val, "test": test, "best_params": best_params, "val_metrics": val_metrics, "test_metrics": test_metrics, "test_preds": test_preds_for_plot}


def main(args):
    """
    Main function to run the forecasting pipeline.
    """
    target = args.series
    trun = args.truncate
    #targets = ["concurrent_cases", "resource_utilization", "throughput_time"]
    #truncate_values = [None, 0.75]
    print("TARGET: " + target + " - truncation: " + str(trun))
    # General parameters including dataset settings, target series configuration, and split config:
    general_params = {
        # Path to dataset
        "dataset_path": args.dataset,
        # Path to results
        "results_path": args.results,
        # Data columns
        "case_col": args.case_col,
        "act_col": args.activity_col,
        "time_col": args.time_col,
        "res_col": args.resource_col,
        # Target series
        "target_series": target, #"throughput_time",
        # Target series variant
        "target_series_variant": args.variant if (args.variant != "" and args.variant != None) else ("sweepline" if target == "concurrent_cases" else "span"),#"sweepline" if target == "concurrent_cases" else "span",#"span",
        "target_variant_param": args.variant_param,#"1D",
        # Target frequency of series
        "freq": args.frequency,#"1D",
        # Recompute results or use cached results
        "recompute": args.recompute,#True,
        # Split timestamp for hard split train/test
        "split_timestamp": None,
        # Split config if no split timestamp is set (fraction-based splits)
        "split_config": Split3WayConfig(train_frac=args.train, val_frac=args.val, test_frac=args.test),
        # Metric to evaluate
        "metric": mean_squared_error,
        # Seed for reproducibility
        "seed": 42,
        # Cache directory
        "cache_dir": args.cache_dir,
        # Optional truncation for concurrent cases and throughput time
        "truncate": trun
    }

    # Model and hyperparameter tuning parameters:
    model_params = {
        # Models to run. If empty, runs all available models. Models must be in available.keys().
        "models_to_run": args.models,#None,
        # Hyperparameter tuning candidates
        "ets_trend_candidates": ["add", "mul", None],
        "ets_seasonal_candidates": ["add", "mul", None],
        "ets_error_candidates": ["add", "mul"],
        "ets_damped_trend_candidates": [True, False],
        "season_candidates": [7, 30],      # seasonal candidates, used for seasonal naive, ETS, SARIMAX
        "ridge_alpha_candidates": [0.1, 1.0, 10.0], # alpha candidates for Ridge
        "n_steps_candidates": [7, 30],       # input window candidates
        "gru_n_future_steps_candidates": [7, 30],  # training future steps candidates 
        "batch_size_candidates": [64],     # batch size candidates
        "gru_hidden_size_candidates": [128], # hidden size candidates 
        "num_layers_candidates": [2, 4],              # number of layers candidates
        "dropout_candidates": [0.0, 0.2],            # dropout candidates 
        "learning_rate_candidates": [0.001, 0.01],   # learning rate candidates 
        "n_stacks_candidates": [2, 3],           
        "n_blocks_candidates": [2, 3],           
        "layer_width_candidates": [128],            
        "expansion_coefficient_candidates": [32],
        "chronos_candidates": None,  # if None, will use all available Chronos pretrained models
    }
    # Execute forecasting pipeline
    out = _execute(general_params, model_params)

    # Print sorted validation results (in MSE; other metrics could be added to the output dict and sorted/printed similarly)
    print("\n=== FINAL VALIDATION RESULTS (sorted) ===")
    items = sorted(out["val_metrics"].items(), key=lambda kv: (np.isnan(kv[1]), kv[1]))
    for key, score in items:
        print(f"{key:15s}  val_metric={score:.6f}  best_params={out['best_params'][key]}")

    # Print sorted test results (in MSE; other metrics could be added to the output dict and sorted/printed similarly)
    print("\n=== FINAL TEST RESULTS (sorted) ===")
    items = sorted(out["test_metrics"].items(), key=lambda kv: (np.isnan(kv[1]), kv[1]))
    for key, score in items:
        print(f"{key:15s}  test_metric={score:.6f}  best_params={out['best_params'][key]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create forecasts for process KPIs using various models and hyperparameter tuning.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to dataset (XES or CSV)")
    parser.add_argument("-r", "--results", type=str, required=True, help="Results directory")
    parser.add_argument("-c_col", "--case_col", type=str, required=False, default=DEFAULT_CASE_COL, help=f"name of the case ID column (default: {DEFAULT_CASE_COL})")
    parser.add_argument("-r_col", "--resource_col", type=str, required=False, default=DEFAULT_RES_COL, help=f"name of the resource column (default: {DEFAULT_RES_COL})")
    parser.add_argument("-t_col", "--time_col", type=str, required=False, default=DEFAULT_TIME_COL, help=f"name of the time column (default: {DEFAULT_TIME_COL})")
    parser.add_argument("-a_col", "--activity_col", type=str, required=False, default=DEFAULT_ACT_COL, help=f"name of the activity name column (default: {DEFAULT_ACT_COL})")
    parser.add_argument("-v", "--variant", type=str, required=False, help="Variant for KPI computation")
    parser.add_argument("-vp", "--variant_param", type=str, required=False, help="Variant parameter for KPI computation")
    parser.add_argument("-freq", "--frequency", type=str, required=False, default="1D", help="Frequency of final KPI series (default: 1D)")
    parser.add_argument("-recompute", "--recompute", action="store_true", help="Recompute results")
    parser.add_argument("-tr", "--truncate", type=float, required=True, help="Level of truncation")
    parser.add_argument("-train", "--train", type=float, required=False, default=0.7, help="Train fraction")
    parser.add_argument("-val", "--val", type=float, required=False, default=0.1, help="Validation fraction")
    parser.add_argument("-test", "--test", type=float, required=False, default=0.2, help="Test fraction")
    parser.add_argument("-c", "--cache_dir", type=str, required=False, help="Optional caching directory")
    parser.add_argument('-m', '--models', nargs='+', default=[])
    parser.add_argument("-s", "--series", type=str, required=True, help="KPI Series type")
    
    args = parser.parse_args()
    main(args=args)
