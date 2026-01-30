"""
Time-series forecasting runner for process-mining features
=========================================================

What this version adds (requested):
- ✅ A proper VALIDATION SET (train / val / test split, time-ordered)
- ✅ Hyperparameter tuning "as usual":
    - For each model: try a parameter grid
    - Choose best params on validation (min MSE)
    - Refit on (train + val)
    - Evaluate once on test
- ✅ Thorough code comments throughout
- ✅ Keeps your improved data-processing pipeline integration:
    expects module `data_processing` exporting:
      Config, import_xes,
      build_concurrency_series, build_resource_utilization_series, build_throughput_time_series

Important practical note (especially with freq="1S"):
- Some models (ETS/SARIMAX) can become expensive with huge seasonal periods (e.g., 3600, 86400).
  This code includes *small, safe* default grids, and you can adjust the candidate lists.
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union, Literal, Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sktime.forecasting.compose import make_reduction
from sktime.forecasting.base import ForecastingHorizon

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

import torch
import torch.nn as nn

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

try:
    # ChronosPipeline works for "amazon/chronos-t5-*" models
    # ChronosBoltPipeline works for "amazon/chronos-bolt-*" models
    from chronos import ChronosPipeline, ChronosBoltPipeline
except ImportError as e:
    raise ImportError(
        "Please install chronos-forecasting: pip install chronos-forecasting"
    ) from e


ArrayLike = Union[np.ndarray, Sequence[float]]

import os
# =============================================================================
# 1) Basic utilities (preprocessing, splitting, metrics, plotting)
# =============================================================================

@dataclass(frozen=True)
class Split3WayConfig:
    """
    Three-way split configuration for time series.

    Example: train=0.7, val=0.15, test=0.15
    - train: used to fit model during tuning
    - val: used to select hyperparameters
    - test: only used once at the end for unbiased evaluation
    """
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15

    def __post_init__(self):
        s = self.train_frac + self.val_frac + self.test_frac
        if not np.isclose(s, 1.0):
            raise ValueError(f"train_frac+val_frac+test_frac must sum to 1.0; got {s}")
        if min(self.train_frac, self.val_frac, self.test_frac) <= 0:
            raise ValueError("All split fractions must be positive.")


def preprocess_series(
    s: pd.Series,
    *,
    enforce_float: bool = True,
    sort_index: bool = True,
    fillna_method: Literal["ffill", "zero", "drop"] = "ffill",
) -> pd.Series:
    """
    Preprocess a time series to make models more stable.

    Steps:
    1) sort index (important for temporal ordering)
    2) convert values to float
    3) replace inf with NaN
    4) fill/drop NaNs
    """
    if s is None:
        raise ValueError("Input series is None.")
    if not isinstance(s, pd.Series):
        raise TypeError(f"Expected pd.Series, got {type(s)}")

    out = s.copy()

    if sort_index:
        out = out.sort_index()

    if enforce_float:
        out = pd.to_numeric(out, errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)

    if fillna_method == "ffill":
        out = out.ffill().fillna(0.0)
    elif fillna_method == "zero":
        out = out.fillna(0.0)
    elif fillna_method == "drop":
        out = out.dropna()
    else:
        raise ValueError(f"Unknown fillna_method: {fillna_method}")

    if len(out) < 20:
        raise ValueError(f"Series too short after preprocessing. length={len(out)}")

    return out


def split_train_val_test(
    s: pd.Series,
    cfg: Split3WayConfig,
    split_timestamp: str=None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Time-ordered train/val/test split (no shuffling).

    Uses positional slicing (.iloc) to avoid ambiguity with DateTimeIndex.
    """
    if not split_timestamp == None:
        split_timestamp = pd.to_datetime(split_timestamp)
        train_val = s[s.index <= split_timestamp]
        test = s[s.index > split_timestamp]
        frac_train = cfg.train_frac / (cfg.train_frac + cfg.val_frac)
        n_train = int(len(train_val) * frac_train)
        train = train_val.iloc[:n_train]
        val = train_val.iloc[n_train:]
        print('Last train timestamp: ', train.index[-1])
        print('Last validation timestamp: ', val.index[-1])
        print('Last test timestamp: ', test.index[-1])
        return train, val, test

    n = len(s)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)

    # Ensure each split has at least 1 observation
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError("Not enough data to create non-empty test split.")

    train = s.iloc[:n_train]
    print('Last train timestamp: ', train.index[-1])
    val = s.iloc[n_train:n_train + n_val]
    print('Last validation timestamp: ', val.index[-1])
    test = s.iloc[n_train + n_val:]
    print('Last test timestamp: ', test.index[-1])
    return train, val, test


def mse(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """
    Mean Squared Error (MSE) with strict length checking.
    """
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(yt) != len(yp):
        raise ValueError(f"MSE length mismatch: len(y_true)={len(yt)}, len(y_pred)={len(yp)}")
    return float(mean_squared_error(yt, yp))


def plot_forecasts(
    test: pd.Series,
    preds: Dict[str, np.ndarray],
    title: str,
) -> None:
    """
    Plot test truth vs predicted arrays.

    We plot against step index (0..horizon-1) to keep it simple/consistent.
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
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
# 2) Model interface and hyperparameter tuning helpers
# =============================================================================

# Every model must accept (train_series, horizon:int, params:dict|None) and return a 1D np.ndarray
ForecastFn = Callable[[pd.Series, int, Optional[dict]], np.ndarray]


def safe_predict(
    model_fn: ForecastFn,
    train: pd.Series,
    horizon: int,
    params: Optional[dict],
) -> np.ndarray:
    """
    Calls the model and enforces a clean output shape.

    This prevents silent failures due to:
    - returning list instead of array
    - wrong horizon length
    """
    yhat = model_fn(train, horizon, params)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)

    if len(yhat) != horizon:
        raise ValueError("Wrong horizon length.")

    # ✅ discard non-finite predictions (NaN/inf)
    if not np.all(np.isfinite(yhat)):
        raise ValueError("Non-finite predictions (NaN/inf).")
    
    return yhat


def tune_on_validation(
    model_fn: ForecastFn,
    train: pd.Series,
    val: pd.Series,
    param_grid: List[Optional[dict]],
    *,
    verbose: bool = True,
) -> Tuple[Optional[dict], float]:
    """
    "As usual" hyperparameter selection using a validation set.

    Procedure:
    - For each candidate param dict:
        - Fit model on TRAIN
        - Predict VAL horizon (len(val))
        - Compute MSE against VAL
    - Choose parameters with minimum validation MSE
    - Return (best_params, best_val_mse)

    Notes:
    - We do NOT use test here. Test is reserved for final evaluation only.
    - For expensive models, keep grid small.
    """
    horizon = len(val)

    best_params: Optional[dict] = None
    best_score = float("inf")

    for i, params in enumerate(param_grid):
        try:
            yhat = safe_predict(model_fn, train, horizon, params)
            score = mse(val, yhat)
        except Exception:
            score = float("inf")

        if verbose:
            print(f"  - candidate {i+1}/{len(param_grid)} params={params}  val_MSE={score:.6f}")

        if score < best_score:
            best_score = score
            best_params = params

    return best_params, best_score


# =============================================================================
# 3) Baseline models
# =============================================================================

def model_naive(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    """
    Persistence baseline: forecast equals last observed value.
    """
    last = float(train.iloc[-1])
    return np.full(horizon, last, dtype=float)


def model_seasonal_naive(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    """
    Seasonal naive baseline: repeats values from the previous season.

    params:
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


# =============================================================================
# 4) Statistical models (SES, Holt, ETS, SARIMAX)
# =============================================================================

def model_ses(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    """
    Simple Exponential Smoothing (level-only).
    """
    fit = SimpleExpSmoothing(train).fit(optimized=True)
    return fit.forecast(horizon).to_numpy(dtype=float)


def model_holt(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    """
    Holt method (level + trend).
    """
    fit = Holt(train).fit(optimized=True)
    return fit.forecast(horizon).to_numpy(dtype=float)


def model_ets(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    """
    ETS (Holt-Winters) using statsmodels ExponentialSmoothing.

    IMPORTANT (statsmodels behavior):
    - `use_boxcox` is set at model initialization, NOT in `.fit()`.
      If you pass it to `.fit()` you can trigger:
      "ValueError: use_boxcox was set at model initialization and cannot be changed"

    params (optional):
      - trend: "add"|"mul"|None
      - seasonal: "add"|"mul"|None
      - seasonal_periods: int
      - use_boxcox: bool
      - damped_trend: bool
      - remove_bias: bool
    """
    p = params or {}

    trend = p.get("trend", "add")
    seasonal = p.get("seasonal", "add")
    seasonal_periods = int(p.get("seasonal_periods", 24))
    use_boxcox = bool(p.get("use_boxcox", False))
    damped_trend = bool(p.get("damped_trend", False))
    remove_bias = bool(p.get("remove_bias", False))

    ts = train.copy()

    # If Box-Cox: data must be positive -> shift if needed
    shift = 0.0
    if use_boxcox:
        mn = float(ts.min())
        if mn <= 0:
            shift = abs(mn) + 1.0
            ts = ts + shift

    # ✅ Set use_boxcox here (constructor), not in fit()
    model = ExponentialSmoothing(
        ts,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        damped_trend=damped_trend,
        use_boxcox=use_boxcox,   # <-- here
    )

    # ✅ Do NOT pass use_boxcox to fit() / Catch non-convergence
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        try:
            fit = model.fit(optimized=True, remove_bias=remove_bias)
        except ConvergenceWarning:
            return np.full(horizon, np.nan, dtype=float)

    fc = fit.forecast(horizon).to_numpy(dtype=float)

    # shift back if we shifted for positivity
    if shift != 0.0:
        fc = fc - shift

    return fc



def model_sarimax(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    """
    Robust SARIMAX:
    - Uses a bounded optimizer config to reduce blow-ups
    - Treats convergence warnings as failures (returns NaNs)
    - Avoids expensive diagnostics
    """
    p = params or {}
    order = p.get("order", (1, 1, 1))
    seasonal_order = p.get("seasonal_order", (0, 0, 0, 0))

    # Optional fit controls (tunable)
    #method = p.get("fit_method", "lbfgs")
    #maxiter = int(p.get("maxiter", 50))
    method = p.get("fit_method", "powell")
    maxiter = int(p.get("maxiter", 25))

    # These often reduce numerical headaches for real-world spiky series
    enforce_stationarity = bool(p.get("enforce_stationarity", False))
    enforce_invertibility = bool(p.get("enforce_invertibility", False))

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
    )

    # Convert convergence warnings to errors so tuning can discard them
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        try:
            res = model.fit(disp=False, method=method, maxiter=maxiter)
        except ConvergenceWarning:
            return np.full(horizon, np.nan, dtype=float)
        except Exception:
            return np.full(horizon, np.nan, dtype=float)

    fc = res.forecast(steps=horizon)
    return np.asarray(fc, dtype=float).reshape(-1)


# =============================================================================
# 5) ML baseline: Ridge regression on lags + time features (fast, strong)
# =============================================================================
def _make_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical time features from a DatetimeIndex (kept consistent with your training builder)."""
    hour = index.hour
    dow = index.dayofweek

    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        },
        index=index,
    )

def model_ridge_lags_sktime(
    train: pd.Series,
    horizon: int,
    params: Optional[dict] = None
) -> np.ndarray:
    """
    Ridge regression on lag features using sktime's reduced regression forecaster.

    params:
      - lags: int                -> window_length
      - alpha: float             -> Ridge(alpha)
      - add_time_features: bool  -> include hour/dow cyclical encodings as exogenous X (if possible)

    Notes:
      - Multi-step forecasting is done recursively (like your loop), but handled by sktime.
      - Time features require a DatetimeIndex AND a known frequency to create future timestamps.
      - If the series is too short, falls back to model_naive(train, horizon).
    """
    p = params or {}
    lags = int(p.get("lags", 60))
    alpha = float(p.get("alpha", 1.0))
    add_time_features = bool(p.get("add_time_features", True))

    y = train.astype(float)

    # Need enough history for window_length + some samples
    # (rule of thumb similar to your <50 check)
    if len(y) < max(50, lags + 5):
        return model_naive(train, horizon)

    # ----------------------------
    # Exogenous time features (X)
    # ----------------------------
    X_train = None
    X_pred = None

    if add_time_features and isinstance(y.index, pd.DatetimeIndex):
        # Fit-time features always possible
        X_train = _make_time_features(y.index)

        # For predict-time features we need a future index; easiest is if freq is known
        if y.index.freq is not None:
            future_index = pd.date_range(
                start=y.index[-1] + y.index.freq,
                periods=horizon,
                freq=y.index.freq,
            )
            X_pred = _make_time_features(future_index)
        else:
            # If we cannot construct future timestamps, we omit X at predict time.
            # (You can alternatively infer freq, but that can be risky if irregular.)
            X_train = None
            X_pred = None

    # ----------------------------
    # Ridge regressor (with scaling)
    # ----------------------------
    regressor = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=0)),
        ]
    )

    forecaster = make_reduction(
        estimator=regressor,
        window_length=lags,
        strategy="recursive",   # matches your recursive loop behavior
    )

    # Forecast horizon: 1..horizon steps ahead (relative)
    fh = ForecastingHorizon(np.arange(1, horizon + 1), is_relative=True)

    # Fit & predict
    forecaster.fit(y=y, X=X_train)
    y_pred = forecaster.predict(fh=fh, X=X_pred)

    # Return as numpy array like your current function
    return np.asarray(y_pred, dtype=float)


# =============================================================================
# 6) Neural model: GRU forecaster (with small tuning grid by default)
# =============================================================================

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

    params:
      - n_steps, n_future
      - hidden_size, num_layers
      - epochs, lr, batch_size
      - seed
      - device: "cpu"|"cuda"
    """
    p = params or {}
    n_steps = int(p.get("n_steps", 600))
    n_future = int(p.get("n_future", 30))
    hidden_size = int(p.get("hidden_size", 64))
    num_layers = int(p.get("num_layers", 1))
    epochs = int(p.get("epochs", 30))
    lr = float(p.get("lr", 1e-3))
    batch_size = int(p.get("batch_size", 256))
    seed = int(p.get("seed", 0))
    device = p.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

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

        # Light progress printing
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"[GRU] epoch {epoch+1}/{epochs}  loss={total_loss/len(dataset):.6f}")

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


# =============================================================================
# 6b) Neural model: N-BEATS forecaster
# =============================================================================

class NBeatsBlock(nn.Module):
    """
    One N-BEATS block: backcast + forecast.
    We use a generic block (no explicit trend/seasonality decomposition)
    which is usually strong and simple.
    """
    def __init__(self, input_size: int, hidden_size: int, n_layers: int, theta_size: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        self.mlp = nn.Sequential(*layers)

        # theta projects to backcast+forecast parameters
        self.theta = nn.Linear(hidden_size, theta_size)

        # final linear maps to backcast/forecast vectors
        self.backcast_lin = nn.Linear(theta_size, input_size)
        self.forecast_lin = nn.Linear(theta_size, 1)  # we will produce 1-step and iterate
        # (Iterative forecasting keeps interface consistent for any horizon.)

    def forward(self, x):
        # x: (batch, input_size)
        h = self.mlp(x)
        theta = self.theta(h)
        backcast = self.backcast_lin(theta)        # (batch, input_size)
        forecast1 = self.forecast_lin(theta)       # (batch, 1)
        return backcast, forecast1


class NBeatsModel(nn.Module):
    """
    Stacked residual blocks.
    Produces 1-step forecast; multi-step is done iteratively.
    """
    def __init__(
        self,
        input_size: int,
        n_blocks: int,
        hidden_size: int,
        n_layers: int,
        theta_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_size=input_size,
                hidden_size=hidden_size,
                n_layers=n_layers,
                theta_size=theta_size,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        # x: (batch, input_size)
        residual = x
        forecast = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)

        for blk in self.blocks:
            backcast, f1 = blk(residual)
            residual = residual - backcast
            forecast = forecast + f1

        return forecast  # (batch, 1)


def _create_window_dataset_1step(series: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-step supervised dataset:
      X[t] = series[t-n_steps : t]
      y[t] = series[t]
    """
    X, y = [], []
    for t in range(n_steps, len(series)):
        X.append(series[t - n_steps:t])
        y.append(series[t])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def model_nbeats(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    """
    N-BEATS iterative forecaster trained on the provided train series only.

    params:
      - n_steps: int          (input window length)
      - n_blocks: int
      - hidden_size: int
      - n_layers: int         (MLP depth per block)
      - theta_size: int       (internal projection size)
      - dropout: float
      - epochs: int
      - lr: float
      - batch_size: int
      - weight_decay: float
      - seed: int
      - device: "cpu"|"cuda"
      - grad_clip: float|None
      - patience: int (optional early stopping; small & simple)
    """
    p = params or {}
    n_steps = int(p.get("n_steps", 168))
    n_blocks = int(p.get("n_blocks", 3))
    hidden_size = int(p.get("hidden_size", 256))
    n_layers = int(p.get("n_layers", 2))
    theta_size = int(p.get("theta_size", 128))
    dropout = float(p.get("dropout", 0.0))
    epochs = int(p.get("epochs", 30))
    lr = float(p.get("lr", 1e-3))
    batch_size = int(p.get("batch_size", 256))
    weight_decay = float(p.get("weight_decay", 0.0))
    seed = int(p.get("seed", 0))
    grad_clip = p.get("grad_clip", None)
    patience = int(p.get("patience", 0))  # 0 disables early stopping
    device = p.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    y = train.astype(float).to_numpy().reshape(-1, 1)
    scaler = StandardScaler()
    y_s = scaler.fit_transform(y).reshape(-1)

    if len(y_s) < max(50, n_steps + 10):
        return model_naive(train, horizon)

    X, y1 = _create_window_dataset_1step(y_s, n_steps=n_steps)
    if len(X) < 50:
        return model_naive(train, horizon)

    X_t = torch.tensor(X, dtype=torch.float32)                  # (samples, n_steps)
    y_t = torch.tensor(y1, dtype=torch.float32).view(-1, 1)     # (samples, 1)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = NBeatsModel(
        input_size=n_steps,
        n_blocks=n_blocks,
        hidden_size=hidden_size,
        n_layers=n_layers,
        theta_size=theta_size,
        dropout=dropout,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # Simple early stopping on training loss (since you already use VAL outside the model)
    best_loss = float("inf")
    bad_epochs = 0

    model.train()
    for epoch in range(epochs):
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            optim.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)

        avg = total / max(1, count)

        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"[NBEATS] epoch {epoch+1}/{epochs}  loss={avg:.6f}")

        if patience > 0:
            if avg + 1e-12 < best_loss:
                best_loss = avg
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    # Iterative forecasting
    model.eval()
    hist = y_s.tolist()
    preds_s: List[float] = []

    with torch.no_grad():
        for _ in range(horizon):
            x_in = torch.tensor(hist[-n_steps:], dtype=torch.float32).view(1, n_steps).to(device)
            y_next = model(x_in).cpu().numpy().reshape(-1)[0]
            preds_s.append(float(y_next))
            hist.append(float(y_next))

    preds = scaler.inverse_transform(np.array(preds_s, dtype=float).reshape(-1, 1)).reshape(-1)
    return preds.astype(float)





@dataclass
class ChronosConfig:
    model_name: str = "amazon/chronos-bolt-base"  # or "amazon/chronos-t5-base"
    device: Optional[str] = None                  # "cuda", "mps", or "cpu"
    torch_dtype: Optional[str] = None             # "bfloat16", "float16", "float32"
    # Forecast behavior
    prediction_length: int = 24
    num_samples: int = 200
    seed: int = 42


class ChronosForecaster:
    """
    A small, practical wrapper around Chronos / Chronos-Bolt.

    - predict_samples: returns (num_samples, horizon)
    - predict_quantiles: returns dict {q: (horizon,)}
    - predict_mean: returns (horizon,)
    """

    def __init__(self, cfg: ChronosConfig):
        self.cfg = cfg
        self._pipeline = self._load_pipeline(cfg)

    @staticmethod
    def _infer_device(user_device: Optional[str]) -> str:
        if user_device:
            return user_device
        if torch.cuda.is_available():
            return "cuda"
        # macOS Apple Silicon
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _infer_dtype(device: str, user_dtype: Optional[str]) -> Optional[torch.dtype]:
        if user_dtype is None:
            # sensible defaults
            if device in ("cuda", "mps"):
                return torch.float16
            return None  # let library default (float32 on CPU)
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if user_dtype not in mapping:
            raise ValueError(f"torch_dtype must be one of {list(mapping.keys())}")
        return mapping[user_dtype]

    def _load_pipeline(self, cfg: ChronosConfig):
        device = self._infer_device(cfg.device)
        dtype = self._infer_dtype(device, cfg.torch_dtype)

        is_bolt = "chronos-bolt" in cfg.model_name.lower()
        PipelineCls = ChronosBoltPipeline if is_bolt else ChronosPipeline

        # device_map works for CUDA/CPU; for MPS it is safest to omit device_map and .to("mps") afterwards
        kwargs = {}
        if device != "mps":
            kwargs["device_map"] = device
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        pipe = PipelineCls.from_pretrained(cfg.model_name, **kwargs)

        if device == "mps":
            # move model explicitly for Apple Silicon
            pipe.model.to("mps")

        # reproducibility for sampling
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        return pipe

    @staticmethod
    def _to_1d_float_array(y: ArrayLike) -> np.ndarray:
        arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            raise ValueError("Input series must have at least 2 observations.")
        if not np.isfinite(arr).all():
            raise ValueError("Input series contains NaN/Inf. Please impute/clean first.")
        return arr

    def predict_samples(
        self,
        y_history: ArrayLike,
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
    ) -> np.ndarray:
        y = self._to_1d_float_array(y_history)
        horizon = int(prediction_length or self.cfg.prediction_length)
        n = int(num_samples or self.cfg.num_samples)

        # Chronos pipelines accept a list/array of series; we pass a single series
        # Most versions return a torch.Tensor of shape (batch, num_samples, horizon)
        out = self._pipeline.predict(
            [y],
            prediction_length=horizon,
            num_samples=n,
        )

        # Convert to numpy: shape (num_samples, horizon)
        if hasattr(out, "detach"):
            out_np = out.detach().cpu().numpy()
        else:
            out_np = np.asarray(out)

        # handle (1, n, h) or (n, h)
        if out_np.ndim == 3:
            out_np = out_np[0]
        return out_np

    def predict_quantiles(
        self,
        y_history: ArrayLike,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        prediction_length: Optional[int] = None,
    ) -> Dict[float, np.ndarray]:
        y = self._to_1d_float_array(y_history)
        horizon = int(prediction_length or self.cfg.prediction_length)

        # Many versions expose predict_quantiles; if not, we fallback to sampling.
        if hasattr(self._pipeline, "predict_quantiles"):
            out = self._pipeline.predict_quantiles(
                [y],
                prediction_length=horizon,
                quantiles=list(quantiles),
            )
            # typically shape (batch, len(q), horizon)
            if hasattr(out, "detach"):
                out_np = out.detach().cpu().numpy()
            else:
                out_np = np.asarray(out)
            if out_np.ndim == 3:
                out_np = out_np[0]
            return {float(q): out_np[i] for i, q in enumerate(quantiles)}

        # fallback: sample then compute quantiles
        samples = self.predict_samples(y, prediction_length=horizon, num_samples=self.cfg.num_samples)
        return {float(q): np.quantile(samples, q, axis=0) for q in quantiles}

    def predict_mean(
        self,
        y_history: ArrayLike,
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
    ) -> np.ndarray:
        samples = self.predict_samples(
            y_history,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )
        return samples.mean(axis=0)

# =============================================================================
# 6c) Foundation model: Chronos forecaster (zero-shot)
# =============================================================================


def model_chronos(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    """
    Works with BOTH:
      - amazon/chronos-bolt-*  (quantile forecasting; no num_samples in predict())
      - amazon/chronos-t5-*    (supports sampling, but we use predict_quantiles for consistency)

    params supported:
      - model_name: str (default "amazon/chronos-bolt-base")
      - device: "cpu"|"cuda"|"mps"|None
      - torch_dtype: "bfloat16"|"float16"|"float32"|None
      - seed: int (default 42)
      - agg: "mean"|"median"|"pXX"   (default "mean")
           examples: "median", "p50", "p10", "p90"
      - quantile_levels: list[float] (default [0.1, 0.5, 0.9])
      - context_length: int|None (optional truncate history)
    """
    p = params or {}

    model_name = str(p.get("model_name", "amazon/chronos-bolt-base"))
    device = p.get("device", None)
    torch_dtype = p.get("torch_dtype", None)
    seed = int(p.get("seed", 42))
    agg = str(p.get("agg", "mean")).lower()
    q_levels = p.get("quantile_levels", [0.1, 0.5, 0.9])
    context_length = p.get("context_length", None)
    if context_length is not None:
        context_length = int(context_length)

    # history -> 1D float tensor
    y = train.astype(float).to_numpy().reshape(-1)
    if context_length is not None and len(y) > context_length:
        y = y[-context_length:]

    # Build Chronos config for this horizon
    cfg = ChronosConfig(
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
        prediction_length=int(horizon),
        num_samples=0,   # unused here (we use predict_quantiles)
        seed=seed,
    )
    forecaster = ChronosForecaster(cfg)

    # IMPORTANT: use predict_quantiles for Bolt (and works for T5 too)
    # Per library docs: returns (quantiles, mean)
    # quantiles shape: [batch, prediction_length, num_quantiles]
    # mean shape:      [batch, prediction_length]
    quantiles, mean = forecaster._pipeline.predict_quantiles(
        inputs=torch.tensor(y),
        prediction_length=int(horizon),
        quantile_levels=list(q_levels),
    )

    # move to numpy
    if hasattr(mean, "detach"):
        mean_np = mean.detach().cpu().numpy()
        q_np = quantiles.detach().cpu().numpy()
    else:
        mean_np = np.asarray(mean)
        q_np = np.asarray(quantiles)

    # remove batch dim
    mean_np = mean_np[0].reshape(-1)          # (horizon,)
    q_np = q_np[0]                            # (horizon, num_q)

    # Aggregate to deterministic forecast for your MSE pipeline
    if agg == "mean":
        yhat = mean_np
    elif agg in ("median", "p50", "q50"):
        # find quantile level closest to 0.5; fallback to mean if not present
        idx = int(np.argmin(np.abs(np.asarray(q_levels, dtype=float) - 0.5)))
        yhat = q_np[:, idx].reshape(-1)
    elif agg.startswith("p") and agg[1:].isdigit():
        target = int(agg[1:]) / 100.0
        idx = int(np.argmin(np.abs(np.asarray(q_levels, dtype=float) - target)))
        yhat = q_np[:, idx].reshape(-1)
    else:
        raise ValueError("agg must be one of: 'mean', 'median', 'pXX' (e.g. 'p10', 'p90').")

    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    
    if len(yhat) != horizon:
        raise ValueError(f"Chronos returned wrong horizon length: {len(yhat)} != {horizon}")
    if not np.all(np.isfinite(yhat)):
        raise ValueError("Chronos produced non-finite values (NaN/inf).")

    return yhat

# =============================================================================
# 7) Execution: integrate improved data processing + tuning + final test eval
# =============================================================================

TargetSeries = Literal["concurrency", "resources", "throughput"]
ConcurrencyVariant = Literal["hourly_sweepline", "exact_changepoints", "event_sampled"]
ResourceVariant = Literal["observed_resources_per_bin", "event_rate_per_bin"]
TTMethod = Literal["row", "span", "rolling", "none"]

ModelKey = Literal[
    "naive",
    "seasonal_naive",
    "ses",
    "holt",
    "ets",
    "sarimax",
    "ridge",
    "gru",
    "nbeats",
    "chronos",
    "chronos": ("Chronos", model_chronos),
]


def _default_param_grid(
    model_key: ModelKey,# type: ignore
    *,
    season_candidates: List[int],
    ets_season_candidates: List[int],
    sarimax_season_candidates: List[int],
    ridge_lag_candidates: List[int],
    ridge_alpha_candidates: List[float],
    gru_candidates: List[dict],
    nbeats_candidates: List[dict],
) -> List[Optional[dict]]:
    """
    Build a parameter grid per model.

    Returns:
      - list of dicts (candidate param sets)
      - for models without params -> [None]
    """
    if model_key == "naive":
        return [None]

    if model_key == "seasonal_naive":
        return [{"season": s} for s in season_candidates]

    if model_key in ("ses", "holt"):
        return [None]

    if model_key == "ets":
        # Keep ETS grid small to avoid long runtimes.
        #
        # ETS params:
        # - use_boxcox affects how the model handles variance stabilization.
        #   It requires strictly positive data -> we handle this inside model_ets by shifting if needed.
        #
        # NOTE: Some combinations (mul) require positive values and are fragile on zero-heavy series,
        # so we keep only additive seasonality by default.
        grid: List[dict] = []
        for sp in ets_season_candidates:
            for use_boxcox in [False, True]:
                # Common strong default: additive trend + additive seasonality
                grid.append({
                    "trend": "add",
                    "seasonal": "add",
                    "seasonal_periods": sp,
                    "use_boxcox": use_boxcox,
                    "damped_trend": False,
                    "remove_bias": False,
                })
                # No trend (often good if the series is stationary-ish)
                grid.append({
                    "trend": None,
                    "seasonal": "add",
                    "seasonal_periods": sp,
                    "use_boxcox": use_boxcox,
                    "damped_trend": False,
                    "remove_bias": False,
                })
                # Damped trend variant (sometimes helps avoid explosive trend extrapolation)
                grid.append({
                    "trend": "add",
                    "seasonal": "add",
                    "seasonal_periods": sp,
                    "use_boxcox": use_boxcox,
                    "damped_trend": True,
                    "remove_bias": False,
                })
        return grid

    if model_key == "sarimax":
        # Small SARIMAX grid: a few classical candidates.
        orders = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
        seasonal_base = [(0, 0, 0), (1, 0, 1)]
        grid: List[dict] = []
        for order in orders:
            for (P, D, Q) in seasonal_base:
                for s in sarimax_season_candidates:
                    grid.append({"order": order, "seasonal_order": (P, D, Q, s)})
        return grid

    if model_key == "ridge":
        grid: List[dict] = []
        for lags in ridge_lag_candidates:
            for alpha in ridge_alpha_candidates:
                grid.append({"lags": lags, "alpha": alpha, "add_time_features": True})
        return grid

    if model_key == "gru":
        return gru_candidates
    
    if model_key == "nbeats":
        return nbeats_candidates

    if model_key == "chronos":
        return [
            {
                "model_name": "amazon/chronos-bolt-base",
                "num_samples": 200,
                "seed": 42,
                "agg": "mean",
                # "context_length": 5000,  # optional
            },
            {
                "model_name": "amazon/chronos-bolt-base",
                "num_samples": 200,
                "seed": 42,
                "agg": "median",
            },
            # Optionally try original Chronos-T5 (slower):
            # {"model_name": "amazon/chronos-t5-base", "num_samples": 200, "seed": 42, "agg": "mean"},
        ]

    raise ValueError(f"No grid builder for model_key={model_key}")

def _fingerprint(s: pd.Series, name: str):
    import pandas as pd, numpy as np, hashlib
    arr = np.asarray(s.values, dtype=float)
    idx = s.index.astype("int64") if isinstance(s.index, pd.DatetimeIndex) else np.arange(len(s))
    h = hashlib.md5(np.concatenate([idx.astype("int64"), np.nan_to_num(arr, nan=1e308).view("int64")]).tobytes()).hexdigest()
    print(f"{name}: len={len(s)} start={s.index[0]} end={s.index[-1]} md5={h}")


def _execute(
    *,
    # ----------------------------
    # Data / processing parameters
    # ----------------------------
    root_path: str = "/Volumes/Daniel/Thesis/resources",
    dataset: str = "BPI_2012/BPI_Challenge_2012.xes",
    cache_dir: Optional[str] = None,
    utc: bool = True,
    freq: str = "1S",
    target_series: TargetSeries = "concurrency",
    concurrency_variant: ConcurrencyVariant = "hourly_sweepline",
    resource_variant: ResourceVariant = "observed_resources_per_bin",
    throughput_method: TTMethod = "row",
    throughput_param: Union[int, str] = 100,

    # ----------------------------
    # Split and feasibility controls
    # ----------------------------
    split_cfg: Split3WayConfig = Split3WayConfig(0.7, 0.1, 0.2),
    split_timestamp: str = None,
    max_n: int = 200_000,
    recompute: bool = False,

    # ----------------------------
    # Model selection
    # ----------------------------
    models_to_run: Optional[List[ModelKey]] = None, # type: ignore
    models_to_plot: Optional[List[ModelKey]] = None, # type: ignore
    plot: bool = True,
    verbose_tuning: bool = True,

    # ----------------------------
    # Hyperparameter candidate lists (customize depending on freq)
    # ----------------------------
    season_candidates: Optional[List[int]] = None,
    ets_season_candidates: Optional[List[int]] = None,
    sarimax_season_candidates: Optional[List[int]] = None,
    ridge_lag_candidates: Optional[List[int]] = None,
    ridge_alpha_candidates: Optional[List[float]] = None,
    gru_candidates: Optional[List[dict]] = None,
    nbeats_candidates: Optional[List[dict]] = None,
) -> Dict[str, object]:
    """
    Full pipeline with validation-based hyperparameter tuning.

    Steps:
    1) Load XES -> dataframe
    2) Build chosen target time series at requested frequency (default 1-second)
    3) Preprocess series
    4) Split into train/val/test (time-ordered)
    5) For each model:
        - Tune hyperparams on validation set
        - Refit on train+val
        - Evaluate on test
    6) Optionally plot selected models on test horizon

    Returns a dictionary with:
      - series, train, val, test
      - best_params: model_name -> dict|None
      - val_mse: model_name -> float
      - test_mse: model_name -> float
      - test_preds: model_name -> np.ndarray (only for plotted models)
    """
    # ----------------------------
    # Import improved data processing pipeline
    # ----------------------------
    from pathlib import Path
    from data_processing_improved import (
        Config, import_xes,
        build_concurrency_series,
        build_resource_utilization_series,
        build_throughput_time_series,
    )

    # ----------------------------
    # Candidate defaults (sane for 1-second sampling + compute)
    # ----------------------------
    # For 1-second data:
    # - 60 = 1 minute
    # - 300 = 5 minutes
    # - 3600 = 1 hour (can be expensive for ETS/SARIMAX depending on length)
    if season_candidates is None:
        season_candidates = [60, 300, 3600]
    if ets_season_candidates is None:
        ets_season_candidates = [60, 300]        # keep ETS smaller by default
    if sarimax_season_candidates is None:
        sarimax_season_candidates = [60, 300]    # keep SARIMAX smaller by default
    if ridge_lag_candidates is None:
        ridge_lag_candidates = [60, 300, 600]    # 1m, 5m, 10m
    if ridge_alpha_candidates is None:
        ridge_alpha_candidates = [0.1, 1.0, 10.0]
    if gru_candidates is None:
        # Keep GRU grid very small (costly). Expand only if you have GPU/time.
        gru_candidates = [
            {"n_steps": 300, "n_future": 30, "hidden_size": 64, "epochs": 20, "lr": 1e-3, "batch_size": 256, "seed": 0},
            {"n_steps": 600, "n_future": 30, "hidden_size": 64, "epochs": 20, "lr": 1e-3, "batch_size": 256, "seed": 0},
        ]
    if nbeats_candidates is None:
        # Keep small: N-BEATS is heavier than Ridge and similar-ish to GRU.
        # For hourly data, n_steps around 168 (1 week) is a natural candidate.
        nbeats_candidates = [
            {
                "n_steps": 168,
                "n_blocks": 3,
                "hidden_size": 256,
                "n_layers": 2,
                "theta_size": 128,
                "dropout": 0.0,
                "epochs": 30,
                "lr": 1e-3,
                "batch_size": 256,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "seed": 0,
                "patience": 0,
            },
            {
                "n_steps": 336,
                "n_blocks": 3,
                "hidden_size": 256,
                "n_layers": 2,
                "theta_size": 128,
                "dropout": 0.0,
                "epochs": 30,
                "lr": 1e-3,
                "batch_size": 256,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "seed": 0,
                "patience": 0,
            },
            {
                "n_steps": 168,
                "n_blocks": 4,
                "hidden_size": 512,
                "n_layers": 2,
                "theta_size": 256,
                "dropout": 0.1,
                "epochs": 30,
                "lr": 5e-4,
                "batch_size": 256,
                "weight_decay": 1e-4,
                "grad_clip": 1.0,
                "seed": 0,
                "patience": 0,
            },
        ]
    # ----------------------------
    # Configure caching and load XES
    # ----------------------------
    cache_dir_path = Path(cache_dir) if cache_dir is not None else (Path.home() / "tmp" / "pm4py_cache")

    cfg = Config(
        root=Path(root_path),
        dataset=dataset,
        cache_dir=cache_dir_path,
        utc=utc,
    )

    df = import_xes(cfg, from_file=not recompute)
    #print(df.sort_values(by=["time:timestamp"]))
    # ----------------------------
    # Build chosen target time series
    # ----------------------------
    from_file_flag = not recompute
    
    if target_series == "concurrency":
        series = build_concurrency_series(
            cfg, df,
            from_file=from_file_flag,
            variant=concurrency_variant,
            freq=freq,
            keep_datetime_index=True,
        )
        series.name = f"concurrency_{freq}"

    elif target_series == "resources":
        series = build_resource_utilization_series(
            cfg, df,
            from_file=from_file_flag,
            variant=resource_variant,
            freq=freq,
            smoothing=True,
        )
        series.name = f"resources_{resource_variant}_{freq}"

    elif target_series == "throughput":
        series = build_throughput_time_series(
            cfg, df,
            from_file=from_file_flag,
            method=throughput_method,
            method_param=throughput_param,
            freq=freq,
            smoothing=True,
        )
        series.name = f"throughput_{throughput_method}_{freq}"

    else:
        raise ValueError(f"Unknown target_series: {target_series}")
    # Cap length (especially for 1-second data)
    #if max_n is not None and len(series) > max_n:
    #    series = series.iloc[-max_n:]
    # Try to enforce .freq metadata (helps time-features in Ridge)
    try:
        series = series.asfreq(freq, method="ffill")
    except Exception:
        pass

    # Preprocess values
    series = preprocess_series(series)

    # Split into train/val/test
    train, val, test = split_train_val_test(series, split_cfg, split_timestamp)
    _fingerprint(series, "SERIES")
    #_fingerprint(train, "TRAIN")
    #_fingerprint(val, "VALIDATION")
    #_fingerprint(test, "TEST")

    # Truncate test
    test = test[:-300]

    # ----------------------------
    # Model registry
    # ----------------------------
    available: Dict[ModelKey, Tuple[str, ForecastFn]] = {# type: ignore
        "naive": ("Naive", model_naive),
        "seasonal_naive": ("SeasonalNaive", model_seasonal_naive),
        "ses": ("SES", model_ses),
        "holt": ("Holt", model_holt),
        "ets": ("ETS", model_ets),
        "sarimax": ("SARIMAX", model_sarimax),
        "ridge": ("Ridge", model_ridge_lags_sktime),
        "gru": ("GRU", model_gru),
        "nbeats": ("N-BEATS", model_nbeats),
        "chronos": ("Chronos", model_chronos)
    }

    if models_to_run is None:
        models_to_run = list(available.keys())

    # Validate selection
    unknown = [m for m in models_to_run if m not in available]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Allowed: {list(available.keys())}")

    # Default plot subset (must be in models_to_run)
    if models_to_plot is None:
        models_to_plot = [m for m in list(available.keys()) if m in models_to_run]

    # Validate plot selection
    unknown_plot = [m for m in models_to_plot if m not in available]
    if unknown_plot:
        raise ValueError(f"Unknown plot model keys: {unknown_plot}. Allowed: {list(available.keys())}")

    # ----------------------------
    # Hyperparameter tuning and final test evaluation
    # ----------------------------
    best_params: Dict[str, Optional[dict]] = {}
    val_mse: Dict[str, float] = {}
    test_mse: Dict[str, float] = {}
    test_preds_for_plot: Dict[str, np.ndarray] = {}

    # Concatenate train+val for final refit
    train_plus_val = pd.concat([train, val])

    for key in models_to_run:
        human_base, fn = available[key]

        # Build a parameter grid for this model
        grid = _default_param_grid(
            key,
            season_candidates=season_candidates,
            ets_season_candidates=ets_season_candidates,
            sarimax_season_candidates=sarimax_season_candidates,
            ridge_lag_candidates=ridge_lag_candidates,
            ridge_alpha_candidates=ridge_alpha_candidates,
            gru_candidates=gru_candidates,
            nbeats_candidates=nbeats_candidates,
        )

        # Make human-readable name include tuned elements where relevant
        # (We will later store final names like SARIMAX(order=..., seasonal=...))
        print(f"\n[TUNE] Model: {human_base} ({key})  grid_size={len(grid)}")

        # Tune on validation
        bp, bscore = tune_on_validation(fn, train, val, grid, verbose=verbose_tuning)
        best_params[key] = bp
        val_mse[key] = bscore

        # Refit on train+val and evaluate on test
        try:
            yhat_test = safe_predict(fn, train_plus_val, len(test), bp)
            test_mse[key] = mse(test, yhat_test)
        except Exception:
            traceback.print_exc()
            yhat_test = np.full(len(test), np.nan, dtype=float)
            test_mse[key] = float("nan")

        # Keep predictions for plotting if requested
        if plot and key in models_to_plot:
            # Build a readable label
            label = human_base
            if bp is not None:
                # include key hyperparams compactly
                if key == "seasonal_naive":
                    label = f"SeasonalNaive(season={bp.get('season')})"
                elif key == "ets":
                    label = f"ETS(sp={bp.get('seasonal_periods')}, trend={bp.get('trend')}, seas={bp.get('seasonal')})"
                elif key == "sarimax":
                    label = f"SARIMAX(order={bp.get('order')}, seas={bp.get('seasonal_order')})"
                elif key == "ridge":
                    label = f"Ridge(lags={bp.get('lags')}, alpha={bp.get('alpha')})"
                elif key == "gru":
                    label = f"GRU(n_steps={bp.get('n_steps')}, hidden={bp.get('hidden_size')})"
            test_preds_for_plot[label] = yhat_test

        print(f"[RESULT] {human_base} best_val_MSE={val_mse[key]:.6f}  test_MSE={test_mse[key]:.6f}  best_params={bp}")

    # Plot comparisons on test set
    if plot and test_preds_for_plot:
        #plot_forecasts(
        #    test,
        #    test_preds_for_plot,
        #    title=f"Test forecasts (target={series.name}, freq={freq})",
        #)
        plot_train_test_with_forecasts(
            train=train,
            val=val,
            test=test,
            preds=test_preds_for_plot,
            title=f"Train/Val/Test + forecasts (target={series.name}, freq={freq})",
            show_val=True,   # set False if you want to visually treat val as part of train
        )

    return {
        "series": series,
        "train": train,
        "val": val,
        "test": test,
        "best_params": best_params,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "test_preds": test_preds_for_plot,
    }


# =============================================================================
# 8) Main
# =============================================================================

def main() -> None:
    """
    Example main calling _execute with:
    - improved processing
    - 1-second sampling
    - a moderate model set (fast enough to iterate)
    - validation-based tuning

    Tip:
    - For very large series, reduce max_n.
    - For very expensive stats models at 1-second, keep seasonal candidates small (e.g., [60, 300]).
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    out = _execute( 
        # Processing
        freq="1H",
        target_series="throughput",
        concurrency_variant="hourly_sweepline",
        throughput_method="row",
        throughput_param=100,
        max_n=200_000,
        recompute=False,

        # Model selection
        models_to_run=["ses", "holt", "ets", "sarimax", "ridge", "gru"],
        models_to_plot=None,#["seasonal_naive", "ridge", "sarimax"],

        # Splits
        split_cfg=Split3WayConfig(train_frac=0.7, val_frac=0.1, test_frac=0.2),
        split_timestamp="2012-02-10 12:00:00+00:00",

        # Hyperparam candidates
        season_candidates=[24, 168],      # seasonal naive candidates
        ets_season_candidates=[24, 168],        # keep ETS manageable
        sarimax_season_candidates=[24, 168],    # keep SARIMAX manageable
        ridge_lag_candidates=[24, 48, 168],
        ridge_alpha_candidates=[0.1, 1.0, 10.0],

        # GRU candidates 
        gru_candidates=[
            {"n_steps": 168, "n_future": 24, "hidden_size": 64, "epochs": 30, "lr": 1e-3, "batch_size": 128, "seed": 0},
            {"n_steps": 336, "n_future": 24, "hidden_size": 64, "epochs": 30, "lr": 1e-3, "batch_size": 128, "seed": 0},
        ],

        plot=True,
        verbose_tuning=True,
    )

    print("\n=== FINAL TEST MSE (sorted) ===")
    items = sorted(out["test_mse"].items(), key=lambda kv: (np.isnan(kv[1]), kv[1]))
    for key, score in items:
        print(f"{key:15s}  test_MSE={score:.6f}  best_params={out['best_params'][key]}")

    print("\n=== FINAL VALIDATION MSE (sorted) ===")
    items = sorted(out["val_mse"].items(), key=lambda kv: (np.isnan(kv[1]), kv[1]))
    for key, score in items:
        print(f"{key:15s}  val_MSE={score:.6f}  best_params={out['best_params'][key]}")


if __name__ == "__main__":
    main()
