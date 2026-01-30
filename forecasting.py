import pandas as pd
from sklearn.metrics import mean_squared_error
import traceback
import matplotlib.pyplot as plt
import numpy as np
import sys, math, os, pickle
import torch
import torch.nn as nn
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

import data_processing


# Forecast on process level with given model and parameters
def forecast(data: pd.Series, model, parameters, train_split=0.8):
    print('Forecast start')

    # Index of the last element of training series
    index_train = int(len(data) * train_split)

    # Build Train ans Test Series
    # Train: From 0 to (exclusive) index_train
    time_series_train = data[0:index_train]
    time_series_test = data[index_train:]

    # Calculate forecast horizon
    steps = len(time_series_test)

    # Apply model to time series and store result
    print('Forecast model')
    forecast = model(time_series_train, steps, parameters)

    # return y_true, y_pred
    return (time_series_test, forecast)

def calculate_mse_results_hp_tuning(data: pd.Series, hp_tuning_function, train_split=0.8):
    results = hp_tuning_function(data, train_split)
    for r in results:
        print("Parameters:", r[0], " - MSE =", r[1])
    return results

def calculate_mse_result(data: pd.Series, model, train_split=0.8):
    # Forecast
    try:
        y_true, y_pred = forecast(data, model, None, train_split)
        # Calculate MSE
        mse = mean_squared_error(y_true, y_pred)
        # Add to results
    except Exception as e:
        print(traceback.print_exc())
        mse = float('nan')
    print(mse)
    return mse

def plot_results(data: pd.Series, models: list, train_split=0.8):
    # Index of the last element of training series
    index_train = int(len(data) * train_split)
    time_series_test = data[index_train:]

    time = np.arange(len(time_series_test))
    plt.figure(figsize=(8,4))

    plt.plot(time, time_series_test, label='True future')
    # Forecast and add to plot
    for model in models:
        try:
            y_true, y_pred = forecast(data, model, None, train_split)
            plt.plot(time, y_pred, label='Predicted future for model ' + model.__name__, linestyle='--')
        except Exception as e:
            print(traceback.print_exc())
            print('Model', model, 'did not work')
    
    # Plot results
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Results')
    plt.show()


# ----------- Statistical models ----------

def apply_arima_model(data: pd.Series, horizon: pd.Timedelta, parameters: list):
    # Dummy parameters
    if parameters is None:
        parameters = [1, 1, 1, 1, 1, 1, 12]
    
    # Define ARIMA model
    model = ARIMA(data, order=(parameters[0], parameters[1], parameters[2]), seasonal_order=(parameters[3], parameters[4], parameters[5], parameters[6]))
    result = model.fit()
    print(result.summary())
    # Forecast
    forecast = result.forecast(horizon) # Can be tested with additional alpha (confidence interval)
    # Return predicted values
    return forecast

def _hp_tuning_arima(data: pd.Series, train_split=0.8):
    results = []
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                 for P in range(0, 3):
                     for D in range(0, 2):
                         for Q in range(0, 3):
                             for s in [24, 12]: # Seasonality (12 - half day; 24 - day)
                                # Build parameters
                                parameters = [p, d, q, P, D, Q, s]
                                print(parameters)
                                # Forecast
                                try:
                                    y_true, y_pred = forecast(data, apply_arima_model, parameters, train_split=train_split)
                                    # Calculate MSE
                                    mse = mean_squared_error(y_true, y_pred)
                                    # Add to results
                                except:
                                    mse = float('nan')
                                r = [parameters, mse]
                                results.append(r)
    return results

def _hp_tuning_holt_winter(data: pd.Series, train_split=0.8):
    results = []
    for trend in ["add", "mul", None]:
        for seasonal in ["add", "mul", None]:
            for seasonal_periods in [24, 12]:
                for box_cox in[True, False]:
                    for remove_bias in [True, False]:
                        for use_brute in [True, False]:
                            # Build parameters
                                parameters = [trend, seasonal, seasonal_periods, box_cox, remove_bias, use_brute]
                                print(parameters)
                                # Forecast
                                try:
                                    y_true, y_pred = forecast(data, apply_holt_winter_model, parameters, train_split=train_split)
                                    # Calculate MSE
                                    mse = mean_squared_error(y_true, y_pred)
                                    # Add to results
                                except Exception as e:
                                    mse = float('nan')
                                r = [parameters, mse]
                                results.append(r)
    return results

def apply_ses_model(ts: pd.Series, horizon: int, parameters: list):
    model = SimpleExpSmoothing(ts).fit(optimized=True)
    forecast = model.forecast(horizon)
    # Return predicted values
    return forecast

def apply_holt_linear_model(ts: pd.Series, horizon: int, parameters: list):
    model = Holt(ts).fit(optimized=True)
    forecast = model.forecast(horizon)
    # Return predicted values
    return forecast

def apply_holt_winter_model(ts: pd.Series, horizon: int, parameters: list):
    # Dummy parameters if parameters is None
    if parameters is None:
        parameters = ["add", "add", 12, False, False, True]

    # If box cox is used, 0-values are not allowed - shift series
    if parameters[3]:
        ts = ts + 1
    
    model = ExponentialSmoothing(ts, trend=parameters[0], seasonal=parameters[1], seasonal_periods=parameters[2], use_boxcox=parameters[3]).fit(remove_bias=parameters[4], use_brute=parameters[5], optimized=True)
    forecast = model.forecast(horizon)
    # Return predicted values
    return forecast


# ---------- Simple ML models ----------

# Simple RNN
class SimpleRNNForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=5):
        super(SimpleRNNForecaster, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)              # out: (batch, seq_len, hidden)
        out = out[:, -1, :]               # take last time step’s output
        out = self.fc(out)                # project to future values
        return out

def create_dataset(series, n_steps=20, n_future=5):
    X, y = [], []
    for i in range(len(series) - n_steps - n_future):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps:i+n_steps+n_future])
    return np.array(X), np.array(y)

def forecast_rnn(model, ts, n_steps, n_future, steps_ahead=50):
    model.eval()
    preds = []
    series = ts.to_numpy()
    input_seq = torch.tensor(series[-n_steps:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    for _ in range(math.ceil(steps_ahead / n_future)):
        with torch.no_grad():
            next_vals = model(input_seq).numpy().flatten()
        preds.extend(next_vals)
        # shift window forward
        new_input = np.concatenate([input_seq.numpy().flatten()[n_future:], next_vals])
        input_seq = torch.tensor(new_input, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    return preds[:steps_ahead]

def apply_rnn_model(ts: pd.Series, horizon: int, parameters: list):
    # If no parameters given, use dummy
    if parameters is None:
        parameters = [20, 5]
    
    # Create rnn dataset - n_steps as parameters[0] - n_future as parameters[1]
    X, y = create_dataset(ts, parameters[0], parameters[1])

    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # shape: (samples, n_steps, 1)
    y = torch.tensor(y, dtype=torch.float32)                # shape: (samples, n_future)

    # Train model
    model = SimpleRNNForecaster(input_size=1, hidden_size=32, output_size=parameters[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_epochs = 200
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
    forecast = forecast_rnn(model, ts, parameters[0], parameters[1], steps_ahead=horizon)
    return forecast


if __name__ == "__main__":
    data = data_processing.build_concurrency_data(None, from_f=True)
    plot_results(data, [apply_arima_model, apply_holt_linear_model, apply_holt_winter_model, apply_ses_model, apply_rnn_model], 0.8)


