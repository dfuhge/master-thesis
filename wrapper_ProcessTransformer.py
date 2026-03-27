"""
Wrapper executing ProcessTransformer repository.
"""

from __future__ import annotations

import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Literal

import pandas as pd
import ast


TaskName = Literal["next_activity", "next_time", "remaining_time"]
PT_TIMESTAMP_FORMAT = "%Y-%d-%m %H:%M:%S.%f%z"  # matches repo hard-coded format

PROCESS_TRANSFORMER_REPO = "/Path/to/ProcessTransformer/repo"  # repo root containing data_processing.py, next_activity.py, ...
sys.path.append(str(PROCESS_TRANSFORMER_REPO))

from processtransformer.data.processor import LogsDataProcessor  # type: ignore (local import from repo)
from processtransformer.constants import Task  # type: ignore (local import from repo)
from processtransformer.models import transformer  # type: ignore (local import from repo)
from processtransformer.data.loader import LogsDataLoader  # type: ignore (local import from repo)

import tensorflow as tf
from sklearn import metrics 
import numpy as np

@dataclass(frozen=True)
class ProcessTransformerPaths:
    """
    Paths for the ProcessTransformer repository.
    """
    repo_dir: Path
    datasets_dir: Path
    models_dir: Path
    results_dir: Path

    @staticmethod
    def from_repo(repo_dir: str | Path) -> "ProcessTransformerPaths":
        repo_dir = Path(repo_dir).expanduser().resolve()
        return ProcessTransformerPaths(
            repo_dir=repo_dir,
            datasets_dir=repo_dir / "datasets",
            models_dir=repo_dir / "models",
            results_dir=repo_dir / "results",
        )


def load_event_log_as_dataframe( path: str | Path, columns_map: Optional[Dict[str, str]] = None, timestamp_col: str = "time:timestamp", csv_timeformat: Optional[str] = None) -> pd.DataFrame:
    """
    Load an event log from .xes or .csv into a DataFrame, ensuring the timestamp column is parsed as datetime.
    
    - path: Path to the event log file (.xes or .csv).
    - columns_map: Optional mapping to rename columns after loading (e.g., {"old_name": "new_name"}).
    - timestamp_col: Name of the timestamp column (after renaming if columns_map is provided).
    - csv_timeformat: Optional strftime format for parsing timestamps in CSV files (e.g., "%Y-%m-%d %H:%M:%S"). If None, pandas will attempt to infer the format.
    """
    path = Path(path).expanduser().resolve()
    suffix = path.suffix.lower()

    if suffix == ".xes":
        import pm4py  # local import on purpose
        log = pm4py.read_xes(str(path))
        df = pm4py.convert_to_dataframe(log)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix} (expected .xes or .csv)")

    if columns_map:
        df = df.rename(columns=columns_map)

    if timestamp_col not in df.columns:
        raise KeyError(
            f"timestamp_col '{timestamp_col}' not found. Available columns: {list(df.columns)}"
        )

    if csv_timeformat is not None:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=csv_timeformat, errors="raise")
    else:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="raise")

    return df


def export_processtransformer_raw_csv(
    df: pd.DataFrame,
    out_csv_path: str | Path,
    case_col: str,
    activity_col: str,
    timestamp_col: str,
    sort_by: Optional[Tuple[str, ...]] = None,
    index: bool = False,
) -> str:
    """
   Export a DataFrame to a CSV format compatible with ProcessTransformer's data_processing.py.
    The output CSV will have columns renamed to "Case ID", "Activity", and "Complete Timestamp" as expected by the repo. Timestamps will be parsed and serialized in a consistent format.

    - df: Input DataFrame containing the event log data.
    - out_csv_path: Path where the output CSV should be saved.
    - case_col: Name of the column in df that identifies the case ID.
    - activity_col: Name of the column in df that identifies the activity name.
    - timestamp_col: Name of the column in df that contains the event timestamps.
    - sort_by: Optional tuple of column names to sort the output by (e.g., ("Complete Timestamp", "Case ID")). Sorting is done in a stable way to preserve original order where possible.
    - index: Whether to include the DataFrame index in the output CSV (default False).
    """
    out_csv_path = Path(out_csv_path).expanduser().resolve()
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    for col in (case_col, activity_col, timestamp_col):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing. Available: {list(df.columns)}")

    out = df[[case_col, activity_col, timestamp_col]].copy()
    out = out.rename(
        columns={
            case_col: "Case ID",
            activity_col: "Activity",
            timestamp_col: "Complete Timestamp",
        }
    )

    # Handle missing values in a conservative way (avoid crashing downstream tokenization)
    out["Activity"] = out["Activity"].astype("string").fillna("unk")
    out["Case ID"] = out["Case ID"].astype("string").fillna("unk")

    # Ensure timestamps are serialized consistently
    out["Complete Timestamp"] = pd.to_datetime(out["Complete Timestamp"],
                                                format="mixed",
                                                errors="coerce")

    if sort_by:
        out = out.sort_values(list(sort_by), kind="mergesort")

    out.to_csv(out_csv_path, index=index)
    return str(out_csv_path)


class ProcessTransformerWrapper:


    def __init__(
        self,
        repo_dir: str | Path,
        python_executable: str = sys.executable,
        env: Optional[Dict[str, str]] = None,
        split_timestamp = None,
        last_timestamp = None
    ):
        self.paths = ProcessTransformerPaths.from_repo(repo_dir)
        self.python = python_executable
        self.env = dict(os.environ, **(env or {}))
        self.split_timestamp = split_timestamp
        self.last_timestamp = last_timestamp

        if not self.paths.repo_dir.exists():
            raise FileNotFoundError(f"ProcessTransformer repo_dir not found: {self.paths.repo_dir}")

        # We call the repo scripts by filename; make sure they exist.
        for script in ("data_processing.py", "next_activity.py", "next_time.py", "remaining_time.py"):
            p = self.paths.repo_dir / script
            if not p.exists():
                raise FileNotFoundError(
                    f"Expected script '{script}' not found in repo root: {p}"
                )

    def dataset_dir(self, dataset_name: str) -> Path:
        return self.paths.datasets_dir / dataset_name

    def raw_csv_path(self, dataset_name: str, filename: Optional[str] = None) -> Path:
        """
        CSV path.
        """
        if filename is None:
            filename = f"{dataset_name}.csv"
        return self.dataset_dir(dataset_name) / filename

    def prepare_dataset_csv(
        self,
        event_log_path: str | Path,
        dataset_name: str,
        *,
        # How to read/rename your input:
        columns_map: Optional[Dict[str, str]] = None,
        input_timestamp_col: str = "time:timestamp",
        csv_timeformat: Optional[str] = None,
        # Which columns to use after rename:
        case_col: str = "case:concept:name",
        activity_col: str = "concept:name",
        timestamp_col: Optional[str] = None,
        # Output options:
        sort_by_timestamp_then_case: bool = True,
        index: bool = False,
    ) -> str:
        """
        Create a CSV file in the format expected by ProcessTransformer's data_processing.py, with columns renamed to "Case ID", "Activity", and "Complete Timestamp". Timestamps will be parsed and serialized in a consistent format.
        
        - event_log_path: Path to the input event log file (.xes or .csv).
        - dataset_name: Name of the dataset (used for output path).
        - columns_map: Optional mapping to rename columns after loading (e.g., {"old_name": "new_name"}).
        - input_timestamp_col: Name of the timestamp column in the input file (before renaming).
        - csv_timeformat: Optional strftime format for parsing timestamps in CSV files (e.g., "%Y-%m-%d %H:%M:%S"). If None, pandas will attempt to infer the format.
        - case_col: Name of the column in the DataFrame that identifies the case ID (after renaming).
        - activity_col: Name of the column in the DataFrame that identifies the activity name (after renaming).
        - timestamp_col: Name of the column in the DataFrame that contains the event timestamps (after renaming). If None, defaults to input_timestamp_col.
        - sort_by_timestamp_then_case: Whether to sort the output CSV by timestamp and then case ID (default True). Sorting is done in a stable way to preserve original order where possible.
        - index: Whether to include the DataFrame index in the output CSV (default False).
        """
        df = load_event_log_as_dataframe(
            event_log_path,
            columns_map=columns_map,
            timestamp_col=input_timestamp_col,
            csv_timeformat=csv_timeformat,
        )

        # If user renamed timestamp_col in columns_map, allow overriding it here
        ts_col = timestamp_col or input_timestamp_col

        out_csv = self.raw_csv_path(dataset_name)
        sort_by = ("Complete Timestamp", "Case ID") if sort_by_timestamp_then_case else None

        # export helper expects sort_by column names AFTER renaming;
        # easiest: sort in original space first (stable), then export.
        if sort_by_timestamp_then_case:
            if ts_col not in df.columns:
                raise KeyError(f"timestamp_col '{ts_col}' not found after renaming.")
            if case_col not in df.columns:
                raise KeyError(f"case_col '{case_col}' not found after renaming.")
            df = df.sort_values([ts_col, case_col], kind="mergesort")

        return export_processtransformer_raw_csv(
            df=df,
            out_csv_path=out_csv,
            case_col=case_col,
            activity_col=activity_col,
            timestamp_col=ts_col,
            sort_by=None,  # already sorted above if requested
            index=index,
        )


    def run_preprocessing(self, dataset_name: str, task: TaskName, *, raw_log_file: Optional[str | Path] = None, dir_path: Optional[str | Path] = None, sort_temporally: bool = False, only_full_traces: bool = True) -> None:
        """
        Preprocess the raw CSV log using ProcessTransformer's data_processing.py. This will read the raw CSV, process it according to the specified task, and save the processed data in the repo's expected format for model training.

        - dataset_name: Name of the dataset (used to locate raw CSV and determine output paths).
        - task: The prediction task for which to preprocess the data (e.g., "remaining_time").
        - raw_log_file: Optional path to the raw CSV log file. If not provided, it defaults to the path returned by self.raw_csv_path(dataset_name).
        - dir_path: Optional directory path where the processed data should be stored. If not provided, it defaults to self.paths.datasets_dir.
        - sort_temporally: Whether to sort the log temporally during processing (default False).
        - only_full_traces: Whether to include only full traces in the processed data (default True).
        """
        
        raw_log_file = Path(raw_log_file) if raw_log_file else self.raw_csv_path(dataset_name)
        dir_path = Path(dir_path) if dir_path else self.paths.datasets_dir

        # --- Taken from data_processing.p ---
        import time
        
        # Process raw logs
        start = time.time()
        data_processor = LogsDataProcessor(name=dataset_name, 
            filepath=raw_log_file, 
            columns = ["Case ID", "Activity", "Complete Timestamp"], #["case:concept:name", "concept:name", "time:timestamp"], 
            dir_path=dir_path, pool = 1) #changed from 4 to 1
        data_processor.process_logs(task=task, sort_temporally=sort_temporally, split_timestamp=self.split_timestamp, last_timestamp=self.last_timestamp, only_full_traces=only_full_traces)
        end = time.time()
        print(f"Total processing time: {end - start}")
    
    def _call_remaining_time(self, dataset_name: str, model_dir: Path, result_dir: Path, learning_rate: float, epochs: int, batch_size: int) -> None:
        """
        Train and evaluate the ProcessTransformer model for the remaining time prediction task. This method loads the preprocessed data, prepares it for training, creates and trains the transformer model, and evaluates it on the test set across all prefixes (k). The results are saved to CSV files in the specified result directory.

        - dataset_name: Name of the dataset (used to locate preprocessed data and determine output paths).
        - model_dir: Directory where the trained model weights will be saved.
        - result_dir: Directory where the evaluation results will be saved.
        - learning_rate: Learning rate for model training.
        - epochs: Number of epochs to train the model.
        - batch_size: Batch size for model training.

        """
        
        model_path = f"{model_dir}/{dataset_name}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = f"{model_path}/remaining_time_ckpt.weights.h5"

        result_path = f"{result_dir}/{dataset_name}"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        result_path = f"{result_path}/results"

        # Load data
        print("LOAD DATA 1")
        data_loader = LogsDataLoader(name = dataset_name, dir_path="Path/to/preprocessed/data") # Insert actual path to preprocessed data
        print("LOAD DATA 2")
        (train_df, test_df, x_word_dict, y_word_dict, max_case_length, 
            vocab_size, num_output) = data_loader.load_data(Task.REMAINING_TIME)
        print("PREPARE DATA")
        # Prepare training examples for next time prediction task
        (train_token_x, train_time_x, 
            train_token_y, time_scaler, y_scaler) = data_loader.prepare_data_remaining_time(train_df, 
            x_word_dict, max_case_length)
        print("CREATE MODEL")
        # Create and train a transformer model
        transformer_model = transformer.get_remaining_time_model(
            max_case_length=max_case_length, 
            vocab_size=vocab_size)
        print("COMPILE")
        transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), 
            loss=tf.keras.losses.LogCosh())
        print("CALLBACK")
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_weights_only=True,
            monitor="loss", save_best_only=True)
        print("FIT")
        transformer_model.fit([train_token_x, train_time_x], train_token_y, 
            epochs=epochs, batch_size=batch_size, 
            verbose=2, callbacks=[model_checkpoint_callback]) #shuffle=True, 

        # Evaluate over all the prefixes (k) and save the results
        k, maes, mses, rmses = [],[],[],[]

        # D: Added to store results 
        results = {}
        # D: Do case-wise prediction
        case_ids = test_df["case_id"].unique()
        for case_id in case_ids:
            print(case_id)
            test_df_filtered = test_df[test_df["case_id"] == case_id]
            case_results = []
            for i in range(max_case_length):
                test_data_subset = test_df_filtered[test_df_filtered["k"]==i]

                if len(test_data_subset) > 0:
                    #print(test_data_subset)
                    test_token_x, test_time_x, test_y, _, _ = data_loader.prepare_data_remaining_time(
                        test_data_subset, x_word_dict, max_case_length, time_scaler, y_scaler, False)   
                    # Defensive checks
                    if test_token_x is None or test_time_x is None or test_y is None:
                        print(f"Skipping case {case_id}, k={i}: prepare_data returned None")
                        continue

                    if len(test_token_x) == 0 or len(test_time_x) == 0 or len(test_y) == 0:
                        print(f"Skipping case {case_id}, k={i}: empty prepared arrays")
                        print("test_token_x shape:", getattr(test_token_x, "shape", None))
                        print("test_time_x shape:", getattr(test_time_x, "shape", None))
                        print("test_y shape:", getattr(test_y, "shape", None))
                        continue
                    #print(test_token_x)
                    y_pred = transformer_model.predict([test_token_x, test_time_x], verbose=0)
                    _test_y = y_scaler.inverse_transform(test_y)
                    _y_pred = y_scaler.inverse_transform(y_pred)
                    case_results.append(_y_pred)

                    k.append(i)
                    maes.append(metrics.mean_absolute_error(_test_y, _y_pred))
                    mses.append(metrics.mean_squared_error(_test_y, _y_pred))
                    rmses.append(np.sqrt(metrics.mean_squared_error(_test_y, _y_pred)))
                    
            case_res = case_results[-1]
            results[case_id] = case_res


        res = pd.DataFrame(results.items())
        res.to_csv(result_path+"_remaining_time_predictions.csv", index=False)

        k.append(i + 1)
        maes.append(np.mean(maes))
        mses.append(np.mean(mses))
        rmses.append(np.mean(rmses))
        print('Average MAE across all prefixes:', np.mean(maes))
        print('Average MSE across all prefixes:', np.mean(mses))
        print('Average RMSE across all prefixes:', np.mean(rmses))  
        results_df = pd.DataFrame({"k":k, "mean_absolute_error":maes, 
            "mean_squared_error":mses, 
            "root_mean_squared_error":rmses})
        results_df.to_csv(result_path+"_remaining_time.csv", index=False)


    def _improved_call_remaining_time(self, dataset_name: str, model_dir: Path, result_dir: Path, learning_rate: float, epochs: int, batch_size: int) -> None:
        """
        Batched version of the remaining time prediction evaluation. Instead of iterating per case and k, it prepares the entire test set at once, makes predictions in batch, and then attaches predictions back to the original test metadata for evaluation. This should be more efficient while still reproducing the original logic of evaluating per prefix (k) and storing the last non-empty prediction per case. Yields other results than original, therefore not used for experiments.

        - dataset_name: Name of the dataset (used to locate preprocessed data and determine output paths).
        - model_dir: Directory where the trained model weights will be saved.
        - result_dir: Directory where the evaluation results will be saved.
        - learning_rate: Learning rate for model training.
        - epochs: Number of epochs to train the model.
        - batch_size: Batch size for model training.
        """
        model_path = f"{model_dir}/{dataset_name}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = f"{model_path}/remaining_time_ckpt.weights.h5"

        result_path = f"{result_dir}/{dataset_name}"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        result_path = f"{result_path}/results"

        # Load data
        print("LOAD DATA 1")
        data_loader = LogsDataLoader(name = dataset_name, dir_path="/Users/dfuhge/Documents/Studium/Uni Mannheim/Master/Masterarbeit/Masterarbeit Wifo/Code/master-thesis/master-thesis/src/compare/processtransformer-main/datasets")
        print("LOAD DATA 2")
        (train_df, test_df, x_word_dict, y_word_dict, max_case_length, 
            vocab_size, num_output) = data_loader.load_data(Task.REMAINING_TIME)
        print("PREPARE DATA")
        # Prepare training examples for next time prediction task
        (train_token_x, train_time_x, 
            train_token_y, time_scaler, y_scaler) = data_loader.prepare_data_remaining_time(train_df, 
            x_word_dict, max_case_length)
        print("CREATE MODEL")
        # Create and train a transformer model
        transformer_model = transformer.get_remaining_time_model(
            max_case_length=max_case_length, 
            vocab_size=vocab_size)
        print("COMPILE")
        transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), 
            loss=tf.keras.losses.LogCosh())
        print("CALLBACK")
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_weights_only=True,
            monitor="loss", save_best_only=True)
        print("FIT")
        transformer_model.fit([train_token_x, train_time_x], train_token_y, 
            epochs=epochs, batch_size=batch_size, 
            verbose=2, callbacks=[model_checkpoint_callback]) #shuffle=True, 

        # Evaluate over all the prefixes (k) and save the results
        k, maes, mses, rmses = [],[],[],[]

        # D: Added to store results 
        # Keep aligned metadata in the same row order as prepare_data_remaining_time uses
        test_meta = test_df[["case_id", "k"]].reset_index(drop=True).copy()

        # Prepare all test data at once
        test_token_x, test_time_x, test_y, _, _ = data_loader.prepare_data_remaining_time(
            test_df,
            x_word_dict,
            max_case_length,
            time_scaler,
            y_scaler,
            False
        )

        # Defensive checks
        if test_token_x is None or test_time_x is None or test_y is None:
            raise ValueError("prepare_data_remaining_time returned None")

        if len(test_token_x) == 0 or len(test_time_x) == 0 or len(test_y) == 0:
            raise ValueError("Prepared test arrays are empty")

        # Predict once in batch
        y_pred = transformer_model.predict([test_token_x, test_time_x], verbose=0)

        # Inverse-transform once
        test_y_inv = y_scaler.inverse_transform(test_y)
        y_pred_inv = y_scaler.inverse_transform(y_pred)

        # Attach arrays back to metadata
        test_meta["row_idx"] = np.arange(len(test_meta))

        results = {}

        # Reproduce original grouping logic:
        # original code iterates per case, then k from 0..max_case_length-1,
        # and stores the prediction batch of the last non-empty k
        for case_id, case_df in test_meta.groupby("case_id", sort=False):
            case_results = []

            # preserve original behavior: iterate k in ascending order
            for k_val, subset in case_df.groupby("k", sort=True):
                idx = subset["row_idx"].to_numpy()

                _test_y = test_y_inv[idx]
                _y_pred = y_pred_inv[idx]

                case_results.append(_y_pred)

                mse = metrics.mean_squared_error(_test_y, _y_pred)
                k.append(k_val)
                maes.append(metrics.mean_absolute_error(_test_y, _y_pred))
                mses.append(mse)
                rmses.append(np.sqrt(mse))

            if case_results:
                results[case_id] = case_results[-1]

        results = pd.DataFrame(results.items())
        results.to_csv(result_path+"_remaining_time_predictions.csv", index=False)

        #k.append(i + 1)
        #maes.append(np.mean(maes))
        #mses.append(np.mean(mses))
        #rmses.append(np.mean(rmses))
        #print('Average MAE across all prefixes:', np.mean(maes))
        #print('Average MSE across all prefixes:', np.mean(mses))
        #print('Average RMSE across all prefixes:', np.mean(rmses))  
        #results_df = pd.DataFrame({"k":k, "mean_absolute_error":maes, 
        #    "mean_squared_error":mses, 
        #    "root_mean_squared_error":rmses})
        #results_df.to_csv(result_path+"_remaining_time.csv", index=False)



    def train_and_evaluate(self, dataset_name: str, task: TaskName, *, epochs: int = 10, batch_size: int = 12, learning_rate: float = 0.001, gpu: int = 0, model_dir: Optional[str | Path] = None, result_dir: Optional[str | Path] = None,) -> None:
        """
        Train and evaluate the ProcessTransformer model for the specified task. This method orchestrates the entire workflow: it prepares the dataset, trains the model, and evaluates it, saving results to the specified directories.

        - dataset_name: Name of the dataset (used to locate preprocessed data and determine output paths).
        - task: The prediction task to perform (e.g., "remaining_time").
        - epochs: Number of epochs to train the model (default 10).
        - batch_size: Batch size for model training (default 12).
        - learning_rate: Learning rate for model training (default 0.001).
        - gpu: GPU device index to use for training (default 0).
        - model_dir: Directory where the trained model weights will be saved. If not provided, it defaults to self.paths.models_dir.
        - result_dir: Directory where the evaluation results will be saved. If not provided, it defaults to self.paths.results_dir.

        """
        model_dir = Path(model_dir) if model_dir else self.paths.models_dir
        result_dir = Path(result_dir) if result_dir else self.paths.results_dir
        if task == Task.REMAINING_TIME:
            self._call_remaining_time(dataset_name, model_dir, result_dir, learning_rate, epochs, batch_size)
        else:
            raise ValueError(f"Unsupported task: {task}")


def join_results_on_dataset(dataset_path: str, results_path: str, split_timestamp: str, store_path: str, columns: dict) -> pd.DataFrame:
    """
    Join the evaluation results with the dataset based on the case ID. This is used to create a new dataset that includes the predicted remaining time for each case, which can then be used for further analysis or as input to other models.

        - dataset_path: Path to the original event log dataset (.xes or .csv).
        - results_path: Path to the CSV file containing the evaluation results with columns "case_id" and "remaining_time".
        - split_timestamp: The timestamp used to split the original dataset into training and test sets.
        - store_path: Path where the joined dataset will be saved as a CSV file.
        - columns: A dictionary mapping the original column names in the dataset to the expected column names (e.g., {"case:concept:name": "case_id", "time:timestamp": "time:timestamp"}).
    """
    # Load dataset
    df = load_event_log_as_dataframe(dataset_path, columns_map=columns)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    # Get train set with split timestamp
    split_timestamp = pd.to_datetime(split_timestamp)
    df_train = df[df["time:timestamp"] <= split_timestamp].copy()
    # Load results
    results = pd.read_csv(results_path).rename(columns={"0": "case_id", "1": "remaining_time"})
    # Convert string remaining time into array
    results["remaining_time"] = results["remaining_time"].apply(lambda x: ast.literal_eval(x))
    # Flatten remaining time to get value
    results["remaining_time"] = results["remaining_time"].apply(lambda x: x[0][0])
    # Convert to numeric
    results["remaining_time"] = pd.to_numeric(results["remaining_time"], errors="coerce")

    last_train_ts = df_train.groupby("case:concept:name")["time:timestamp"].max()
    first_full_ts = df.groupby("case:concept:name")["time:timestamp"].min()

    dummy_df = pd.DataFrame(pd.NA, index=results.index, columns=df_train.columns)
    dummy_df["case:concept:name"] = results["case_id"].values
    dummy_df["concept:name"] = "DUMMY"
    # Difference is calculated in days for process transformer
    results["case_id"] = results["case_id"].astype(str)

    base_ts = results["case_id"].map(last_train_ts)
    base_ts = base_ts.fillna(results["case_id"].map(first_full_ts))

    dummy_df["time:timestamp"] = base_ts + pd.to_timedelta(results["remaining_time"], unit="D")

    df_train = pd.concat([df_train, dummy_df], ignore_index=True)

    df_train.to_csv(Path(store_path))

    






if __name__ == "__main__":
    store_to = "/Path/to/store/joined_dataset.csv"
    only_full_traces = False

    dataset_paths = {
        #"bpic2012": "",
        #"bpic2020_domestic_declarations": "",
        #"bpic2020_international_declarations": "",
        #"bpic2020_prepaid_travel": "",
        #"bpic2020_request_paymentt": "",
        #"bpic2020_travel_permit": "",
        #"road_traffic": "",
        #"helpdesk": "",
        "sepsis": "/Path/to/sepsis/dataset.xes"
    }
    split_timestamps = {
        #"bpic2012": [
        #    ["2012-02-09 00:00:00+00:00", None], ["2012-02-10 00:00:00+00:00", None], 
        #    ["2012-01-07 00:00:00+00:00", "2012-02-02 00:00:00+00:00"], ["2012-01-06 00:00:00+00:00", "2012-02-01 00:00:00+00:00"]],
        #"bpic2020_domestic_declarations": [
        #    ["2018-12-22 00:00:00+00:00", None], ["2018-12-20 00:00:00+00:00", None], 
        #    ["2018-06-25 00:00:00+00:00", "2018-11-06 00:00:00+00:00"], ["2018-06-25 00:00:00+00:00", "2018-11-07 00:00:00+00:00"]],
        #"bpic2020_international_declarations": [
        #    ["2019-08-20 00:00:00+00:00", None], ["2019-09-10 00:00:00+00:00", None], 
        #    ["2018-11-30 00:00:00+00:00", "2019-06-16 00:00:00+00:00"], ["2019-01-12 00:00:00+00:00", "2019-07-13 00:00:00+00:00"]],
        #"bpic2020_prepaid_travel": [
        #    ["2018-09-21 00:00:00+00:00", None], ["2018-09-19 00:00:00+00:00", None], 
        #    ["2018-04-21 00:00:00+00:00", "2018-08-14 00:00:00+00:00"], ["2018-04-18 00:00:00+00:00", "2018-08-12 00:00:00+00:00"]],
        #"bpic2020_request_paymentt": [
        #    ["2019-02-01 00:00:00+00:00", None], ["2019-01-31 00:00:00+00:00", None], 
        #    ["2018-07-27 00:00:00+00:00", "2018-12-16 00:00:00+00:00"]],
        #"bpic2020_travel_permit": [
        #    ["2020-09-07 00:00:00+00:00", None], ["2020-09-27 00:00:00+00:00", None], 
        #    ["2019-10-26 00:00:00+00:00", "2020-07-06 00:00:00+00:00"], ["2019-09-13 00:00:00+00:00", "2020-06-09 00:00:00+00:00"]],
        #"road_traffic": [
        #    ["2010-10-07 00:00:00+00:00", None], 
        #    ["2010-10-08 00:00:00+00:00", None],
        #    ["2008-01-27 00:00:00+00:00", "2010-02-04 00:00:00+00:00"], ["2008-01-28 00:00:00+00:00", "2010-02-05 00:00:00+00:00"]
        #],
        #"helpdesk": [
        #    ["2013-03-19 00:00:00+00:00", None], 
        #    ["2013-03-24 00:00:00+00:00", None],
        #    ["2012-06-01 00:00:00+00:00", "2013-01-06 00:00:00+00:00"], ["2012-06-12 00:00:00+00:00", "2013-01-12 00:00:00+00:00"]
        #],
        "sepsis": [
            ["2015-02-10 00:00:00+00:00", None], ["2015-02-09 00:00:00+00:00", None],
            ["2014-10-18 00:00:00+00:00", "2015-01-13 00:00:00+00:00"], ["2014-10-17 00:00:00+00:00", "2015-01-12 00:00:00+00:00"],
            ["2014-10-18 00:00:00+00:00", "2015-01-12 00:00:00+00:00"]
        ]
    }

    for d in dataset_paths.keys():
        for split in split_timestamps[d]:
            split_timestamp = split[0]
            last_timestamp = split[1]
            print(split_timestamp, last_timestamp)
            path = dataset_paths[d]
            name = (d + "_split_" + pd.to_datetime(split_timestamp).strftime("%Y%m%d") + "_last_" + ((pd.to_datetime(last_timestamp).strftime("%Y%m%d") + ".csv") if last_timestamp != None else "None.csv"))
            results_path = Path(PROCESS_TRANSFORMER_REPO) / "results" / name / "results_remaining_time_predictions.csv"
            store_path = Path(store_to) / name
            wrapper = ProcessTransformerWrapper(repo_dir=PROCESS_TRANSFORMER_REPO, split_timestamp=split_timestamp, last_timestamp=last_timestamp)
            columns = {
                "case:concept:name": "case:concept:name",
                "concept:name": "concept:name",
                "time:timestamp": "time:timestamp",
            }
            
            raw_csv = wrapper.prepare_dataset_csv(
                event_log_path=path, 
                dataset_name=name, 
                columns_map=columns, 
                input_timestamp_col="time:timestamp", 
                case_col="case:concept:name", 
                activity_col="concept:name", 
                timestamp_col="time:timestamp", 
                sort_by_timestamp_then_case=True, 
                index=False,)
            wrapper.run_preprocessing(dataset_name=name, task=Task.REMAINING_TIME, raw_log_file=raw_csv, only_full_traces=only_full_traces)
            wrapper.train_and_evaluate(dataset_name=name, task=Task.REMAINING_TIME, epochs=25, gpu=0)
            
            join_results_on_dataset(split_timestamp=split_timestamp, store_path=store_path, dataset_path=path, results_path=results_path, columns=columns)
            print("Stored dataset " + name)