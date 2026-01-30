from pathlib import Path
from typing import Tuple
import pandas as pd

import sys
GENERATIVE_LSTM_REPO = "/Users/dfuhge/Documents/Studium/Uni Mannheim/Master/Masterarbeit/Masterarbeit Wifo/Code/master-thesis/master-thesis/src/compare/GenerativeLSTM-master"
sys.path.append(str(GENERATIVE_LSTM_REPO))
from dg_training import main as train_main
from dg_predictiction import main as pred_main

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import pm4py
import ast


def load_and_time_split_event_log(path: str, out_dir: str, c={}, timestamp_col: str = "time:timestamp", split_timestamp=None, csv_timeformat: Optional[str] = None, index: bool = False) -> Tuple[str, str]:
    """
    Load an event log (XES or CSV), split it into train/test based on a split
    timestamp, and export both splits as CSV files.

    Split rule:
      - train: events with timestamp <= split_timestamp
      - test : events with timestamp >  split_timestamp

    Parameters
    ----------
    path : str
        Path to .xes or .csv file
    out_dir : str
        Directory where train/test CSVs will be saved (created if missing)
    timestamp_col : str
        Name of the timestamp column
    split_timestamp : datetime-like
        Timestamp used for splitting
    csv_timeformat : str, optional
        Datetime format string for CSV timestamps (only used for parsing)
    train_filename : str
        Output filename for the training split CSV
    test_filename : str
        Output filename for the test split CSV
    index : bool
        Whether to write the pandas index to CSV

    Returns
    -------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    """

    path = Path(path)
    suffix = path.suffix.lower()
    
    # ---------- load ----------
    if suffix == ".xes":
        import pm4py
        log = pm4py.read_xes(str(path))
        df = pm4py.convert_to_dataframe(log)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    
    # ---------- Renaming -----------
    df = df.rename(columns=c)
    
    # ---------- timestamp parsing ----------
    if timestamp_col not in df.columns:
        raise KeyError(f"timestamp_col '{timestamp_col}' not found in columns: {list(df.columns)}")

    if csv_timeformat is not None:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=csv_timeformat, errors="raise")
    else:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="raise")

    if split_timestamp is None:
        raise ValueError("split_timestamp must be provided")
    split_timestamp = pd.to_datetime(split_timestamp)

    # Replace NaN values for user and task with UNK (otherwise ac_rl throws errors)
    df['task'] = df["task"].fillna("unk")
    df['user'] = df["user"].fillna("unk")
    # Create start_timestamp column for feature manager (although obsolet)
    df['start_timestamp'] = pd.to_datetime(df['end_timestamp'])

    # ---------- split ----------
    train_df = df[df[timestamp_col] <= split_timestamp].copy()
    test_df  = df[df[timestamp_col] >  split_timestamp].copy()

    # ---------- export ----------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = str(out_dir / "training.csv")
    test_path  = str(out_dir / "test.csv")

    train_df.to_csv(train_path, index=index)
    test_df.to_csv(test_path, index=index)

    return train_path, test_path


def _call_dg_training(path: str, model_family: str = "lstm", max_eval: str = "1", optimizer: str = "bayesian"):
    argv = ["-f", path, "-m", model_family, "-e", max_eval, "-o", optimizer]
    train_main(argv)

def get_newest_folder(base_dir: str) -> Optional[str]:
    """
    Return the name of the newest folder inside base_dir.
    If no folders exist, return None.
    """
    base_path = Path(base_dir)

    folders = [p for p in base_path.iterdir() if p.is_dir()]
    if not folders:
        return None

    newest = max(folders, key=lambda p: p.stat().st_mtime)
    return newest.name

def _call_dg_prediction(one_timestamp: bool = True, activity: str = 'pred_sfx', folder: str = '', model_file: str = 'Production.h5', variant: str = 'Random Choice', rep: str = '1'):
    argv = ["-ho", one_timestamp, "-a", activity, "-c", folder, "-b", model_file, "-v", variant, "-r", rep]
    #try:
    pred_main(argv)
    #except TypeError:
    #    print("TypeError - ignored as evaluation not relevant")

def shift_list(lst):
    if not lst:  # safety check for empty lists
        return lst
    if not isinstance(lst, list):
        lst = ast.literal_eval(lst)
    first = int(lst[0])
    
    # Only shift if first value is negative
    if first < 0:
        shift = -first
        return [x + shift for x in lst]
    else:
        return lst
    
def cumulative_list(lst):
    if not lst:  # handle empty lists
        return lst
    if not isinstance(lst, list):
        lst = ast.literal_eval(lst)
    result = []
    total = 0
    
    for x in lst:
        total += x
        result.append(total)
    
    return result

def _calculate_absolute_time(result_path: str, dataset_path: str, out_path: str) -> pd.DataFrame:
    # Read predictions; parse list-like columns on load (faster & cleaner)
    df = pd.read_csv(
        result_path,
        dtype={"caseid": "string"},
        converters={
            "ac_pred_label": ast.literal_eval,
            "rl_pred_label": ast.literal_eval,
            "tm_pred": ast.literal_eval,
        },
    )

    # Read original log and compute first timestamp per case
    df_org = pm4py.read_xes(str(dataset_path))
    df_org["time:timestamp"] = pd.to_datetime(df_org["time:timestamp"], errors="coerce")

    first_ts = (
        df_org.groupby("case:concept:name")["time:timestamp"]
             .min()
             .rename("first_ts")
             .reset_index()
    )
    first_ts["case:concept:name"] = first_ts["case:concept:name"].astype("string")


    # shift values in prediction to avoid negative values
    df["tm_pred"] = df["tm_pred"].apply(shift_list)
    # Cumulate time values
    df["tm_pred"] = df["tm_pred"].apply(cumulative_list)
    
    # Merge first_ts into results
    df = df.merge(first_ts, left_on="caseid", right_on="case:concept:name", how="left")

    # Keep only rows where list lengths match (fast boolean mask)
    lens_ok = (
        df["ac_pred_label"].str.len().eq(df["rl_pred_label"].str.len())
        & df["ac_pred_label"].str.len().eq(df["tm_pred"].str.len())
    )
    df = df.loc[lens_ok, ["caseid", "ac_pred_label", "rl_pred_label", "tm_pred", "first_ts"]].copy()

    # Explode list columns -> one row per predicted event
    df = df.explode(["ac_pred_label", "rl_pred_label", "tm_pred"], ignore_index=True)

    # Filter out start/end activities
    df = df[~df["ac_pred_label"].isin(["start", "end"])].copy()

    # Ensure tm_pred numeric, then compute absolute timestamp
    df["tm_pred"] = pd.to_numeric(df["tm_pred"], errors="coerce")
    df["tm_real"] = df["first_ts"] + pd.to_timedelta(df["tm_pred"], unit="s")

    # Final output columns
    out = df.rename(
        columns={
            "ac_pred_label": "activity",
            "rl_pred_label": "resource",
        }
    )[["caseid", "activity", "resource", "tm_real"]]

    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    dataset_path = "/Volumes/Daniel/Thesis/resources/BPI_2012/BPI_Challenge_2012.xes"
    split_timestamp = "2012-02-10 15:00:00+00:00"
    export_path = "/Volumes/Daniel/Thesis/compare/GenerativeLSTM"
    columns = {"case:concept:name": "caseid", "concept:name": "task", "lifecycle:transition": "event_type", "org:resource": "user", "time:timestamp": "end_timestamp"}
    train_path, test_path = load_and_time_split_event_log(dataset_path, export_path, c=columns, split_timestamp=split_timestamp, timestamp_col="end_timestamp")
    _call_dg_training(str(train_path))
    folder_path = "/Users/dfuhge/Documents/Studium/Uni Mannheim/Master/Masterarbeit/Masterarbeit Wifo/Code/master-thesis/master-thesis/src/compare/GenerativeLSTM-master/output_files/20260128_F6024322_477F_4F21_9065_EA718099E2DC"
    model_path = "/Volumes/Daniel/Thesis/compare/GenerativeLSTM/training.h5"
    #_call_dg_prediction(one_timestamp=True, activity='pred_sfx', folder=folder_path, model_file=model_path, variant="random_choice", rep="1")
    #df = pd.read_csv("/Users/dfuhge/Documents/Studium/Uni Mannheim/Master/Masterarbeit/Masterarbeit Wifo/Code/master-thesis/master-thesis/src/compare/GenerativeLSTM-master/output_files/20260128_F6024322_477F_4F21_9065_EA718099E2DC/results/gen_training_1.csv")
    #print(df['tm_pred'].iloc[3])
    #df["tm_pred"] = df["tm_pred"].apply(shift_list)
    #df["tm_pred"] = df["tm_pred"].apply(cumulative_list)
    #print(df['tm_pred'].iloc[3])
    result_path = "/Users/dfuhge/Documents/Studium/Uni Mannheim/Master/Masterarbeit/Masterarbeit Wifo/Code/master-thesis/master-thesis/src/compare/GenerativeLSTM-master/output_files/20260128_F6024322_477F_4F21_9065_EA718099E2DC/results/gen_training_1.csv"
    out_path = "/Users/dfuhge/Documents/Studium/Uni Mannheim/Master/Masterarbeit/Masterarbeit Wifo/Code/master-thesis/master-thesis/src/compare/GenerativeLSTM-master/output_files/20260128_F6024322_477F_4F21_9065_EA718099E2DC/results/gen_training_1_absolute_time.csv"
    
    #_calculate_absolute_time(result_path, dataset_path, out_path)