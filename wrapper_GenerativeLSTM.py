"""
Wrapper for executing GenerativeLSTM repository.
"""


from pathlib import Path
from typing import Tuple
import pandas as pd
import pm4py
import ast

import sys
# Repository path
GENERATIVE_LSTM_REPO = "/Users/dfuhge/Documents/Studium/Uni Mannheim/Master/Masterarbeit/Masterarbeit Wifo/Code/master-thesis/master-thesis/src/compare/GenerativeLSTM-master"
sys.path.append(str(GENERATIVE_LSTM_REPO))
from dg_training import main as train_main # type: ignore
from dg_predictiction import main as pred_main # type: ignore


def shift_list(lst):
    """
    Shifts list with values such that first value is non-negative

    - lst: list to shift

    Returns: shifted list
    """
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
    """
    Cumulates list values such that each index has the sum of all values until that index.

    - lst: list to cumulate

    Returns: list with cumulated values
    """
    if not lst:  # handle empty lists
        return lst
    if not isinstance(lst, list):
        lst = ast.literal_eval(lst)
    result = []
    total = 0
    
    for x in lst:
        if x != None:
            total += x
            result.append(total)
    
    return result

class GenerativeLSTMWrapper():
    """
    Wrapper class for executing GenerativeLSTM repository
    """
    def __init__(self, repo_dir: str, dataset_dir: str, result_dir: str, columns: dict=None, time_col:str = None, case_col: str = None, split_timestamp=None, last_timestamp=None, full_traces: bool = False):
        self.repo_dir = Path(repo_dir)
        self.dataset_dir = Path(dataset_dir)
        self.result_dir = Path(result_dir)
        self.columns = columns
        self.time_col = time_col
        self.case_col = case_col
        self.split_timestamp = split_timestamp
        self.last_timestamp = last_timestamp
        self.full_traces = full_traces
        # Automatically load specified event log
        self._load_event_log()
    
    def _load_event_log(self) -> pd.DataFrame:
        """
        Loads event log from file and stores it as pandas DataFrame in self.data. Supports .xes and .csv files. For .xes files, uses pm4py to read and convert to DataFrame. For .csv files, uses pandas read_csv. Also checks for specified time column and parses it as datetime if provided. Drops rows with missing or empty case ids. Renames columns according to self.columns mapping if provided.
        """
        dataset_suffix = self.dataset_dir.suffix.lower()
        if dataset_suffix == ".xes":
            import pm4py
            log = pm4py.read_xes(str(self.dataset_dir))
            df = pm4py.convert_to_dataframe(log)
        elif dataset_suffix == ".csv":
            df = pd.read_csv(self.dataset_dir)
        else:
            raise ValueError(f"Unsupported file type: {dataset_suffix}")
        # Check time column
        if self.time_col != None:
            if self.time_col not in df.columns:
                raise KeyError(f"column '{self.time_col}' not found in columns: {list(df.columns)}")
            else:
                # Parse datetimes
                df[self.time_col] = pd.to_datetime(df[self.time_col], errors="raise", utc=True)
        df = df.dropna(subset=[self.case_col])
        df = df[df[self.case_col] != ""]
        # Rename columns if necessary
        if self.columns != None:
            df = df.rename(columns=self.columns)
        # Set data as wrapper data
        self.data = df
        # Return data
        return df
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for training and prediction.
        """
        # Replace NaN values for user and task with UNK (otherwise ac_rl throws errors)
        df['task'] = df["task"].fillna("unk")
        df['user'] = df["user"].fillna("unk")
        # Create start_timestamp column for feature manager (although obsolet)
        df['start_timestamp'] = pd.to_datetime(df['end_timestamp'])
        return df

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply time-based split to data and save train and test data to files. If self.full_traces is True, only use full traces in train split (i.e., cases that have finished before split_timestamp). If self.full_traces is False, use all data before split_timestamp for training. For test split, use all data after split_timestamp (or between split_timestamp and last_timestamp if provided). Also saves full data before split_timestamp to file for reference. Returns train and test DataFrames.
        """
        # Work on local copy
        df = self.data.copy()
        if self.split_timestamp == None:
            raise ValueError("split_timestamp must be provided")
        # Preprocess data
        df = self._preprocess_data(df)
        # Split data
        if self.full_traces:
            # Only use full traces in train split
            # Get case ids of cases that have finished before split_timestamp
            full_cases = (df.groupby("caseid")["end_timestamp"].max().loc[lambda x: x <= self.split_timestamp].index)
            train_df = df[df["caseid"].isin(full_cases)].copy()
        else:
            # Use full data before split_timestamp
            train_df = df[df["end_timestamp"] <= self.split_timestamp].copy()
        full_df = df[df["end_timestamp"] <= self.split_timestamp].copy()
        # Build test data
        # Get all data after split_timestamp
        if last_timestamp == None:
            test_df = df[df["end_timestamp"] > split_timestamp].copy()
        else:
            test_df = df[(df["end_timestamp"] > split_timestamp) & (df["end_timestamp"] <= last_timestamp)].copy()
        self.train = train_df
        self.test = test_df
        # Generate Result directory if necessary
        self.result_dir.mkdir(parents=True, exist_ok=True)
        # Generate paths
        train_path = str(self.result_dir / "training.csv")
        test_path  = str(self.result_dir / "test.csv")
        full_path  = str(self.result_dir / "full.csv")
        # Save to file
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        full_df.to_csv(full_path, index=False)

        return train_df, test_df
    
    def call_training(self, model_family: str = "lstm", max_eval: str = "1", optimizer: str = "bayesian") -> None:
        """
        Calls training function of GenerativeLSTM repository with specified parameters.

            - model_family: model family to use (e.g., "lstm", "transformer")
            - max_eval: maximum number of evaluations for hyperparameter optimization (e.g., "1", "10", "100")
            - optimizer: optimization method for hyperparameter optimization (e.g., "bayesian", "random", "grid")
        """
        argv = ["-f", str(self.result_dir / "training.csv"), "-m", model_family, "-e", max_eval, "-o", optimizer]
        train_main(argv)
    
    def call_prediction(self, folder: str = '', model_file: str = 'Production.h5', variant: str = 'random_choice', rep: str = '1'):
        """
        Calls prediction function of GenerativeLSTM repository with specified parameters.

            - folder: name of output folder
            - model_file: name of model file
            - variant: prediction variant to use (e.g., 'random_choice', 'most_frequent', 'beam_search')
            - rep: repetition number (used for naming output file)
        """
        argv = ["-ho", True, "-a", "pred_sfx", "-c", folder, "-b", model_file, "-v", variant, "-r", rep]
        pred_main(argv)

    def convert_to_absolute_time(self, prediction_path: str, time_col: str, case_col: str, split: str):
        """
        Converts relative time predictions to absolute timestamps. Reads predictions from prediction_path, reads original log from self.dataset_dir, computes last timestamp before split_timestamp for each case, shifts and cumulates predicted time values, merges with last timestamp, explodes list columns to have one row per predicted event, filters out start/end activities, computes absolute timestamps by adding predicted time to last timestamp, and saves final results to file in self.result_dir with name based on split.

            - prediction_path: path to csv file with predictions (output of GenerativeLSTM prediction)
            - time_col: name of timestamp column in original log (e.g., "time:timestamp")
            - case_col: name of case id column in original log (e.g., "case:concept:name")
            - split: name of split (used for naming output file)
        """
        # Read predictions; parse list-like columns on load
        df = pd.read_csv(prediction_path, dtype={"caseid": "string"}, converters={
            "ac_prefix": ast.literal_eval,
            "ac_expect": ast.literal_eval,
            "ac_pred": ast.literal_eval,
            "rl_prefix": ast.literal_eval,
            "rl_expect": ast.literal_eval,
            "rl_pred": ast.literal_eval,
            "ac_prefix_label": ast.literal_eval,
            "ac_expect_label": ast.literal_eval,
            "ac_pred_label": ast.literal_eval,
            "rl_prefix_label": ast.literal_eval,
            "rl_expect_label": ast.literal_eval,
            "rl_pred_label": ast.literal_eval,
            "tm_prefix": ast.literal_eval,
            "tm_expect": ast.literal_eval,
            "tm_pred": ast.literal_eval,},)
        # Read original log 
        suffix = self.dataset_dir.suffix.lower()
        if suffix == ".xes":
            import pm4py  # local import on purpose
            log = pm4py.read_xes(str(self.dataset_dir))
            df_org = pm4py.convert_to_dataframe(log)
        elif suffix == ".csv":
            df_org = pd.read_csv(self.dataset_dir)
        else:
            raise ValueError(f"Unsupported file type: {suffix} (expected .xes or .csv)")
        
        df_org[time_col] = pd.to_datetime(df_org[time_col], errors="coerce", utc=True)
        # Compute last timestamp before split_timestamp per case
        #split_timestamp = pd.to_datetime(split.split("split_")[1][:8], format="%Y%m%d")
        #df_org = df_org[df_org[time_col] <= self.split_timestamp]
        # Last timestamp before split
        before = df_org[df_org[time_col] <= self.split_timestamp]
        last_before = (
            before.groupby(case_col)[time_col]
            .max()
            .rename("last_before")
        )

        # First timestamp after split
        after = df_org[df_org[time_col] > self.split_timestamp]
        first_after = (
            after.groupby(case_col)[time_col]
            .min()
            .rename("first_after")
        )
        # Combine with fallback logic
        last_ts = pd.concat([last_before, first_after], axis=1)

        # Prefer last_before, otherwise take first_after
        last_ts["last_ts"] = last_ts["last_before"].combine_first(last_ts["first_after"])

        # Final formatting
        last_ts = last_ts["last_ts"].reset_index()
        
        # shift values in prediction to avoid negative values
        df["tm_pred"] = df["tm_pred"].apply(shift_list)
        # Cumulate time values
        df["tm_pred"] = df["tm_pred"].apply(cumulative_list)
        # Merge first_ts into results
        df = df.merge(last_ts, left_on="caseid", right_on=case_col, how="left")
        # Keep only rows where list lengths match
        lens_ok = (df["ac_pred_label"].str.len().eq(df["rl_pred_label"].str.len()) & df["ac_pred_label"].str.len().eq(df["tm_pred"].str.len()))
        df = df.loc[lens_ok, ["caseid", "ac_pred_label", "rl_pred_label", "tm_pred", "last_ts"]].copy()
        # Explode list columns -> one row per predicted event
        df = df.explode(["ac_pred_label", "rl_pred_label", "tm_pred"], ignore_index=True)
        # Filter out start/end activities
        df = df[~df["ac_pred_label"].isin(["start", "end"])].copy()
        # Ensure tm_pred numeric, then compute absolute timestamp
        df["tm_pred"] = pd.to_numeric(df["tm_pred"], errors="coerce")
        df["tm_real"] = df["last_ts"] + pd.to_timedelta(df["tm_pred"], unit="s")
        # Drop invalid predictions
        #df = df.dropna(subset=["last_ts"])
        # Final output columns
        out = df.rename(columns={"ac_pred_label": "task", "rl_pred_label": "user", "tm_real": "end_timestamp"})[["caseid", "task", "user", "end_timestamp"]]
        t = self.train[["caseid", "task", "user", "end_timestamp"]].copy()
        # Add minimal prefixes
        train_cases = set(t["caseid"].dropna())
        test_cases = set(self.test["caseid"].dropna())
        missing_cases = test_cases - train_cases
        test_missing = test[test["caseid"].isin(missing_cases)]
        
        first_events = (
            test_missing.sort_values("end_timestamp")
            .groupby("caseid", as_index=False)
            .first()
        )
        t_augmented = pd.concat([t, first_events], ignore_index=True)
        full = pd.concat([t_augmented, out], axis=0)
        
        #full = out
        # Save to file
        full.to_csv((self.result_dir /  (split + ".csv")), index=False)
        return full
    
    



if __name__ == "__main__":
    # General config
    dataset_dir = "/Volumes/Daniel/Thesis/resources/BPI_2012/BPI_Challenge_2012.xes"
    split_timestamp = "2012-02-10 00:00:00+00:00"
    last_timestamp = None#"2018-11-07 00:00:00+00:00"
    results_dir = "/Volumes/Daniel/Thesis/compare/GenerativeLSTM"
    # Map dataset columns on [case_id, task, event_type, user, end_timestamp] as required by repository
    columns_normal = {"case:concept:name": "caseid", "concept:name": "task", "lifecycle:transition": "event_type", "org:resource": "user", "time:timestamp": "end_timestamp"}
    columns_sepsis = {"case:concept:name": "caseid", "concept:name": "task", "lifecycle:transition": "event_type", "org:group": "user", "time:timestamp": "end_timestamp"}
    columns_helpdesk = {"Case ID": "caseid", "Activity": "task", "Variant": "event_type", "Resource": "user", "Complete Timestamp": "end_timestamp"}
    wrapper = GenerativeLSTMWrapper(repo_dir=GENERATIVE_LSTM_REPO, dataset_dir=dataset_dir, result_dir=results_dir, columns=columns_normal, time_col="time:timestamp", case_col="case:concept:name", split_timestamp=split_timestamp, last_timestamp=last_timestamp, full_traces=False)

    # Step 1: Train model
    train, test = wrapper.split_data()
    wrapper.call_training()

    # Step 2: Predict
    # split: name of output folder and model (have to be named same, model with _training.h5)
    split = "bpic2012_split_20120210_last_None_all_data"
    folder_dir = GENERATIVE_LSTM_REPO + "/output_files/" + split
    model_dir = results_dir + "/" + split + "_training.h5"
    wrapper.call_prediction(folder=folder_dir, model_file=model_dir)
    # final_result: csv-file where final prediction results are stored
    final_result = GENERATIVE_LSTM_REPO + "/output_files/" + split + "/results/gen_" + split + "_training_1.csv"
    wrapper.convert_to_absolute_time(final_result, time_col="time:timestamp", case_col="case:concept:name", split=split)

    #datasets = {
    #    "bpic2012": "/Volumes/Daniel/Thesis/resources/BPI_2012/BPI_Challenge_2012.xes",
    #    "bpic2020_domestic_declarations": "/Volumes/Daniel/Thesis/resources/BPI_2020/DomesticDeclarations/DomesticDeclarations.xes",
    #    "bpic2020_international_declarations": "/Volumes/Daniel/Thesis/resources/BPI_2020/InternationalDeclarations/InternationalDeclarations.xes",
    #    "bpic2020_prepaid_travel": "/Volumes/Daniel/Thesis/resources/BPI_2020/PrepaidTravel/PrepaidTravelCost.xes",
    #    "bpic2020_request_payment": "/Volumes/Daniel/Thesis/resources/BPI_2020/RequestPayment/RequestForPayment.xes",
    #    "bpic2020_travel_permit": "/Volumes/Daniel/Thesis/resources/BPI_2020/TravelPermit/PermitLog.xes",
    #    "road_traffic": "/Volumes/Daniel/Thesis/resources/Road_Traffic/Road_Traffic_Fine_Management_Process.xes",
    #    "helpdesk": "/Volumes/Daniel/Thesis/resources/Helpdesk/finale.csv",
    #    "sepsis": "/Volumes/Daniel/Thesis/resources/Sepsis/Sepsis_Event_Log.xes"
    #}

    #splits = [
    #    "bpic2012_split_20120106_last_20120201", "bpic2012_split_20120107_last_20120202", "bpic2012_split_20120209_last_None", 
    #    "bpic2012_split_20120210_last_None",
    #    "bpic2020_domestic_declarations_split_20180624_last_20181106", 
    #    "bpic2020_domestic_declarations_split_20180625_last_20181106",
    #    "bpic2020_domestic_declarations_split_20180625_last_20181107", "bpic2020_international_declarations_split_20181130_last_20190616",
    #    "bpic2020_international_declarations_split_20190112_last_20190713", "bpic2020_international_declarations_split_20190820_last_None",
    #    "bpic2020_international_declarations_split_20190910_last_None", "bpic2020_prepaid_travel_split_20180417_last_20180811",
    #    "bpic2020_prepaid_travel_split_20180418_last_20180812", "bpic2020_prepaid_travel_split_20180421_last_20180814", "bpic2020_prepaid_travel_split_20180918_last_None",
    #    "bpic2020_prepaid_travel_split_20180919_last_None", "bpic2020_prepaid_travel_split_20180921_last_None", "bpic2020_request_payment_split_20180726_last_20181215",
    #    "bpic2020_request_payment_split_20180727_last_20181216", "bpic2020_request_payment_split_20190131_last_None", "bpic2020_request_payment_split_20190201_last_None",
    #    "bpic2020_travel_permit_split_20190913_last_20200609", "bpic2020_travel_permit_split_20191026_last_20200706", "bpic2020_travel_permit_split_20200907_last_None",
    #    "bpic2020_travel_permit_split_20200927_last_None", 
    #    "helpdesk_split_20120531_last_20130105", "helpdesk_split_20120601_last_20130106", "helpdesk_split_20120612_last_20130112",
    #    "helpdesk_split_20130318_last_None", "helpdesk_split_20130319_last_None", "helpdesk_split_20130324_last_None", "sepsis_split_20141017_last_20150112", "sepsis_split_20141018_last_20150112",
    #    "sepsis_split_20141018_last_20150113", "sepsis_split_20150209_last_None", "sepsis_split_20150210_last_None",
    #    "bpic2020_domestic_declarations_split_20181220_last_None", "bpic2020_domestic_declarations_split_20181221_last_None", "bpic2020_domestic_declarations_split_20181222_last_None"
    #]
    #for s in splits:
    #    data = s.split("_split_")[0]
    #    d = datasets[data]
    #    print(s)
    #    s_str = s.split("split_")[1].split("_last")[0][:8]
    #    split_timestamp = pd.to_datetime(s_str, format="%Y%m%d", utc=True)
    #    l_str = s.split("last_")[1][:8]
    #    if l_str == "None":
    #        last_timestamp = None
    #    else:
    #        last_timestamp = pd.to_datetime(l_str, format="%Y%m%d", utc=True)
    #    if data == "sepsis":
    #        columns = columns_sepsis
    #        t_c = "time:timestamp"
    #        c_c = "case:concept:name"
    #    elif data == "helpdesk":
    #        columns = columns_helpdesk
    #        t_c = "Complete Timestamp"
    #        c_c = "Case ID"
    #    else:
    #        columns = columns_normal
    ##        t_c = "time:timestamp"
    #        c_c = "case:concept:name"
    #    print(d, split_timestamp, last_timestamp, t_c, c_c)
    #    wrapper = GenerativeLSTMWrapper(repo_dir=GENERATIVE_LSTM_REPO, dataset_dir=d, result_dir=results_dir, columns=columns, time_col=t_c, case_col=c_c, split_timestamp=split_timestamp, last_timestamp=last_timestamp, full_traces=True)
    #    train, test = wrapper.split_data()
    #    split = s + "_full_traces"
    #    folder_dir = GENERATIVE_LSTM_REPO + "/output_files/" + split
    #    model_dir = results_dir + "/full_traces/models/" + split + "_training.h5"
    #    try:
    #        print()
    #        wrapper.call_prediction(folder=folder_dir, model_file=model_dir)
    #    except:
    #        print()
        # final_result: csv-file where final prediction results are stored
    #    final_result = GENERATIVE_LSTM_REPO + "/output_files/" + split + "/results/gen_" + split + "_training_1.csv"
   #     wrapper.convert_to_absolute_time(final_result, time_col=t_c, case_col=c_c, split=s)