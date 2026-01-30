import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pm4py
import pickle
import sys

XPATH = "/Volumes/Daniel/Thesis/resources"
DATASET = "BPI_2012/BPI_Challenge_2012.xes"

# Build path
path = os.path.join(XPATH, DATASET)

# Function to import XES and return log
def import_xes(p=path, from_file=False):
    path_pickle = os.path.join(XPATH, 'tmp/pickle/raw_data.pickle')
    if not os.path.exists(path_pickle):
        with open(path_pickle, 'x'): pass
    if from_file:
        with open(path_pickle, 'rb') as handle:
            data = pickle.load(handle)
        return data
    # Import XES
    df = pm4py.read_xes(path)
    # Two timestamp columns (one on trace, one on event level) - overwrite to event level to avoid misunderstandings
    # df.iloc[:, 4] = df.iloc[:, 3]
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True, errors="coerce")
    # Convert date to date format
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    # Extrapolate timestamps per trace to avoid events at the same time
    # Filter names from data
    names = df['case:concept:name']
    # Drop duplicate names
    # names.drop_duplicates(inplace=True)
    names = df["case:concept:name"].drop_duplicates()
    
    # Extrapolate time -- not necessary
    # Process each name separately
    """for n in names:
        print(n)
        # Get data
        name_data = df[df['case:concept:name'] == n]
        name_data.sort_values(by=['time:timestamp'])
        first = True
        prev_index = -1
        offset = 0
        for index, row in name_data.iterrows():
            df.at[index, 'time:timestamp'] = df.at[index, 'time:timestamp'] + pd.Timedelta(microseconds=offset)
            if not first:
                if df.at[index, 'time:timestamp'] == df.at[prev_index, 'time:timestamp']:
                    delta = df.at[index, 'time:timestamp'] + pd.Timedelta(microseconds=1)
                    df.at[index, 'time:timestamp'] = delta
                    offset += 1
            else:
                first = False
            prev_index = index"""


    print('Store to pickle')
    with open(path_pickle, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Successfully stored to pickle')
    
    return df


# Create DFG
def build_dfg(p=path):
    log = import_xes(p)
    d = {}
    directly_follows_graph = pm4py.discover_dfg(log)
    pm4py.view_dfg(directly_follows_graph[0], directly_follows_graph[1], directly_follows_graph[2])

def build_first_last_df(data: pd.DataFrame, from_file=False) -> dict:
    path = os.path.join(XPATH, 'tmp/pickle/build_first_last_df.pickle')
    if not os.path.exists(path):
        with open(path, 'x'): pass
    if from_file:
        with open(path, 'rb') as handle:
            cc = pickle.load(handle)
        return cc
    
    if data.empty:
        return None
    # Build total cases matrix
    # New Data Frame
    data_list = []
    # Get all traces by concept:name
    names = data['case:concept:name']
    # Drop Duplicates
    names = names.drop_duplicates()
    # Foreach name, search all data and get first and last timestamp
    for n in names:
        print("Search - Current name: " + n)
        # Get all data with same name
        n_data = data[data['case:concept:name'] == n]
        # Reduce on name and time
        n_data = n_data[['case:concept:name', 'time:timestamp']]
        # Sort values by time
        n_data = n_data.sort_values("time:timestamp")
        # Reset index
        n_data.reset_index(drop=True, inplace=True)
        # Get first and last value depending on length
        if len(n_data) == 0:
            continue
        elif len(n_data) == 1:
            c = {'name':n_data.loc[0, 'case:concept:name'], 'first':n_data.loc[0, 'time:timestamp'], 'last':n_data.loc[0, 'time:timestamp']}
        else:
            c = {'name':n_data.loc[0, 'case:concept:name'], 'first':n_data.loc[0, 'time:timestamp'], 'last':n_data.loc[len(n_data)-1, 'time:timestamp']}
        data_list.append(c)

    # Merge all dicts to DataFrame
    result_data = pd.DataFrame(data_list)
    return result_data

# Plot series
def plot_series(ts: pd.Series):
    ts.plot(figsize=(10, 5), title='Number of Items per Hour')
    plt.xlabel('Timestamp')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()


# ---------- Concurrent cases data ----------

# Build Concurrency Measurement
def build_total_concurrency_dict(data: pd.DataFrame, from_file=False) -> dict:
    path = os.path.join(XPATH, 'tmp/pickle/build_total_concurrency_data.pickle')
    if not os.path.exists(path):
        with open(path, 'x'): pass
    if from_file:
        with open(path, 'rb') as handle:
            cc = pickle.load(handle)
        return cc

    if data.empty:
        return None

    # Get data with first and last timestamp
    result_data = build_first_last_df(data, from_file=from_file)

    # Build concurrency data frame
    # Get all timestamps and concatenate
    timestamps = []
    #timestamps.extend(result_data['first'].to_list())
    timestamps.extend(data['time:timestamp'].to_list())
    # Eliminate Duplicates
    timestamps = list(set(timestamps))
    # Sort timestamps
    timestamps.sort()
    # Create corresponding value vector
    values = np.zeros(len(timestamps))
    # Create concurrency dictionary
    cc = dict(zip(timestamps, values))
    # Foreach name
    for n in result_data['name']:
        print('Concurrency - Current name: ' + n)
        # Get current row and timestamps
        row = result_data[result_data['name'] == n]
        first_ts = row.iloc[0]['first']
        last_ts = row.iloc[0]['last']
        # Foreach timestamp in the dictionary
        for ts in cc:
            # Check if it is between the first and last timestamp of the process
            if ts >= first_ts and ts < last_ts:
                # If yes, increase value of the timestamp by 1
                value = cc[ts]
                value += 1
                cc[ts] = value
    
    print('Store to pickle')
    with open(path, 'wb') as handle:
        pickle.dump(cc, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Successfully stored to pickle')


    return cc

# Build concurrency time series as Series
def build_concurrency_data(data: pd.DataFrame, from_f=False) -> pd.Series:
    # Get concurrency dictionary
    cc = build_total_concurrency_dict(data, from_file=from_f)
    # Build DataFrame
    data = pd.DataFrame(cc.items(), columns=['time:timestamp', 'concurrent_cases'])
    # Extend to frequent series
    data = data.set_index('time:timestamp')['concurrent_cases']
    
    
    # Define new frequent index 
    new_index = pd.date_range(start=data.index.min().floor('h'), end=data.index.max().ceil('h'), freq="h")
    # Reindex and forward-fill
    _data = data.reindex(new_index, method='ffill')
    # Substitute NaN values by 0
    _data.fillna(0, inplace=True)
    # Drop index
    _data.reset_index(drop=True, inplace=True)
    #_data.plot()
    #plt.show()

    return _data

# ---------- Resource utilization data ----------

def get_all_resources(data: pd.DataFrame) -> set:
    resources = set(data['org:resource'].unique())
    return resources

def build_resource_utilization_data(data: pd.DataFrame, from_file = False, agg_method="span", activate_smoothing=True) -> pd.Series:
    path = os.path.join(XPATH, 'tmp/pickle/build_resource_utilization_data.pickle')
    if not os.path.exists(path):
        with open(path, 'x'): pass
    if from_file:
        with open(path, 'rb') as handle:
            cc = pickle.load(handle)
        return cc
    # If no data
    if data.empty:
        return None
    
    # Get all distinct resources
    resources = get_all_resources(data)
    total_resources = len(resources)
    
    # Aggregation method
    if agg_method == "span":
        # Build time series in time spans of 1 hour
        #first_last = build_first_last_df(data, from_file)
        #start = first_last['first'].min()
        #end = first_last['last'].max()

        # Generate hourly timestamps
        #timestamps = pd.date_range(start=start, end=end, freq='1H')

        # Build dictionary with each timestamp initialized to 0
        #hour_dict = {ts: 0 for ts in timestamps}

        resource_dict = {}
        data = data.rename(columns={"org:resource": "org_resource", "time:timestamp": "time_timestamp"})
        for row in data.itertuples(index=False):
            # Get values of row
            resource = row.org_resource
            time = row.time_timestamp.floor('h') # Floor timestamp to lower hour for bucketing
            # Create set for timestamp if not already existing
            if time not in resource_dict:
                resource_dict[time] = set()
            # Add resource to set -- avoids duplicate resource counting
            resource_dict[time].add(resource)

        # Convert to series, take number / total resources
        ts = pd.Series({t: (len(r) / total_resources) for t, r in resource_dict.items()})
        
        # Fill Series
        if activate_smoothing:
            ts = ts.asfreq('1H').ffill()
        else:
            ts = ts.asfreq('1H', fill_value=0)

        # Ensure datetime index
        ts.index = pd.to_datetime(ts.index)
        # Return series
        return ts
                

# ---------- Throughput time data ----------

# Build dictionary with process end date -> throughput time
def build_throughput_time_data(data: pd.DataFrame, from_file = False, agg_method="row") -> pd.Series:
    path = os.path.join(XPATH, 'tmp/pickle/build_throughput_time_data.pickle')
    if not os.path.exists(path):
        with open(path, 'x'): pass
    if from_file:
        with open(path, 'rb') as handle:
            cc = pickle.load(handle)
        return cc
    # If no data
    if data.empty:
        return None
    # Get data
    result_data = build_first_last_df(data, from_file=from_file)

    # Compute throughput time
    result_data['throughput_time'] = result_data['last'] - result_data['first']
    result_data['throughput_time'] = result_data['throughput_time'].dt.total_seconds()

    # Sort by last and reset index
    result_data = result_data[['last', 'throughput_time']].sort_values(by='last')
    result_data = result_data.reset_index(drop=True)

    # Apply aggregation method
    if agg_method == 'row':
        result_data = aggregate_tt_row_number(result_data, 100)
    elif agg_method == 'span':
        result_data = aggregate_tt_timespan(result_data, '1D')
    elif agg_method == 'cont':
        result_data = aggregate_tt_continuous(result_data, 100)
    else:
        # Do not aggregate but just take last
        result_data = result_data.set_index('last')['throughput_time']

    # Define new frequent index 
    new_index = pd.date_range(start=result_data.index.min().floor('h'), end=result_data.index.max().ceil('h'), freq="h")
    # Reindex and forward-fill
    _data = result_data.reindex(new_index, method='ffill')
    # Substitute NaN values by 0
    _data.fillna(0, inplace=True)
    # Drop index
    _data.reset_index(drop=True, inplace=True)
    _data.plot()
    plt.show()

    return _data

def aggregate_tt_row_number(data: pd.DataFrame, group_size=10) -> pd.DataFrame:
    data['group'] = data.index // group_size
    data_agg = data.groupby('group').agg({
        'last': 'max',
        'throughput_time': 'mean'
    })
    data_agg = data_agg.set_index('last')['throughput_time']
    return data_agg

def aggregate_tt_timespan(data: pd.DataFrame, time_length='1D') -> pd.DataFrame:
    data = data.set_index('last')
    data_agg = data.resample(time_length).mean()
    return data_agg
"""
def aggregate_tt_continuous(data: pd.DataFrame, last_rows=10) -> pd.DataFrame:
    data['throughput_avg_10'] = data['throughput_time'].rolling(window=last_rows).mean()
    data_agg = data[['last', 'throughput_time']]
    data_agg = data_agg.rename(columns={'last':'last', 'throughput_avg_10':'throughput_time'})
    data_agg = data_agg.set_index('last')['throughput_time']
    return data_agg
"""

def aggregate_tt_continuous(data: pd.DataFrame, last_rows=10) -> pd.Series:
    data = data.copy()
    data["throughput_time"] = data["throughput_time"].rolling(window=last_rows).mean()
    return data.set_index("last")["throughput_time"]




if __name__=="__main__":
    data = import_xes(from_file=True)
    tt = build_resource_utilization_data(data, False, 'span', True)
    print(tt)
    plot_series(tt)
