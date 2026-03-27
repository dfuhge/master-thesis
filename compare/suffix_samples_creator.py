# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:03:26 2020

@author: Manuel Camargo
"""
import itertools

import pandas as pd
import numpy as np


class SuffixSamplesCreator():
    """
    This is the man class encharged of the model training
    """

    def __init__(self):
        self.log = pd.DataFrame
        self.ac_index = dict()
        self.rl_index = dict()
        self._samplers = dict()
        self._samp_dispatcher = {'basic': self._sample_suffix,
                                 'inter': self._sample_suffix_inter}

    def create_samples(self, params, log, ac_index, rl_index, add_cols):
        self.log = log
        self.ac_index = ac_index
        self.rl_index = rl_index
        columns = self.define_columns(add_cols, params['one_timestamp'])
        sampler = self._get_model_specific_sampler(params['model_type'])
        return sampler(columns, params)

    @staticmethod
    def define_columns(add_cols, one_timestamp):
        columns = ['ac_index', 'rl_index', 'dur_norm']
        add_cols = [x+'_norm' if x != 'weekday' else x for x in add_cols ]
        columns.extend(add_cols)
        if not one_timestamp:
            columns.extend(['wait_norm'])
        return columns
    # def define_columns(add_cols, one_timestamp):
    #     columns = ['ac_index', 'rl_index', 'dur_norm']
    #     add_cols = [x+'_norm' for x in add_cols]
    #     columns.extend(add_cols)
    #     if not one_timestamp:
    #         columns.extend(['wait_norm'])
    #     return columns

    def register_sampler(self, model_type, sampler):
        try:
            self._samplers[model_type] = self._samp_dispatcher[sampler]
        except KeyError:
            raise ValueError(sampler)

    def _get_model_specific_sampler(self, model_type):
        sampler = self._samplers.get(model_type)
        if not sampler:
            raise ValueError(model_type)
        return sampler
    
    def __sample_suffix(self, columns, parms): # D: original function - added "_" before name
        """
        Extraction of prefixes and expected suffixes from event log.
        Args:
            self.log (dataframe): testing dataframe in pandas format.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            pref_size (int): size of the prefixes to extract.
        Returns:
            list: list of prefixes and expected sufixes.
        """
        print(columns)
        times = ['dur_norm'] if parms['one_timestamp'] else ['dur_norm', 'wait_norm']
        equi = {'ac_index': 'activities', 'rl_index': 'roles'}
        vec = {'prefixes': dict(),
               'next_evt': dict()}
        x_times_dict = dict()
        y_times_dict = dict()
        self.log = self.reformat_events(columns, parms['one_timestamp'])
        # n-gram definition
        for i, _ in enumerate(self.log):
            for x in columns:
                serie, y_serie = list(), list()
                for idx in range(1, len(self.log[i][x])):
                    serie.append(self.log[i][x][:idx])
                    y_serie.append(self.log[i][x][idx:])
                if x in list(equi.keys()):
                    vec['prefixes'][equi[x]] = (
                        vec['prefixes'][equi[x]] + serie
                        if i > 0 else serie)
                    vec['next_evt'][equi[x]] = (
                        vec['next_evt'][equi[x]] + y_serie
                        if i > 0 else y_serie)
                elif x in times:
                    x_times_dict[x] = (
                        x_times_dict[x] + serie if i > 0 else serie)
                    y_times_dict[x] = (
                        y_times_dict[x] + y_serie if i > 0 else y_serie)
        vec['prefixes']['times'] = list()
        x_times_dict = pd.DataFrame(x_times_dict)
        for row in x_times_dict.values:
            new_row = [np.array(x) for x in row]
            new_row = np.dstack(new_row)
            new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))
            vec['prefixes']['times'].append(new_row)
        # Reshape intercase expected attributes (prefixes, # attributes)
        vec['next_evt']['times'] = list()
        y_times_dict = pd.DataFrame(y_times_dict)
        for row in y_times_dict.values:
            new_row = [np.array(x) for x in row]
            new_row = np.dstack(new_row)
            new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))
            vec['next_evt']['times'].append(new_row)
        return vec


    def _sample_suffix(self, columns, parms):
        """
        Build samples for suffix prediction. Uses original implementation extended by usage of minimal prefixes.
        For using with complete-trace variant, please ensure that the train_log.csv file is replaced by the file of the all-data variant
        to include the active traces for prediction.
        """
        times = ['dur_norm'] if parms['one_timestamp'] else ['dur_norm', 'wait_norm']
        equi = {'ac_index': 'activities', 'rl_index': 'roles'}

        # Build base log in the "trace-list-of-dicts" format used downstream
        self.log = self.reformat_events(columns, parms['one_timestamp'])

        # Get all trace ids from log
        cases = []
        for i, item in enumerate(self.log):
            cases.append(item['caseid'])


        

        # Append minimal prefixes from test set
        test_csv = "/Volumes/Daniel/Thesis/compare/GenerativeLSTM/test.csv"
        if test_csv is not None:
            df_test = pd.read_csv(test_csv, low_memory=False)


            # ensure timestamp sortable
            df_test["end_timestamp"] = pd.to_datetime(df_test["end_timestamp"], errors="coerce")

            # take first (earliest) event per case
            first_events = (
                df_test.sort_values("end_timestamp")
                .groupby("caseid", as_index=False)
                .first()
            )
            # Filter dataframe to only cases NOT in 'cases'
            first_events = first_events[~first_events["caseid"].isin(cases)]

            # helper: map label -> id using training indices
            ac_index = parms["index_ac"]  # label -> int
            rl_index = parms["index_rl"]  # label -> int

            # pick safe fallback ids if unknown labels appear
            def _fallback_id(index_map):
                for k in ("UNK", "unknown", "start", "end"):
                    if k in index_map:
                        return index_map[k]
                # last resort: 0
                return 0

            ac_fallback = _fallback_id(ac_index)
            rl_fallback = _fallback_id(rl_index)

            for _, row in first_events.iterrows():
                ac_label = row["task"]
                rl_label = row["user"]

                ac_id = ac_index.get(ac_label, ac_fallback)
                rl_id = rl_index.get(rl_label, rl_fallback)

                # Create a single-event trace in the same format as self.reformat_events output
                new_trace = {
                    "caseid": row["caseid"],
                    "ac_index": [ac_id],
                    "rl_index": [rl_id],
                }
                # Dummy values for input
                new_trace["dur_norm"] = [0.0]
                if not parms["one_timestamp"]:
                    new_trace["wait_norm"] = [0.0]

                self.log.append(new_trace)

        vec = {"prefixes": {}, "next_evt": {}}
        x_times_dict = {}
        y_times_dict = {}

        vec["prefixes"]["caseid"] = [] # D: Added for trace id
        vec["next_evt"]["caseid"] = [] #. D: Added for trace id

        for i, _ in enumerate(self.log):
            # D: Added for trace id
            caseid = self.log[i].get("caseid", i)  # fallback to i if missing
            # D: store caseid once per produced prefix (exactly one per trace)
            vec["prefixes"]["caseid"].append(caseid)
            vec["next_evt"]["caseid"].append(caseid)


            
            for x in columns:
                trace_seq = self.log[i][x]

                # If trace length >= 2, use longest prefix with a real "next event"
                if len(trace_seq) >= 2:
                    serie = [trace_seq[:-1]]
                    y_serie = [trace_seq[-1:]]
                    #serie = [trace_seq[-1:]]
                    #y_serie = [[]]
                else:
                    # length 1: keep that as prefix, but no expected suffix available
                    serie = [trace_seq[:]]
                    y_serie = [[]]

                if x in equi:
                    key = equi[x]  # 'activities' or 'roles'
                    vec["prefixes"][key] = vec["prefixes"].get(key, []) + serie
                    vec["next_evt"][key] = vec["next_evt"].get(key, []) + y_serie

                elif x in times:
                    x_times_dict[x] = x_times_dict.get(x, []) + serie
                    y_times_dict[x] = y_times_dict.get(x, []) + y_serie

        # --- 4) Convert times lists into the original (T, num_time_attrs) arrays ---
        vec["prefixes"]["times"] = []
        x_times_df = pd.DataFrame(x_times_dict) if x_times_dict else pd.DataFrame(columns=times)

        for row in x_times_df.values:
            # row contains lists/arrays like [dur_seq, wait_seq] (or only dur_seq)
            new_row = [np.array(x, dtype=float) for x in row]  # each is shape (T,)
            new_row = np.dstack(new_row)                      # shape (1, T, K)
            new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))  # (T, K)
            vec["prefixes"]["times"].append(new_row)

        vec["next_evt"]["times"] = []
        y_times_df = pd.DataFrame(y_times_dict) if y_times_dict else pd.DataFrame(columns=times)

        for row in y_times_df.values:
            new_row = [np.array(x, dtype=float) for x in row]
            # if y_serie was [[]], we want an empty (0, K) array rather than failing
            if len(new_row) > 0 and new_row[0].size == 0:
                # no expected suffix times
                K = len(times)
                vec["next_evt"]["times"].append(np.zeros((0, K), dtype=float))
            else:
                new_row = np.dstack(new_row)
                new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))
                vec["next_evt"]["times"].append(new_row)

        return vec


    def _sample_suffix_inter(self, columns, parms):
        self.log = self.reformat_events(columns, parms['one_timestamp'])
        spl = {'prefixes': dict(), 'suffixes': dict()}
        # n-gram definition
        equi = {'ac_index': 'activities',
                'rl_index': 'roles',
                'dur_norm': 'times'}
        x_inter_dict, y_inter_dict = dict(), dict()
        for i, _ in enumerate(self.log):
            for x in columns:
                serie, y_serie = list(), list()
                for idx in range(1, len(self.log[i][x])):
                    serie.append(self.log[i][x][:idx])
                    y_serie.append(self.log[i][x][idx:])
                if x in list(equi.keys()):
                    spl['prefixes'][equi[x]] = (
                        spl['prefixes'][equi[x]] + serie if i > 0 else serie)
                    spl['suffixes'][equi[x]] = (
                        spl['suffixes'][equi[x]] + y_serie if i > 0 else y_serie)
                else:
                    x_inter_dict[x] = (
                        x_inter_dict[x] + serie if i > 0 else serie)
                    y_inter_dict[x] = (
                        y_inter_dict[x] + y_serie if i > 0 else y_serie)
        # Reshape intercase attributes (prefixes, n-gram size, # attributes)
        spl['prefixes']['inter_attr'] = list()
        x_inter_dict = pd.DataFrame(x_inter_dict)
        for row in x_inter_dict.values:
            new_row = [np.array(x) for x in row]
            new_row = np.dstack(new_row)
            new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))
            spl['prefixes']['inter_attr'].append(new_row)
        # Reshape intercase expected attributes (prefixes, # attributes)
        spl['suffixes']['inter_attr'] = list()
        y_inter_dict = pd.DataFrame(y_inter_dict)
        for row in y_inter_dict.values:
            new_row = [np.array(x) for x in row]
            new_row = np.dstack(new_row)
            new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))
            spl['suffixes']['inter_attr'].append(new_row)
        return spl

# =============================================================================
# Reformat
# =============================================================================
    def reformat_events(self, columns, one_timestamp):
        """Creates series of activities, roles and relative times per trace.
        Args:
            log_df: dataframe.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
        Returns:
            list: lists of activities, roles and relative times.
        """
        temp_data = list()
        log_df = self.log.to_dict('records')
        key = 'end_timestamp' if one_timestamp else 'start_timestamp'
        log_df = sorted(log_df, key=lambda x: (x['caseid'], key))
        for key, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
            trace = list(group)
            temp_dict = dict()
            for x in columns:
                serie = [y[x] for y in trace]
                if x == 'ac_index':
                    serie.insert(0, self.ac_index[('start')])
                    serie.append(self.ac_index[('end')])
                elif x == 'rl_index':
                    serie.insert(0, self.rl_index[('start')])
                    serie.append(self.rl_index[('end')])
                else:
                    serie.insert(0, 0)
                    serie.append(0)
                temp_dict = {**{x: serie}, **temp_dict}
            #temp_dict = {**{'caseid': key}, **temp_dict}
            temp_dict = {**{'caseid': trace[0]['caseid']}, **temp_dict} # D: Changed to get actual trace id
            temp_data.append(temp_dict)
        return temp_data
