# Time Series Analytics for Predictive Process Monitoring. An Approach for Predicting Process KPIs with Time Series Methods.

This repository contains the source code for my master thesis.
It provides the implementation for analyzing datasets, creating process-level KPI time series (for concurrent cases, resource utilization and throughput time), and for training, tuning, forecasting and evaluating on these time series.
The usage of the files as well as the integration of the external repositories are explained below.

## Dataset Statistics and Exploratory Data Analysis

For obtaining the statistics and performing the EDA, the `dataset_statistics.py` file is used. It contains multiple methods to analyze an event log. For executing, call 

```
python dataset_statistics.py -d "Path/to/your/dataset"
```
Per default, column names are set as defined in the XES standard. Deviating column names can be parsed with `-c_col`for case ID, `-r_col`for resource, `t_col`for time and `a_col`for activity.


## Data Processing Pipeline
In the `data_processing.py` file, all implementation on data preprocessing and creation of process KPI time series is included. To load a dataset and create its KPI time series, execute the following:
```
python data_processing.py -d "Path/to/your/dataset" -series "concurrent_cases"
```
Of course, other series types are also possible, just replace `concurrent_cases`with the type of your choice. The series creation can further be configured:
* `-series` series type: `concurrent_cases`, `resource_utilization` or `throughput_time``
* `-v`series variant: 
    * `sweepline`, `exact_changepoints`, `event_sampled`for Concurrent Cases KPI
    * `row`, `span`, `rolling`for Throughput Time KPI
* `-vp`variant param: only for Throughput Time KPI; defines further properties of variant
* `-c` optional cache directory
* `-c_col`, `-r_col`, `-t_col`, `-a_col` column names for case ID, resource, time and activity. Per default, the XES standard columns are taken.
* `-freq`frequency of the KPI time series
* `-utc` if listed, timestamps are converted to UTC
* `-smoothing`if listed, forward-fill is enabled for missing values


## Forecasting Pipeline

In `forecasting.py`, the pipeline and all presented models are implemented. They can be called via
```
python forecasting.py -d "Path/to/dataset" -r "Results/directory" -s "Series type" -tr truncation -m "Model1" "Model2" ...
```
The call can further be configured, for this please refer to the `--help`call and the source code. This calls the model with the preconfigured hyperparameter grid, performs hyperparameter tuning, and creates forecast results. The numerical forecasts are stored to the results directory for further evaluation.


## External Baselines

In this work, two external baselines are used:
* GenerativeLSTM (Camargo et al., 2019): <https://github.com/AdaptiveBProcess/GenerativeLSTM>
* ProcessTransformer (Bukhsh et al., 2021): <https://github.com/Zaharah/processtransformer>

Both works are also licensed under the Apache License, Version 2.0. To reproduce the baseline results, you have to download the respective repository. The files marked with * in the thesis are provided in folder `/compare/`, and have to replace the original files of the repository.

### GenerativeLSTM
To execute GenerativeLSTM with the constructed wrapper, two manuell starts are necessary.
1. Start the training step. Define a wrapper object and call `wrapper.call_training()`. This produces a `training.h5`file at your results directory and a corresponding folder in your downloaded repository under `/output_files`. For the next step, make sure that the training file has the filename `<<foldername>>_training.h5`.
2. Start the prediction process on the folder name. This will iteratively predict suffixes. At the end, the method `wrapper.convert_to_absolute_time`converts the predicted time deltas to real timestamps. The results are exported to a CSV file at your results directory.

### ProcessTransformer
For ProcessTransformer, no manual renaming of files is necessary. To execute, fill all attributes of the wrapper class, and execute the methods in the order shown in the wrapper class.

## Evaluation
The evaluation script `evaluation.py` provides a simple implementation to compute errors on the models against the test set and plot them together. The predicted results of the TSF models must all be located in the same folder. The results of the external baselines must be named the same way and be located in their respective directory. Then evaluation can be called like follows:
```
python evaluation.py -pred "Path/to/Predictions" -test "Path/to/test.csv" -glstm "Path/to/GenerativeLSTM/repository" -pt "Path/to/ProcessTransformer/repository" -file "filename.csv"
```
This prints the error results (MSE, MAE, RMSE) for each forecast and plots the forecasts together with the true test values.


## License and Author

This work is licensed under the Apache License, Version 2.0
&copy; 2026 Daniel Fuhge
