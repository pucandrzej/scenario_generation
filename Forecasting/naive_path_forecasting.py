'''
File contains the code for GAMLSS forecast of continuous market prices on DE intraday market.
'''
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
try:
    os.chdir('Forecasting') # for debuger run
except:
    pass
from numba import njit
from datetime import datetime, timedelta
import argparse

import time
from multiprocessing import Pool, RawArray
import scipy
from scipy.stats import johnsonsu
from scipy.stats import wasserstein_distance_nd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics.pairwise import pairwise_kernels

from sklearn.cluster import HDBSCAN, KMeans

import sqlite3
import pickle
import tensorflow as tf
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain

from utils.add_step_dependent_variables import add_step_dependent_variables

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='benchmark', help='Model for forecasting')
parser.add_argument('--daterange_start', default='2020-01-01', help='Start of eval data')
parser.add_argument('--daterange_end', default='2020-12-31', help='End of eval data')
parser.add_argument('--lookback', default=364, help='Min. training window length')
parser.add_argument('--delivery_time', default=0, help='Index from 0 to 95 of the delivery quarter of the day')
parser.add_argument('--forecasting_horizons', default=[31], help='5min intervals from the forecast to the trade that we want to forecast price for. In practice from 185 to 30 minutes before the delivery.')
parser.add_argument('--trade_time', default=0*3 + 8*12 - 6, help='Last 5min interval before the delivery that we want to trade in.')
parser.add_argument('--calibration_window_len', default=182, help='For every date consider a historical results from a calibration window.')
parser.add_argument('--processes', default=1, help='No of processes')
parser.add_argument('--required_scenarios', default=None, help='Number of the most recent scenarios to be used for SVR path scenarios recalculation if scenarios_sampling_method is "latest". Cannot exceed the sample length.')

paths_no = 1000

args = parser.parse_args()
model = args.model
start = args.daterange_start
end = args.daterange_end
lookback = int(args.lookback)
forecasting_horizons = args.forecasting_horizons
trade_time = int(args.trade_time)
delivery_time = int(args.delivery_time)
calibration_window_len = int(args.calibration_window_len)
if args.required_scenarios is not None:
    required_scenarios = int(args.required_scenarios)
else:
    required_scenarios = args.required_scenarios

results_folname = 'BENCHMARK_RESULTS'

if not os.path.exists(os.path.join(results_folname)):
    os.mkdir(os.path.join(results_folname))
for forecasting_horizon in forecasting_horizons:
    if not os.path.exists(os.path.join(results_folname, f'{model}_{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_____{args.required_scenarios}____')):
        os.mkdir(os.path.join(results_folname, f'{model}_{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_____{args.required_scenarios}____'))

dates = pd.date_range(start, end)
dates_calibration = pd.date_range(pd.to_datetime(start) - timedelta(days=calibration_window_len), pd.to_datetime(start) - timedelta(days=1))

# a global dictionary storing the variables passed from the initializer.
var_dict = {}

def init_worker(Model_data, Model_data_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['Model_data'] = Model_data
    var_dict['Model_data_shape'] = Model_data_shape


def run_one_day(inp):

    idx = inp[0]
    date_fore = inp[1]
    forecasting_horizon = inp[2]
    model = inp[3]
    calibration_flag = inp[4]

    '''
    GATHERING THE REQUIRED DATA
    '''
    # to compare ourselves with Ziel results we want to 185-30 min before delivery periods, 31 5min intervals

    daily_data_window = np.swapaxes(np.frombuffer(var_dict['Model_data']).reshape(var_dict['Model_data_shape']), 1, 2)[:idx,:,:] # swapaxes needed after migration from np array to database

    Y = daily_data_window[:, delivery_time, -forecasting_horizon:] - daily_data_window[:, delivery_time, -forecasting_horizon-1:-1]
    trajectory_start_point = daily_data_window[-1, delivery_time, -forecasting_horizon - 1]
    '''
    SAMPLE TRAJECTORIES AND SAVE THEM
    '''
    Y_historical = Y[:-1, :] # select only the historical trajectories to sample from
    if required_scenarios is not None:
        Y_historical = Y_historical[-required_scenarios:, :]
    sampled_indices = np.random.choice(Y_historical.shape[0], size=paths_no, replace=True)
    sampled_Y = Y_historical[sampled_indices, :]
    sampled_cumsum = np.cumsum(sampled_Y, axis=1)
    simulation_results = (sampled_cumsum + trajectory_start_point).T

    results = pd.DataFrame(simulation_results, columns=[f"benchmark_prediction_{i}" for i in range(simulation_results.shape[1])])

    result_file_name = os.path.join(
                            f"{results_folname}",
                            f"{model}_{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_____{args.required_scenarios}____",
                            f"{calibration_flag}_{str((pd.to_datetime(date_fore) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=5*(trade_time - 1))).replace(':',';')}_{forecasting_horizon}___weights___window__.csv"
                        )

    results['actual'] = daily_data_window[-1, delivery_time, -forecasting_horizon:]
    results["naive"] = [trajectory_start_point] * forecasting_horizon

    try:
        results.to_csv(result_file_name)
    except:
        os.remove(result_file_name)
        raise KeyboardInterrupt("Interrupted on saving: last file removed to avoid empty files.")

if __name__ == '__main__':
    con = sqlite3.connect('quarterhourly_data_ID_5min_plus_indicators_paper_grade.db')
    sql_str = f"SELECT * FROM with_dummies WHERE Index_daily <= {trade_time} AND Time >= '2019-01-01 16:00:00' AND Day >= 62;" # load only the data required for simu, so up to last trade time in the trajectory
    daily_data = pd.read_sql(sql_str, con)[
        [str(i) for i in range(192)] + ["288"]
    ].to_numpy()
    daily_data = np.reshape(daily_data, (np.shape(daily_data)[0]//trade_time, trade_time, 193)) # making it a shape of [days, total steps in trajectory, variables no.]

    raw_arr = RawArray('d', np.shape(daily_data)[0] * np.shape(daily_data)[1] * np.shape(daily_data)[2])

    # Wrap X as an numpy array so we can easily manipulates its data in multiprocessing scheme
    daily_data_np = np.frombuffer(raw_arr).reshape(np.shape(daily_data))
    # Copy data to our shared array.
    np.copyto(daily_data_np, daily_data)
    data_shape = np.shape(daily_data)

    # free memory
    daily_data = None
    del daily_data

    simu_start = time.time()
    for forecasting_horizon in forecasting_horizons:
        if calibration_window_len > 0: # perform the calibration forecast
            inputlist_calibration = [[lookback - calibration_window_len + idx + 1, date, forecasting_horizon, model, 'calibration'] for idx, date in enumerate(dates_calibration)]
            try:
                with Pool(processes=int(args.processes), initializer=init_worker, initargs=(raw_arr, data_shape)) as p:
                    _ = p.map(run_one_day, inputlist_calibration)
            except Exception as exception:
                print(f"Failed pool due to {exception}. Restarting with 15 workers")
                with Pool(processes=15, initializer=init_worker, initargs=(raw_arr, data_shape)) as p:
                    _ = p.map(run_one_day, inputlist_calibration)

            inputlist = [[lookback + idx + 1, date, forecasting_horizon, model, 'test'] for idx, date in enumerate(dates)]
            try:
                with Pool(processes=int(args.processes), initializer=init_worker, initargs=(raw_arr, data_shape)) as p:
                    _ = p.map(run_one_day, inputlist)
            except Exception as exception:
                print(f"Failed pool due to {exception}. Restarting with 15 workers")
                with Pool(processes=15, initializer=init_worker, initargs=(raw_arr, data_shape)) as p:
                    _ = p.map(run_one_day, inputlist)

        else:
            inputlist = [[lookback + idx + 1, date, forecasting_horizon, model, 'test'] for idx, date in enumerate(dates)]
            with Pool(processes=int(args.processes), initializer=init_worker, initargs=(raw_arr, data_shape)) as p:
                _ = p.map(run_one_day, inputlist)
    simu_end = time.time()
    print(simu_end - simu_start, 'Total time of simulation:')
    with open(f"timing_results_model_{model}_d_{delivery_time}_t_{trade_time}.txt", "w") as file:
        file.write(f"Execution time: {simu_end - simu_start} seconds\n")
