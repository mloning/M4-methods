#!/usr/bin/env python3 -u

import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from sklearn.utils.validation import check_consistent_length
from sklearn.neural_network import MLPRegressor

from copy import deepcopy
from sktime.forecasting.forecasters import DummyForecaster
from sktime.forecasting.forecasters import ExpSmoothingForecaster
from sktime.highlevel import Forecasting2TSRReductionStrategy
from sktime.highlevel import ForecastingStrategy
from sktime.highlevel import ForecastingTask


#  Define metrics
#  check mase simplification
for _ in range(100):
    a = np.random.normal(size=50)
    y_pred_naive = []
    sp = np.random.randint(2, 24)
    for i in range(sp, len(a)):
        y_pred_naive.append(a[(i - sp)])
    b = a[:-sp]
    assert np.array_equal(b, np.asarray(y_pred_naive))


def mase_loss(y_true, y_pred, y_train, sp=1):
    """
    Mean Absolute Scaled Error
    insample: insample data
    y_true: out of sample target values
    y_pred: predicted values
    sp: data frequency
    """
    #     y_pred_naive = []
    #     for i in range(sp, len(insample)):
    #         y_pred_naive.append(insample[(i - sp)])
    y_train = np.asarray(y_train)

    check_consistent_length(y_true, y_pred)
    check_consistent_length(y_true, y_train)

    #  naive seasonal prediction
    y_pred_naive = y_train[:-sp]
    mae_naive = np.mean(np.abs(y_train[sp:] - y_pred_naive))
    return np.mean(np.abs(y_true - y_pred)) / mae_naive


def smape_loss(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    """
    check_consistent_length(y_true, y_pred)
    k = len(y_true)
    error = y_true - y_pred
    nominator = np.abs(error)
    denominator = np.abs(y_true) + np.abs(y_pred)
    return np.mean(nominator / denominator) * 200


# def smape(a, b):
#     """
#     Calculates sMAPE
#     :param a: actual values
#     :param b: predicted values
#     :return: sMAPE
#     """
#     a = np.reshape(a, (-1,))
#     b = np.reshape(b, (-1,))
#     return np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()

#  Set paths
repodir = "/home/ucfamml/Documents/Research/python_methods/m4-methods/"
datadir = os.path.join(repodir, "Dataset")
traindir = os.path.join(datadir, 'Train')
testdir = os.path.join(datadir, 'Test')
savedir = os.path.join(repodir, "predictions")

assert os.path.exists(traindir)
assert os.path.exists(testdir)
assert os.path.exists(savedir)

#  Import meta data
info = pd.read_csv(os.path.join(datadir, 'M4-info.csv'))

# Load results from M4 competition for comparison
m4_results = pd.read_excel(os.path.join(repodir, 'Evaluation and Ranks.xlsx'),
                           sheet_name='Point Forecasts-Frequency',
                           header=[0, 1]).dropna(axis=0)

mase = m4_results.loc[:, ['Method', 'MASE']]
mase.columns = mase.columns.droplevel()
mase = mase.set_index('User ID')

smape = m4_results.loc[:, ['Method', 'sMAPE']]
smape.columns = smape.columns.droplevel()
smape = smape.set_index('User ID')

methods = m4_results.loc[:, ['Method', 'User ID']].iloc[:, 0]
baselines = [method for method in methods if isinstance(method, str)]
print('Baseline methods:', baselines)

#  Dictionary of forecasting horizons
fhs = info.set_index('SP').Horizon.to_dict()

# Dictionary of frequencies
freqs = info.set_index('SP').Frequency.to_dict()

# Get dataset names
files = os.listdir(os.path.join(traindir))
keys = [f.split('-')[0] for f in files]

#  Select weekly dataset
key = 'Weekly'
assert key in keys
print('Dataset:', key)

#  Get seasonal frequency
sp = freqs[key]
print('Seasonal periodicity:', sp)

# Define baseline forecasters
forecasters = {
    'Naive': DummyForecaster(strategy='last'),  # without seasonality adjustments
    #     'sNaive': None,
    'Naive2': ExpSmoothingForecaster(smoothing_level=1),  # with seasonality adjustments
    'SES': ExpSmoothingForecaster(),
    'Holt': ExpSmoothingForecaster(trend='add', damped=False),
    'Damped': ExpSmoothingForecaster(trend='add', damped=True),
    #     'Com': None,
    #     'Theta': None,
}

# Define baseline regressors
regressors = {
    'MLP': MLPRegressor(hidden_layer_sizes=6, activation='identity', solver='adam',
                        max_iter=100, learning_rate='adaptive', learning_rate_init=0.001),
    #     'RNN': None
}

baselines = []
baselines.extend([ForecastingStrategy(estimator, name=name)
                  for name, estimator in forecasters.items()])
baselines.extend([Forecasting2TSRReductionStrategy(estimator, name=name)
                  for name, estimator in regressors.items()])

n_baselines = len(baselines)

selected_strategies = ['Naive', 'Naive2', 'SES'] #, 'Holt' ,'Damped']
print('Selected methods:', selected_strategies)

strategies = [baseline for baseline in baselines
              if baseline.name in selected_strategies]

# get all train and test datasets
alltrain = pd.read_csv(os.path.join(traindir, f'{key}-train.csv'), index_col=0)
alltest = pd.read_csv(os.path.join(testdir, f'{key}-test.csv'), index_col=0)

# iterate over datasets
n_datasets = alltrain.shape[0]

for strategy in baselines:
    # allocate output array
    losses = np.zeros((n_datasets, 2))

    for i in tqdm(range(n_datasets), desc=f"{strategy.name}", unit='datasets'):

        # get individual series
        y_train = alltrain.iloc[i, :].dropna().reset_index(drop=True)
        y_test = alltest.iloc[i, :].dropna().reset_index(drop=True)
        name = y_train.name

        # specify forecasting horizon
        fh = np.arange(fhs[key]) + 1
        assert len(fh) == len(y_test)

        #  get train data into expected format
        train = pd.DataFrame(pd.Series([y_train]), columns=[name])
        n_obs = len(train.iloc[0, 0])

        # adjust test index to be after train index
        y_test.index = y_test.index + n_obs
        assert y_test.index[0] == train.iloc[0, 0].index[-1] + 1

        #  specify task
        task = ForecastingTask(target=name, fh=fh)

        # clone strategy
        s = deepcopy(strategy)

        # set data-specific params for seasonality if seasonal periodicity > 1
        if sp > 1:
            params = s.get_params()
            seasonal_params = ['seasonal', 'seasonal_periods']
            if all(param in params.keys() for param in seasonal_params):
                s.set_params(**{'seasonal': 'multiplicative',
                                'seasonal_periods': sp})
        #  fit and predict
        s.fit(task, train)
        y_pred = s.predict()
        assert y_pred.index.equals(y_test.index)

        #  save predictions
        fname = f"{strategy.name}_{key[0]}{i}_y_pred.txt"
        np.savetxt(os.path.join(savedir, fname), y_pred)
