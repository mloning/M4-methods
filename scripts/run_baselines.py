#!/usr/bin/env python3 -u

import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from sklearn.neural_network import MLPRegressor

from copy import deepcopy
from sktime.forecasting.forecasters import DummyForecaster
from sktime.forecasting.forecasters import ExpSmoothingForecaster
from sktime.highlevel import Forecasting2TSRReductionStrategy
from sktime.highlevel import ForecastingStrategy
from sktime.highlevel import ForecastingTask


#  Set paths
home = os.path.expanduser("~")

repodir = os.path.join(home, "Documents/Research/python_methods/m4-methods/")
datadir = os.path.join(repodir, "Dataset")
traindir = os.path.join(datadir, 'Train')
testdir = os.path.join(datadir, 'Test')
savedir = os.path.join(repodir, "predictions")

assert os.path.exists(repodir)
assert os.path.exists(datadir)
assert os.path.exists(traindir)
assert os.path.exists(testdir)
assert os.path.exists(savedir)

#  Import meta data
info = pd.read_csv(os.path.join(datadir, 'M4-info.csv'))

# Get M4 baseline methods
m4_results = pd.read_excel(os.path.join(repodir, 'Evaluation and Ranks.xlsx'),
                           sheet_name='Point Forecasts-Frequency',
                           header=[0, 1]).dropna(axis=0)
strategies = m4_results.loc[:, ['Method', 'User ID']].iloc[:, 0]
baselines = [strategy for strategy in strategies if isinstance(strategy, str)]
print('Baseline strategies:', baselines)

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

strategies = []
strategies.extend([ForecastingStrategy(estimator, name=name)
                   for name, estimator in forecasters.items()])
strategies.extend([Forecasting2TSRReductionStrategy(estimator, name=name)
                   for name, estimator in regressors.items()])

selected_strategies = ('Naive',) # 'Naive2', 'SES', 'Holt', 'Damped')
print('Selected strategies:', selected_strategies)

strategies = [strategy for strategy in strategies
              if strategy.name in selected_strategies]

#  Dictionary of forecasting horizons
fhs = info.set_index('SP')['Horizon'].to_dict()

# Dictionary of frequencies
freqs = info.set_index('SP')['Frequency'].to_dict()

# Get dataset names
files = os.listdir(os.path.join(traindir))
datasets = [f.split('-')[0] for f in files]

#  Select weekly dataset
selected_datasets = ('Hourly', 'Daily', 'Weekly', 'Monthly', 'Yearly')

for dataset in selected_datasets:
    print(f"Dataset: {dataset}")

    # Get forecasting horizon
    fh = np.arange(fhs[dataset]) + 1
    print('Forecasting horizon: ', len(fh))

    #  Get seasonal frequency
    sp = freqs[dataset]
    print('Seasonal periodicity:', sp)

    # get all train and test datasets
    alltrain = pd.read_csv(os.path.join(traindir, f'{dataset}-train.csv'), index_col=0)
    alltest = pd.read_csv(os.path.join(testdir, f'{dataset}-test.csv'), index_col=0)

    # Get number of series in dataset
    n_series = alltrain.shape[0]
    assert n_series == info.SP.value_counts()[dataset]

    # Iterate over strategies
    for strategy in strategies:
        # allocate output array
        losses = np.zeros((n_series, 2))

        # Iterate over series in dataset
        for i in tqdm(range(n_series), desc=f"{strategy.name}", unit='series'):

            # Dataset id
            id = f"{dataset[0]}{i + 1}"
            fname = f"{strategy.name}_{dataset[0]}{id}_y_pred.txt"

            # if results file already exists, skip series
            if os.path.isfile(fname):
                continue

            # Get dataset
            y_train = alltrain.loc[id, :].dropna().reset_index(drop=True)
            y_test = alltest.loc[id, :].dropna().reset_index(drop=True)
            target_name = y_train.name

            # Check forecasting horizon
            assert len(fh) == len(y_test)

            #  get train data into expected format
            train = pd.DataFrame(pd.Series([y_train]), columns=[target_name])
            n_obs = len(train.iloc[0, 0])

            # adjust test index to be after train index
            y_test.index = y_test.index + n_obs
            assert y_test.index[0] == train.iloc[0, 0].index[-1] + 1

            # specify task
            task = ForecastingTask(target=target_name, fh=fh)

            # clone strategy
            s = deepcopy(strategy)

            # set data-specific params for seasonality if seasonal periodicity > 1
            if sp > 1:
                params = s.get_params()
                seasonal_params = ['seasonal', 'seasonal_periods']
                if all(param in params.keys() for param in seasonal_params):
                    s.set_params(**{'seasonal': 'multiplicative',
                                    'seasonal_periods': sp})

            # fit and predict
            s.fit(task, train)
            y_pred = s.predict()
            # assert y_pred.index.equals(y_test.index)

            # save predictions
            np.savetxt(os.path.join(savedir, fname), y_pred.values)
