#!/usr/bin/env python3 -u

# preliminaries
import numpy as np
from joblib import Parallel
from joblib import delayed
import pandas as pd
import os
from sklearn.neural_network import MLPRegressor

from copy import deepcopy
from sktime.forecasters import DummyForecaster
from sktime.forecasters import ExpSmoothingForecaster
from sktime.highlevel import Forecasting2TSRReductionStrategy
from sktime.highlevel import ForecastingStrategy
from sktime.highlevel import ForecastingTask


# define function to run in parallel
def run_on_series(strategy, y_train, y_test, fh, sp):

    # remove missing values from padding
    y_train = y_train[~np.isnan(y_train)]
    y_test = y_test[~np.isnan(y_test)]

    # check forecasting horizon
    assert len(fh) == len(y_test)

    # get train data into expected format
    train = pd.DataFrame(pd.Series([pd.Series(y_train)]), columns=['target'])
    n_obs = len(train.iloc[0, 0])

    # adjust test index to be after train index
    y_test = pd.Series(y_test)
    y_test.index = y_test.index + n_obs
    assert y_test.index[0] == train.iloc[0, 0].index[-1] + 1

    # specify task
    task = ForecastingTask(target='target', fh=fh)

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
    assert y_pred.index.equals(y_test.index)

    return y_pred.values


# number of jobs
n_jobs = os.cpu_count()

# set paths
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


#  import meta data
info = pd.read_csv(os.path.join(datadir, 'M4-info.csv'))

# get M4 baseline methods
m4_results = pd.read_excel(os.path.join(repodir, 'Evaluation and Ranks.xlsx'),
                           sheet_name='Point Forecasts-Frequency',
                           header=[0, 1]).dropna(axis=0)
strategies = m4_results.loc[:, ['Method', 'User ID']].iloc[:, 0]
baselines = [strategy for strategy in strategies if isinstance(strategy, str)]
# print('Baseline strategies:', baselines)


# define strategies
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


# select strategies
selected_strategies = ('Naive',) # 'Naive2', 'SES', 'Holt', 'Damped')
print('Selected strategies:', selected_strategies)

strategies = [strategy for strategy in strategies
              if strategy.name in selected_strategies]


# dictionary of forecasting horizons
fhs = info.set_index('SP')['Horizon'].to_dict()


# dictionary of seasonal periodicities
freqs = info.set_index('SP')['Frequency'].to_dict()


# get dataset names
files = os.listdir(os.path.join(traindir))
datasets = [f.split('-')[0] for f in files]


# select datasets
selected_datasets = ('Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly')
print(selected_datasets)


for dataset in selected_datasets:
    # get forecasting horizon
    fh = np.arange(fhs[dataset]) + 1

    #  Get seasonal frequency
    sp = freqs[dataset]

    # print status
    print(f"Dataset: {dataset}, {sp} (sp), {len(fh)} (fh)")

    # get all train and test datasets, make sure to work with numpy arrays to use shared mem for parallelisation
    alltrain = pd.read_csv(os.path.join(traindir, f'{dataset}-train.csv'), index_col=0)
    alltest = pd.read_csv(os.path.join(testdir, f'{dataset}-test.csv'), index_col=0)

    alltrain = alltrain.sort_index().reset_index(drop=True).values
    alltest = alltest.sort_index().reset_index(drop=True).values

    # get number of series in dataset
    n_series = alltrain.shape[0]
    assert n_series == info.SP.value_counts()[dataset]

    # iterate over strategies
    for strategy in strategies:

        # create strategy directory if necessary
        filedir = os.path.join(savedir, strategy.name)
        if not os.path.isdir(filedir):
            os.makedirs(filedir)

        # if results file already exists, skip series
        filename = os.path.join(filedir, f"{strategy.name}_{dataset}_forecasts.txt")
        if os.path.isfile(filename):
            continue

        # iterate over series in dataset
        print('\tStrategy: ', strategy.name)
        y_preds = Parallel(n_jobs=n_jobs)(delayed(run_on_series)(strategy, alltrain[i, :], alltest[i, :], fh, sp)
                                          for i in range(n_series))

        # stack and save results
        np.savetxt(filename, np.vstack(y_preds))
