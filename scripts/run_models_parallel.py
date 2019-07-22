#!/usr/bin/env python3 -u

# load packages
import numpy as np
from joblib import Parallel
from joblib import delayed
import pandas as pd
import os
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
from copy import deepcopy

# import models
from models import define_models


# define function to run training/prediction in parallel
def run_on_series(model, y_train, y_test, fh):

    # silence warnings
    filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')

    # remove missing values from padding
    y_train = y_train[~np.isnan(y_train)]
    y_test = y_test[~np.isnan(y_test)]

    # check forecasting horizon
    assert len(fh) == len(y_test)

    # get train data into expected format
    # train = pd.DataFrame([pd.Series([pd.Series(y_train)])])
    train = pd.Series([pd.Series(y_train)])

    # n_obs = len(train.iloc[0, 0])
    n_obs = len(train.iloc[0])

    # adjust test index to be after train index
    y_test = pd.Series(y_test)
    y_test.index = y_test.index + n_obs
    # assert y_test.index[0] == train.iloc[0, 0].index[-1] + 1
    assert y_test.index[0] == train.iloc[0].index[-1] + 1

    # clone strategy
    m = deepcopy(model)

    # fit and predict
    m.fit(train, fh=fh)
    y_pred = m.predict(fh=fh)
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
savedir = os.path.join(repodir, "predictions/second_run")

assert os.path.exists(repodir)
assert os.path.exists(datadir)
assert os.path.exists(traindir)
assert os.path.exists(testdir)
assert os.path.exists(savedir)
print('Results directory: ', savedir)


# import meta data
info = pd.read_csv(os.path.join(datadir, 'M4-info.csv'))

# get M4 baseline methods
m4_results = pd.read_excel(os.path.join(repodir, 'Evaluation and Ranks.xlsx'),
                           sheet_name='Point Forecasts-Frequency',
                           header=[0, 1]).dropna(axis=0)
strategies = m4_results.loc[:, ['Method', 'User ID']].iloc[:, 0]


# select models
models = define_models(1)
selected_models = list(models.keys())  # ('MLP', 'Naive2', 'SES', 'Holt', 'Damped')
print('Selected models:', selected_models)

# dictionary of forecasting horizons and seasonal periodicities
fhs = info.set_index('SP')['Horizon'].to_dict()
sps = info.set_index('SP')['Frequency'].to_dict()

# get dataset names
files = os.listdir(os.path.join(traindir))
datasets = [f.split('-')[0] for f in files]

# select datasets
selected_datasets = ('Hourly',)  #'('Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly')
print("Selected datasets: ", selected_datasets)

for dataset in selected_datasets:
    # get forecasting horizon
    fh = np.arange(fhs[dataset]) + 1

    #  Get seasonal frequency
    sp = sps[dataset]

    # define and select models
    models = define_models(sp)
    models = {name: model for name, model in models.items() if name in selected_models}

    # print status
    print(f"Dataset: {dataset}, sp: {sp}, fh: {len(fh)}")

    # get all train and test datasets, make sure to work with numpy arrays to use shared mem for parallelisation
    alltrain = pd.read_csv(os.path.join(traindir, f'{dataset}-train.csv'), index_col=0)
    alltest = pd.read_csv(os.path.join(testdir, f'{dataset}-test.csv'), index_col=0)

    alltrain = alltrain.sort_index().reset_index(drop=True).values
    alltest = alltest.sort_index().reset_index(drop=True).values

    # get number of series in dataset
    n_series = alltrain.shape[0]
    assert n_series == info.SP.value_counts()[dataset]

    # iterate over strategies
    for name, model in models.items():

        # create strategy directory if necessary
        filedir = os.path.join(savedir, name)
        if not os.path.isdir(filedir):
            os.makedirs(filedir)

        # if results file already exists, skip series
        filename = os.path.join(filedir, f"{name}_{dataset}_forecasts.txt")
        if os.path.isfile(filename):
            print(f"\tSkipping {name} on {dataset}, forecasts already exists")
            continue

        # iterate over series in dataset
        print('\tModel: ', name)
        y_preds = Parallel(n_jobs=n_jobs)(delayed(run_on_series)(model, alltrain[i, :], alltest[i, :], fh)
                                          for i in range(n_series))

        # stack and save results
        np.savetxt(filename, np.vstack(y_preds))
