import numpy as np
from sklearn.utils.validation import check_consistent_length

__author__ = ['Markus Loning']

# for reference implementations, see https://github.com/M4Competition/M4-methods/blob/master/ML_benchmarks.py


def mase_loss(y_true, y_pred, y_train, sp=1):
    """Mean Absolute Scaled Error
    insample: insample data
    y_true: out of sample target values
    y_pred: predicted values
    sp: data frequency
    """
    check_consistent_length(y_true, y_pred)

    #  naive seasonal prediction
    y_train = np.asarray(y_train)
    y_pred_naive = y_train[:-sp]
    mae_naive = np.mean(np.abs(y_train[sp:] - y_pred_naive))

    return np.mean(np.abs(y_true - y_pred)) / mae_naive


def smape_loss(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error
    """
    check_consistent_length(y_true, y_pred)

    nominator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    return 2 * np.mean(nominator / denominator)
