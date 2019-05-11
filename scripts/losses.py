import numpy as np
from sklearn.utils.validation import check_consistent_length

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
