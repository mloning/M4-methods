from sktime.forecasters import DummyForecaster
from sktime.forecasters import ExpSmoothingForecaster
from sktime.forecasters.compose import EnsembleForecaster
from sktime.pipeline import Pipeline
from sktime.forecasters.compose import TransformedTargetForecaster, ReducedForecastingRegressor
from sktime.transformers.series_to_series import Deseasonaliser, Detrender
from sktime.transformers.compose import Tabulariser

from sklearn.neural_network import MLPRegressor


def define_models(sp):
    """Helper function to define models"""

    # naive baselines
    mean = DummyForecaster(strategy='mean')
    naive = DummyForecaster(strategy='last')  # without seasonality adjustments
    snaive = DummyForecaster(strategy='seasonal_last', sp=sp)
    naive2 = ExpSmoothingForecaster(smoothing_level=1)  # with seasonality adjustments

    # statistical methods
    kwargs = {'seasonal': 'multiplicative', 'seasonal_periods': sp} if sp > 1 else {}
    ses = ExpSmoothingForecaster(**kwargs)
    holt = ExpSmoothingForecaster(trend='add', damped=False, **kwargs)
    damped = ExpSmoothingForecaster(trend='add', damped=True, **kwargs)
    comb = EnsembleForecaster(estimators=[('ses', ses),
                                          ('holt', holt),
                                          ('damped', damped)])
    # theta =

    # machine learning methods
    mlp = TransformedTargetForecaster(
        forecaster=ReducedForecastingRegressor(
            estimator=Pipeline([
                ('tabularise', Tabulariser()),
                ('regressor', MLPRegressor(hidden_layer_sizes=6, activation='identity', solver='adam',
                                           max_iter=100, learning_rate='adaptive', learning_rate_init=0.001))
            ])),
        transformer=Pipeline([
            ('deseasonalise', Deseasonaliser(model='multiplicative', sp=sp)),
            ('detrend', Detrender(order=1))
        ]))

    models = {
        'Mean': mean,
        'Naive': naive,
        'sNaive': snaive,
        'Naive2': naive2,
        'SES': ses,
        'Holt': holt,
        'Damped': damped,
        'Comb': comb,
        'MLP': mlp
    }

    return models

