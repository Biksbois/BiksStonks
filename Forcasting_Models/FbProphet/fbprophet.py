import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2 as pg
import datetime

import time
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import dates as mdates
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.style.use("ggplot")

import utils.settings_utils as settings
import utils.DatasetAccess as db_access
import utils.preprocess as preprocess
import utils.prophet_experiment as exp
import pickle



def model_fit(training_set, mcmc_samples=300, yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, seasonality_mode='multiplicative'):
    from fbprophet import Prophet
    from fbprophet.diagnostics import cross_validation, performance_metrics
    from fbprophet.plot import plot_cross_validation_metric
    model = Prophet(mcmc_samples=mcmc_samples,
                    seasonality_mode=seasonality_mode,
                    yearly_seasonality=yearly_seasonality, 
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality)
    
    model.fit(training_set)
    
    return model


def get_cross_validation(model, initial, period, horizon):
    from fbprophet.diagnostics import cross_validation, performance_metrics
    return cross_validation(model, initial=initial, period=period, horizon=horizon)

def save_model(model, model_path):
    with open(model_path  + '.pckl', 'wb') as fout:
        pickle.dump(model, fout)

def save_metrics(metrics, metrics_path):
    metrics.to_csv(metrics_path + 'metrics.csv')


def get_performance_metrics(df):
    from fbprophet.diagnostics import cross_validation, performance_metrics
    return performance_metrics(df)

def get_future_df(model, period, freq, include_history=False):
    return model.make_future_dataframe(periods=period, freq=freq, include_history=include_history)

def make_prediction(model, df):
    return model.predict(df)

def plot_cross_validation(df):
    df.plot(x='ds', y=['y', 'yhat'], figsize=(12, 8), grid=True, title='Forecast', legend=True, fontsize=12, linewidth=2)
    plt.show()

def plot_forecast(model, forecast, test_data):
    model.plot(forecast)
    plt.plot(test_data.ds, test_data.y, '.', color='#ff3333', alpha=0.6)
    model.plot_components(forecast)
    plt.show()

def get_rsquared(predicted, actual):
    return r2_score(actual, predicted)