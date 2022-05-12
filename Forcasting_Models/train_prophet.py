from operator import truediv
import os
from cv2 import triangulatePoints
import utils.prophet_experiment as exp
import FbProphet.fbprophet as fb
import datetime
import utils.preprocess as preprocess
import utils.DatasetAccess as db_access
from utils.preprocess import add_to_parameters
import pandas as pd
import time


def execute_prophet(arguments, data_lst, from_date, to_date, data, connection):
    start_time = time.time()
    for os in [1, 2]:
        mae, mse, r_squared, parameters, forecasts = _train_prophet(
            arguments, data_lst, arguments.columns[0], os
        )
        duration = time.time() - start_time
        add_to_parameters(arguments, parameters, duration, is_fb_or_arima=True)

        parameters['forecasted_points'] = os
        # if arguments.use_args in ["True", "true", "1"]:
        db_access.upsert_exp_data(
            "prophet",  # model name
            "prophet desc",  # model description
            mae,  # mae
            mse,  # mse
            r_squared,  # r^2
            from_date,  # data from
            to_date,  # data to
            arguments.timeunit,  # time unit
            data[0].id,  # company name
            parameters,  # model parameters
            arguments.use_sentiment,  # use sentiment
            [d.id for d in data],  # used companies
            arguments.columns,  # used columns
            forecasts,
            connection,
        )

def _train_prophet(arguments, data, column, os):

    parameters = {
        "seasonality_mode": arguments.seasonality_mode,
        "yearly_seasonality": arguments.yearly_seasonality,
        "weekly_seasonality": arguments.weekly_seasonality,
        "daily_seasonality": arguments.daily_seasonality,
        "include_history": arguments.include_history,
        "horizon": arguments.horizon,
        "period": arguments.period,
        "initial": arguments.initial,
    }

    print("initial: ", arguments.initial)
    print("horizon: ", arguments.horizon)
    print("period: ", arguments.period)
    print("data: ", data[0].shape)
    

    data = preprocess.rename_dataset_columns(data[0], column)
    # data['ds'] = pd.to_datetime(data["ds"].dt.strftime('%Y/%m/%d-%H:%M:%S'))
    data[data.columns[1:]] = data[data.columns[1:]].apply(lambda x : (x - x.mean()) / x.std(), axis=0)
    training, testing = preprocess.get_split_data(data)
    # result_path = "./FbProphet/Iteration/"
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)
    # date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # iteration = result_path + date_time + "/"
    # if not os.path.exists(iteration):
    #     os.makedirs(iteration)


    model = fb.model_fit(
        training,
        mcmc_samples=100,
        yearly_seasonality=arguments.yearly_seasonality,
        weekly_seasonality=arguments.weekly_seasonality,
        daily_seasonality=arguments.daily_seasonality,
        seasonality_mode=arguments.seasonality_mode,
    )
    # fb.save_model(model, iteration + "model")
    print("model has been trained, now predicting..")
    future = fb.get_future_df(
        model,
        period=os,
        freq=arguments.timeunit,
        include_history=arguments.include_history,
    )

    print(future.head())
    print(future.tail())
    print("-----------")
    print(data.head())
    print(data.tail())

    # forecast = fb.make_prediction(
    #     model,
    #     future,
    # )

    # fb.save_metrics(forecast, iteration + "forecast.csv")
    print("metrics have been saved, now performing cross eval..")
    cross_validation = fb.get_cross_validation(
        model,
        initial=arguments.initial,
        period=arguments.period,
        horizon=arguments.horizon,
    )
    print("prediction has been made, now saving..")
    print("cross validation has been performed, now saving..")
    forecasts = cross_validation[["ds", "y", "yhat"]].copy()
    forecasts = forecasts.rename(columns={"yhat": "y_hat", "ds": "time"})
    print("calling metrics..")
    metrics = fb.get_performance_metrics(
        cross_validation,
    )
    mse = metrics[["mse"]].copy()
    mae = metrics[["mae"]].copy()
    r_squared = fb.get_rsquared(
        cross_validation[["yhat"]].copy(), cross_validation[["y"]].copy()
    )

    print("done!")
    print(mae, mse, r_squared)

    return mae.mae.mean(), mse.mse.mean(), r_squared, parameters, forecasts
