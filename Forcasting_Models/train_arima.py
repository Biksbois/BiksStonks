from locale import normalize

import utils.preprocess as preprocess
import itertools
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import pandas as pd
from Informer.utils_in.metrics import metric
import FbProphet.fbprophet as fb
import utils.DatasetAccess as db_access
from sklearn.metrics import mean_squared_error, r2_score
from utils.preprocess import add_to_parameters
import numpy as np
from tqdm import tqdm
import time
from train_informer import create_shifted_mean
from sklearn.metrics import r2_score as sk_r2_score


def execute_arima(data_lst, arguments, from_date, to_date, data, connection):
    start_time = time.time()
    # for WS in [1, 2, 10]:
    for WS in [2, 10]:
        mae, mse, r_squared, parameters, forecasts = _train_arima(data_lst, WS)
        duration = time.time() - start_time
        add_to_parameters(arguments, parameters, duration, is_fb_or_arima=True)
        parameters['forecasted_points'] = WS
        db_access.upsert_exp_data(
            "arima",  # model name
            "arima desc",  # model description
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


def _train_arima(data, WS):
    #Normalize the pandas dataframe 
    from utils.preprocess import StandardScaler
    scaler = StandardScaler()

    # scaler.fit(data[data.columns[1:]].values)
    # data_scaled = scaler.transform(data.values)
    # data_ = pd.DataFrame(dict(zip(data.columns[1:], data_scaled)))
    # data_['date'] = data['date']

    data[data.columns[1:]] = data[data.columns[1:]].apply(lambda x : (x - x.mean()) / x.std(), axis=0)
    training, testing = preprocess.get_split_data(data, col_name="date")
    #split seems to work though a bit cryptic
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))

    pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    ans = []
    for comb in pdq:
        for combs in pdqs:  
            try:
                mod = sm.tsa.statespace.SARIMAX(
                    training.close,
                    order=comb,
                    seasonal_order=combs,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                output = mod.fit()
                ans.append([comb, combs, output.aic])
            except Exception as e:
                print(f"ERROR: {str(e)}")
                continue
    # Find the parameters with minimal AIC value
    ans_df = pd.DataFrame(ans, columns=["pdq", "pdqs", "aic"])
    min_order = ans_df.loc[ans_df["aic"].idxmin()][0]

    history = [x for x in training.close]
    model_predictions = []

    forecasts = pd.DataFrame(columns=["time", "y", "y_hat"])
    forecasts["time"] = training["date"][-100:]
    forecasts["y"] = training["close"][-100:]
    out_steps=WS
    N_test_observations = len(testing)
    test_ = []
    r2_scores = []
    other_r2s = []
    first = True
    mae, mse, rmse, mape, mspe, r_squared = 0, 0, 0, 0, 0, 0
    if WS == 1:
        for time_point in tqdm(range(0, N_test_observations-out_steps+1), desc="Forecasting with ARIMA..."):
            model = ARIMA(history, order=min_order)
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output
            model_predictions.append(yhat)
            true_test_value = testing.close.iloc[time_point:time_point+out_steps]
            history.extend(true_test_value)
            test_.append(true_test_value.values[0])
            if first:
                history_lengt = min(100, len(list(history[-100:])), len(list(training.date.iloc[-100:])))
                forecasts = pd.DataFrame({'time': list(training.date.iloc[-history_lengt:]) + list(testing.date.iloc[time_point:time_point+out_steps]),
                                        'y': list(history[-history_lengt:]) + list(true_test_value),
                                        'y_hat':np.nan 
                })

                forecasts['y_hat'][history_lengt:] = yhat
                first = False
        mae,mse,rmse,mape,mspe,r_squared = metric(np.array(test_), np.array(model_predictions))
        r_squared = r2_score(np.array(test_), np.array(model_predictions))
    else:
        # measure time for forecasting
        time_start = time.time()

        for time_point in tqdm(range(0, N_test_observations-out_steps+1), desc="Forecasting with ARIMA..."):
            model = ARIMA(history, order=min_order)
            model_fit = model.fit()
            output = model_fit.forecast(steps=out_steps)
            yhat = output
            model_predictions.append(yhat)
            true_test_value = testing.close.iloc[time_point:time_point+out_steps]
            history.extend(true_test_value)
            test_.append(true_test_value)
            r2_scores.append(r2_score(true_test_value.values,output))



            if first:
                history_lengt = min(100, len(list(history[-100:])), len(list(training.date.iloc[-100:])))
                forecasts = pd.DataFrame({'time': list(training.date.iloc[-history_lengt:]) + list(testing.date.iloc[time_point:time_point+out_steps]),
                                        'y': list(history[-history_lengt:]) + list(true_test_value),
                                        'y_hat':np.nan 
                })

                forecasts['y_hat'][history_lengt:] = yhat
                first = False
        # measure time for forecasting
        print("Forecasting time: %.2f seconds" % (time.time() - time_start))

        model_predictions = np.asarray(model_predictions)
        test_ = np.asarray(test_)

        trues = create_shifted_mean(test_, 1, True)
        preds = create_shifted_mean(model_predictions, 1, True)

        # comp r2 scoty 
        
        # print funny joke below, to celebrate completion of prog.


        mae,mse,rmse,mape,mspe,r_squared = metric(np.array(test_), np.array(model_predictions))
        print(trues)
        print(preds)
        r_squared = sk_r2_score(trues, preds)
    
    test_ = np.asarray(test_)
    model_predictions = np.asarray(model_predictions)
    if model_predictions.shape != test_.shape:
        print("ERROR: model predictions and test data have different shapes")
        print(f"model predictions shape: {model_predictions.shape}")
        print(f"test data shape: {test_.shape}")
    
    # 300, 10
    # 300, 10
    parameters = {
        "p": min_order[0],
        "d": min_order[1],
        "q": min_order[2],
    }

    return mae.mean(), mse.mean(), r_squared, parameters, forecasts
