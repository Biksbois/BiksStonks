from locale import normalize
import utils.preprocess as preprocess
import itertools
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import pandas as pd
from Informer.utils_in.metrics import metric
import utils.DatasetAccess as db_access
from utils.preprocess import add_to_parameters
import numpy as np

def execute_arima(data_lst, arguments, from_date, to_date, data, connection):
    mae, mse, r_squared, parameters, forecasts = _train_arima(data_lst)
    add_to_parameters(arguments, parameters, is_fb_or_arima=True)
    # if arguments.use_args in ["True", "true", "1"]:

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


def _train_arima(data):
    #Normalize the pandas dataframe 
    from utils.preprocess import StandardScaler
    scaler = StandardScaler()

    # scaler.fit(data[data.columns[1:]].values)
    # data_scaled = scaler.transform(data.values)
    # data_ = pd.DataFrame(dict(zip(data.columns[1:], data_scaled)))
    # data_['date'] = data['date']

    data = data.apply(lambda x : (x - x.mean()) / x.std())

    training, testing = preprocess.get_split_data(data, col_name="close")
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
    out_steps=10
    N_test_observations = len(testing)
    test_ = []
    first = True
    for time_point in range(0, N_test_observations-out_steps+1, 10):
        model = ARIMA(history, order=min_order)
        model_fit = model.fit()
        output = model_fit.forecast(steps=out_steps)
        yhat = output
        model_predictions.append(yhat)
        true_test_value = testing.close.iloc[time_point:time_point+out_steps]
        history.extend(true_test_value)
        test_.append(true_test_value)

        if first:
            forecasts = pd.DataFrame({'time': training.date.iloc[-100:] + testing.date.iloc[time_point:time_point+out_steps],
                                    'y': list(history[-100:]) + list(true_test_value),
                                    'y_hat':np.nan 
            })

            forecasts['y_hat'][100:] = yhat
            first = False
    print(forecasts.tail())
    
    test_ = np.asarray(test_)
    model_predictions = np.asarray(model_predictions)
    if model_predictions.shape != test_.shape:
        print("ERROR: model predictions and test data have different shapes")
        print(f"model predictions shape: {model_predictions.shape}")
        print(f"test data shape: {test_.shape}")
    mae, mse, rmse, mape, mspe, r_squared = metric(model_predictions, test_)
    # 300, 10
    # 300, 10
    parameters = {
        "p": min_order[0],
        "d": min_order[1],
        "q": min_order[2],
    }

    return mae.mean(), mse.mean(), r_squared.mean(), parameters, forecasts
