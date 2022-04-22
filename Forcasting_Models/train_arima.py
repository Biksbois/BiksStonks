import utils.preprocess as preprocess
import itertools
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import pandas as pd
from Informer.utils_in.metrics import metric
import utils.DatasetAccess as db_access
from utils.preprocess import add_to_parameters

def execute_arima(data_lst, arguments, from_date, to_date, data, connection):
    mae, mse, r_squared, parameters, forecasts = _train_arima(data_lst)
    add_to_parameters(arguments, parameters, is_fb_or_arima=True)
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
    training, testing = preprocess.get_split_data(data[0], col_name="close")
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

    N_test_observations = len(testing)
    for time_point in range(N_test_observations):
        model = ARIMA(history, order=min_order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = testing.close.iloc[time_point]
        history.append(true_test_value)

        new_row = {
            "time": testing.date.iloc[time_point],
            "y": true_test_value,
            "y_hat": yhat,
        }
        forecasts = forecasts.append(new_row, ignore_index=True)

        # forecasts.loc[len(forecasts)] = [testing.date.iloc[time_point], true_test_value, yhat]
    print(forecasts.tail())

    mae, mse, rmse, mape, mspe, r_squared = metric(model_predictions, testing.close)
    parameters = {
        "p": min_order[0],
        "d": min_order[1],
        "q": min_order[2],
    }

    return mae, mse, r_squared, parameters, forecasts
