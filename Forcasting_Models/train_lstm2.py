import utils.DatasetAccess as db_access
import pandas as pd
from überLSTM2 import LSTM2
from utils.preprocess import add_to_parameters
import torch.nn as nn
import time
def execute_lstm2(arguments, data_lst, from_date, to_date, data, connection):
    for WS in [60, 120]:
        for OS in [1]: #[10, 30]:
            for epoch in [1, 15, 30]:
                start_time = time.time()
                mae, mse, r_squared, parameters, forecasts = _train_lstma2(
                    arguments.columns,
                    data_lst,
                    window_size=WS + OS,
                    Output_size=OS,
                    Epoch=epoch,
                    n_class=len(arguments.columns)
                )
                duration = time.time() - start_time
                parameters["windows_size"] = WS
                parameters["forecasted_points"] = OS

                add_to_parameters(arguments, parameters, duration)

                # if arguments.use_args in ["True", "true", "1"]:
                db_access.upsert_exp_data(
                    "lstm2",  # model name
                    "lstm2 desc",  # model description
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

def _train_lstma2(
    columns,
    data,
    window_size=100,
    n_companies=10,
    n_datapoints=5000,
    Output_size=10,
    n_step=90,
    n_hidden=128,
    n_class=5,
    Epoch=50,
    batch_size=32,
    num_layers=1,
    learning_rate=0.001,
):
    # columns = ["close", "open", "high", "low", "volume"]  # TODO: Use arguments.columns
    print("Retriving data from database...")
    companies = [
        db_access.SingleCompany([x], window_size, Output_size, columns) for x in data
    ]
    train_set, test_set = db_access.GenerateDatasets(companies)

    criterion = nn.MSELoss()
    print("training model...")
    model, r2, mse, mae, plots = LSTM2(
        train_set,
        test_set,
        batch_size,
        Epoch,
        n_hidden,
        n_class,
        learning_rate,
        Output_size,
        num_layers,
        criterion,
    )
    print("Model trained")

    print("Model saved")
    print("Freeing memory...")
    del train_set
    del test_set
    del model
    del companies
    print("Memory freed")

    parameters = {
        "window_size": window_size,
        "n_companies": n_companies,
        "n_datapoints": n_datapoints,
        "Output_size": Output_size,
        "n_step": n_step,
        "n_hidden": n_hidden,
        "n_class": n_class,
        "Epoch": Epoch,
        "batch_size": batch_size,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
    }  # (actual, (y, y_hat))

    actual = [p[0].item() for p in plots[0][0][0]]

    y = [p[0].item() for p in plots[0][1][0]]
    y_hat = [p[0].item() for p in plots[0][1][1]]
    result = data_to_pandas(actual, y, y_hat)

    return mae, mse, r2, parameters, result

def data_to_pandas(actual, y, y_hat):
    obsservation = {"y": actual}
    forcast = {"y": y, "y_hat": y_hat}

    observation_df = pd.DataFrame(data=obsservation)
    forecast_df = pd.DataFrame(data=forcast)

    result = observation_df.append(forecast_df)

    if not "time" in result.columns:
        result = result.append(pd.DataFrame(data={"time": []}))

    return result
