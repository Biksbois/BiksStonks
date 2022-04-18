#from curses import window
import random
import pandas as pd
from überLSTM import LSTM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.DatasetAccess as db_access
import utils.preprocess as preprocess
import utils.arguments as arg
import warnings
import pickle

import itertools
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

import os
import sys
sys.path += ["Informer"]
from Informer.exp.exp_informer import Exp_Informer
from Informer.utils_in.tools import dotdict
from Informer.utils_in.metrics import metric
from Informer.parameters import informer_params


warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.catch_warnings()


def ensure_valid_values(input_values, actual_values, value_type):
    for value in input_values:
        if not value in actual_values:
            raise Exception(f"Value '{value}' is not a valid {value_type}")
    return True


def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def get_data(arguments, connection, from_date, to_date):
    attribute_name = ""
    attribute_value = ""

    primary_category = db_access.get_primay_category(connection)
    secondary_category = db_access.get_secondary_category(connection)
    # primary_category = db_access.get_primay_category(connection)
    # secondary_category = db_access.get_secondary_category(connection)
    company_id = db_access.get_companyid(connection)

    if arguments.primarycategory:
        if ensure_valid_values(
            arguments.primarycategory, primary_category, "primary category"
        ):
            print(
                f"Models will be trained on companies with primary category in {arguments.primarycategory}"
            )
            attribute_name = "primarycategory"
            attribute_value = [f"'{x}'" for x in arguments.primarycategory]

    elif arguments.secondarycategory:
        if ensure_valid_values(
            arguments.secondarycategory, secondary_category, "secondary category"
        ):
            print(
                f"Models will be trained on companies with secondary category in {arguments.secondarycategory}"
            )
            attribute_name = "secondarycategory"
            attribute_value = [f"'{x}'" for x in arguments.secondarycategory]

    elif arguments.companyid:
        if ensure_valid_values(
            [int(x) for x in arguments.companyid], company_id, "companyid"
        ):
            print(
                f"Models will be trained on companies with company id in {arguments.companyid}"
            )
            attribute_name = "identifier"
            attribute_value = arguments.companyid

    else:
        print("No information was provided. No models will be trained.")
        return pd.DataFrame()

    return db_access.get_data_for_attribute(
        attribute_name,
        attribute_value,
        connection,
        arguments.timeunit,
        from_time=from_date,
        to_time=to_date,
    )


def train_lstma(data,
    window_size = 100,
    n_companies = 10,
    n_datapoints = 5000,
    Output_size = 10,
    n_step = 90,
    n_hidden = 128,
    n_class = 5,
    Epoch = 50,
    batch_size = 32,
    num_layers = 1,
    learning_rate = 0.001):
    columns = ['close','open','high','low','volume']
    print("Retriving data from database...")
    companies = [db_access.SingleCompany([x],window_size,Output_size,columns) for x in data]
    train_set, test_set = db_access.GenerateDatasets(companies)

    criterion = nn.MSELoss()
    print("training model...")
    model,r2,mse,mae = LSTM(
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
    print("Saving model...")
    pickle.dump( model, open(f"LSTM_Models/R2_{r2}_MSE_{mse}_MAE_{mae}_model_LayerN_{num_layers}_BatchSize_{batch_size}_Epoch_{Epoch}_NHidden_{n_hidden}_NClass_{n_class}_LR_{learning_rate}_WinodwSize_{window_size}_OutputSize_{Output_size}.p", "wb" ) )
    print("Model saved")
    print("Freeing memory...")
    del train_set
    del test_set
    del model
    del companies
    print("Memory freed")

def train_informer(arguments, data, seq_len = None, pred_len = None, epoch = None):
    print("training informer")
    epochs = epoch
    informer_params.train_epochs = 1 # iterate over each df once per epoch
    informer_params.seq_len =seq_len
    informer_params.label_len = seq_len
    informer_params.pred_len = pred_len
    informer_params.train_epochs = 1
    exp = Exp_Informer(informer_params) # here we can change the parameters
    num_of_stocks = len(data)
    for epoch in range(epochs):
            for i, df in enumerate(data):
                    informer_params.df = df
                    # args.df = args.df[len(args.df)//2:]

                    print('>>>>> Training on stock {}/{} | epoch {}'.format(i, num_of_stocks, epoch+1))
                    
                    name = ''
                    if arguments.primarycategory:
                            name = 'primarycategory'
                    elif arguments.secondarycategory:
                            name = 'secondarycategory'
                    elif arguments.companyid:
                            name = 'companyid'

                    # setting record of experiments
                    args = informer_params
                    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}'.format(name, args.model, args.data, args.features, 
                            args.seq_len, args.label_len, args.pred_len,
                            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des)
                    
                    # train
                    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                    exp.train(setting)
    torch.cuda.empty_cache()

    # test
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mae_l, mse_l, rmse_l, mape_l, mspe_l, rs2_l, rs2_intermed_l = [], [], [], [], [], [], []
    for i, df in enumerate(data):
            informer_params.df = df
            test_data, test_loader = exp._get_data(flag='test')
            exp.model.eval()
            preds = []
            trues = []
            
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
                pred, true = exp._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                pred = pred.detach().cpu().numpy()
                true = true.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                rs2_intermed_l.append(r2_score(torch.tensor(pred.reshape(-1)), torch.tensor(true.reshape(-1))))


            preds = np.array(preds)
            trues = np.array(trues)
            preds = preds.reshape(-1)
            trues = trues.reshape(-1)
            
            mae, mse, rmse, mape, mspe, r_squared = metric(preds, trues)
            mae_l.append(mae)
            mse_l.append(mse)
            rmse_l.append(rmse)
            mape_l.append(mape)
            mspe_l.append(mspe)
            rs2_l.append(r_squared)

    torch.cuda.empty_cache()
            
    folder_path = './results/' + setting +'/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    metrics_per_stock = np.array([mae_l, mse_l, rmse_l, mape_l, mspe_l, rs2_l])
    np.save(folder_path+'metrics_per_stock.npy', metrics_per_stock)
    np.save(folder_path+'metrics_agg.npy', np.mean(metrics_per_stock+[np.mean(rs2_intermed_l)], axis=1))    
    print('>>>>> Done!')
    print(f'Metrics: MSE {np.mean(mse_l):.2f}, RMSE {np.mean(rmse_l):.2f}, MAE {np.mean(mae_l):.2f},\
        MAPE {np.mean(mape_l):.2f}, MSPE {np.mean(mspe_l):.2f}, R2 {np.mean(rs2_l):.2f}, R2 IM {np.mean(rs2_intermed_l)}')



def train_prophet(arguments, data):
    from cv2 import triangulatePoints
    import utils.prophet_experiment as exp
    import FbProphet.fbprophet as fb
    import datetime
    
    data = preprocess.rename_dataset_columns(data[0])
    training, testing = preprocess.get_split_data(data)
    result_path = './FbProphet/Iteration/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    iteration = result_path + date_time + '/'
    if not os.path.exists(iteration):
        os.makedirs(iteration)
        
    
    model = fb.model_fit(
        training,
        yearly_seasonality=arguments.yearly_seasonality,
        weekly_seasonality=arguments.weekly_seasonality,    
        daily_seasonality=arguments.daily_seasonality,
        seasonality_mode=arguments.seasonality_mode,
    )
    fb.save_model(model, iteration + 'model'+ arguments)
    print("model has been trained, now predicting..")

    future = fb.get_future_df(
        model,
        period=len(testing),
        freq=arguments.timeunit,
        include_history=arguments.include_history,
    )

    forecast = fb.make_prediction(
        model,
        future,
    )

    forecast.to_csv(iteration + 'forecast.csv')

    # e = exp.Experiment(arguments.timeunit, arguments.predict_periods)
    cross_validation = fb.get_cross_validation(model, horizon=arguments.horizon)

    metrics = fb.get_performance_metrics(
        cross_validation,
    )
    # save metrics to csv
    metrics.to_csv(iteration + 'metrics.csv')

    print("Performance \n")
    metrics.head(10)

    print("-------Cross Validation Plot-------")
    fb.plot_cross_validation(cross_validation)

    print("-------Fututre Forcast Plot-------")
    fb.plot_forecast(
        model,
        forecast,
        testing,
    )
    print("done!")

def train_arima(data):
    training, testing = preprocess.get_split_data(data[0], col_name="close")
    mae_l, mse_l, rmse_l, mape_l, mspe_l, rs2_l = [], [], [], [], [], []
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))

    pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    ans = []
    for comb in pdq:
        for combs in pdqs:
            try:
                mod = sm.tsa.statespace.SARIMAX(training.close,
                                            order=comb,
                                            seasonal_order=combs,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

                output = mod.fit()
                ans.append([comb, combs, output.aic])
            except Exception as e:
                print(f"ERROR: {str(e)}")
                continue
    # Find the parameters with minimal AIC value
    ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic'])
    min_order = ans_df.loc[ans_df['aic'].idxmin()][0]

    history = [x for x in training.close]
    model_predictions = []
    N_test_observations = len(testing)
    for time_point in range(N_test_observations):
        model = ARIMA(history, order=min_order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = testing.close.iloc[time_point]
        history.append(true_test_value)

    mae, mse, rmse, mape, mspe, r_squared = metric(model_predictions, testing.close)
    mae_l.append(mae)
    mse_l.append(mse)
    rmse_l.append(rmse)
    mape_l.append(mape)
    mspe_l.append(mspe)
    rs2_l.append(r_squared)

if __name__ == "__main__":
    arguments = arg.get_arguments()

    connection = db_access.get_connection()

    # primary_category = db_access.get_primay_category(connection)
    # secondary_category = db_access.get_secondary_category(connection)
    company_id = db_access.get_companyid(connection)

    from_date = "2020-12-31 00:00:00"
    to_date = "2021-12-31 23:59:59"

    data = get_data(arguments, connection, from_date, to_date)

    if len(data) > 0:
        if arguments.model == "fb" or arguments.model == "all":
            print("about to train the fb prophet model")
            train_prophet(arguments, data)

        if arguments.model == "informer" or arguments.model == "all":
            print("about to train the informer")
            for WS in [60,120]:
                for OS in [10,30]: 
                    train_informer(arguments, data, seq_len = WS, pred_len = OS, epoch = 25)

        if arguments.model == "lstm" or arguments.model == "all":
            print("about to train the lstma model")
            for WS in [60,120]:
                for OS in [10,30]:
                    train_lstma(data, window_size=WS+OS, Output_size=OS, Epoch = 25)
            train_lstma(data)
 
        if arguments.model == "arima" or arguments.model == "all":
            print("about to train the arima model")
            train_arima(data)
    else:
        print("No data was found. Exiting...")
