import pandas as pd
import sys, os
import numpy as np

import torch
import torch.nn as nn

sys.path += ["Informer"]
from Informer.exp.exp_informer import Exp_Informer
from Informer.utils_in.tools import dotdict
from Informer.utils_in.metrics import metric
from Informer.parameters import informer_params
import utils.DatasetAccess as db_access
from utils.preprocess import add_to_parameters
import time


def execute_informer(arguments, data_lst, from_date, to_date, data, connection):
    for WS in [60, 120]:
        for OS in [10, 30]:
            start_time = time.time()
            mae, mse, r_squared, parameters, forecasts = _train_informer(
                arguments,
                data_lst,
                arguments.columns,
                seq_len=WS,
                pred_len=OS,
                epoch=1,
            )
            duration = time.time() - start_time

            parameters["windows_size"] = WS
            parameters["forecasted_points"] = OS
            
            add_to_parameters(arguments, parameters, duration)
            # if arguments.use_args in ["True", "true", "1"]:
            db_access.upsert_exp_data(
                "informer",  # model name
                "informer desc",  # model description
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


def _train_informer(arguments, data, columns, seq_len=None, pred_len=None, epoch=None):
    print("training informer")
    epochs = epoch
    informer_params.train_epochs = 1  # iterate over each df once per epoch
    informer_params.seq_len = seq_len
    informer_params.label_len = seq_len
    informer_params.pred_len = pred_len
    informer_params.train_epochs = 1
    informer_params.target = columns[0]
    informer_params.cols = columns
    informer_params.enc_in = len(columns) # encoder input size # ohlc + volume + [trade_count, vwap]
    informer_params.dec_in = len(columns) # decoder input size # ohlc + volume + [trade_count, vwap]

    exp = Exp_Informer(informer_params)  # here we can change the parameters
    num_of_stocks = len(data)
    for epoch in range(epochs):
        for i, df in enumerate(data):
            informer_params.df = df
            # args.df = args.df[len(args.df)//2:]

            print(
                ">>>>> Training on stock {}/{} | epoch {}".format(
                    i, num_of_stocks, epoch + 1
                )
            )

            name = ""
            if arguments.primarycategory:
                name = "primarycategory"
            elif arguments.secondarycategory:
                name = "secondarycategory"
            elif arguments.companyid:
                name = "companyid"

            # setting record of experiments
            args = informer_params
            setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}".format(
                name,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.attn,
                args.factor,
                args.embed,
                args.distil,
                args.mix,
                args.des,
            )

            # train
            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)
    torch.cuda.empty_cache()

    # test
    print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
    mae_l, mse_l, rmse_l, mape_l, mspe_l, rs2_l, rs2_intermed_l = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i, df in enumerate(data):
        informer_params.df = df
        test_data, test_loader = exp._get_data(flag="test")
        exp.model.eval()
        preds = []
        trues = []

        for j, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = exp._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark
            )
            if i == 0 and j == 0:
                in_seq = batch_x[0, :, -1].detach().cpu().numpy()
            
            rs2_intermed_l.append(
                r2_score_dim(torch.tensor(pred), torch.tensor(true))
            )
            
            pred = pred.detach().cpu().numpy()
            true = true.detach().cpu().numpy()
            preds.append(pred)
            trues.append(true)

        if i == 0:
            first_pred = preds[0][0, :, 0]
            first_true = trues[0][0, :, 0]

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

    folder_path = "./results/" + setting + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    metrics_per_stock = np.array([mae_l, mse_l, rmse_l, mape_l, mspe_l, rs2_l])
    np.save(folder_path + "metrics_per_stock.npy", metrics_per_stock)
    np.save(
        folder_path + "metrics_agg.npy",
        np.mean(metrics_per_stock + [np.mean(rs2_intermed_l)], axis=1),
    )
    print(">>>>> Done!")
    print(
        f"Metrics: MSE {np.mean(mse_l):.2f}, RMSE {np.mean(rmse_l):.2f}, MAE {np.mean(mae_l):.2f},\
        MAPE {np.mean(mape_l):.2f}, MSPE {np.mean(mspe_l):.2f}, R2 {np.mean(rs2_l):.2f}, R2 IM {np.mean(rs2_intermed_l)}"
    )
    mae, mse, r_squared = np.mean(mae_l), np.mean(mse_l), np.mean(rs2_l)
    informer_params.df = None
    informer_params.rs2_intermediate = np.mean(rs2_intermed_l)
    informer_params.rs2_long = r_squared
    parameters = informer_params
    y_hat = first_pred.reshape(-1)
    y = np.concatenate((in_seq, first_true.reshape(-1)))
    forecast = pd.DataFrame({"y": y, "y_hat": np.nan})
    forecast["y_hat"][in_seq.shape[0] :] = y_hat

    return mae, mse, np.mean(rs2_intermed_l), parameters, forecast

def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def r2_score_dim(output, target):
    target_mean = torch.mean(target, dim=1, keepdim=True)
    ss_tot = torch.sum((target - target_mean) ** 2, dim=1)
    ss_res = torch.sum((target - output) ** 2, dim=1)
    r2 = 1 - (ss_res / (ss_tot))
    return torch.mean(r2)
