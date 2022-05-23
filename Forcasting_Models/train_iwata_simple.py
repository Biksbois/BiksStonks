import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import r_regression as pearson_r

import sys 
sys.path += ["Informer"]
sys.path += ["Iwata_FewShot"]
from Iwata_FewShot.model import Iwata_simple
from Iwata_FewShot.data import Iwata_Dataset_DB_Stock

from Informer.utils_in.tools import dotdict

import os
import time
import random
import pandas as pd


from utils.preprocess import add_to_parameters
import utils.DatasetAccess as db_access


def execute_iwata_simple(arguments, data_lst, from_date, to_date, data, connection):
    for WS in [60]:
            start_time = time.time()
            mae, mse, r_squared, parameters, forecasts = _train_iwata_simple(
                arguments,
                data_lst,
                arguments.columns,
                data[0].id, # target_id
                connection,
                seq_len=WS,
                pred_len=1, # only 1 for Iwata Simple 
                epoch=5, # 15 epochs? 
            )
            duration = time.time() - start_time

            parameters["WS"] = WS
            parameters["OS"] = parameters.pred_len
            add_to_parameters(arguments, parameters, duration)
            # if arguments.use_args in ["True", "true", "1"]:
            db_access.upsert_exp_data(
                "few-shot",  # model name
                "few-shot desc",  # model description
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

def _train_iwata_simple(arguments, data_lst, columns, target_id, connection,
                        seq_len=None, pred_len=None, epoch=None):
    print("Training Iwata Simple 1 timepoint predictor")
    epochs = epoch
    iwata_params = dotdict()
    iwata_params.columns = columns
    iwata_params.bidirectional = True
    iwata_params.seq_len = seq_len 
    iwata_params.enc_in = len(columns)
    iwata_params.pred_len = 1 # must be for Iwata Simple
    iwata_params.hidden_size = 64
    iwata_params.c_out = 1 # must be 1 for Iwata Simple
    iwata_params.s_n_layers = 2
    iwata_params.S_N = 16 # = support size
    iwata_params.Q_N = 1 # must be 1 for Iwata Simple
    iwata_params.direcs = 2 if iwata_params.bidirectional else 1
    iwata_params.seed_start = 0
    iwata_params.seed_end = iwata_params.seed_start 
    iwata_params.freq = arguments.timeunit
    iwata_params.dropout = 0.2
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # Construct Model 
    model = Iwata_simple(iwata_params.enc_in, iwata_params.hidden_size, iwata_params.c_out, iwata_params.s_n_layers,
                         bidirectional=iwata_params.bidirectional, dropout=iwata_params.dropout)
    model.to(device)


    # Train 
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()


    num_of_stocks = len(data_lst)
    time_now = time.time()
    total_epochs = epochs * num_of_stocks
    min_len = int(((iwata_params.seq_len + iwata_params.pred_len) / 30) * 100) + 1
    for epoch in range(epochs):
        for i, df in enumerate(data_lst):
            target_df = df 

            if len(df) < min_len:
                print("SKIPPINT TRAINING stock {} / {} because it is too short".format(i, num_of_stocks))
                print(f'Min length: {min_len} | Stock length {len(df)}')
                continue

            print(
                ">>>>> Training on stock {}/{} | epoch {}".format(
                    i, num_of_stocks, epoch + 1
                )
            )
            # Load Train DS
            conn = connection
            print(iwata_params.columns)
            print(df.columns)
            iwata_stck_ds = Iwata_Dataset_DB_Stock(conn, iwata_params.S_N, iwata_params.Q_N, size=[seq_len, seq_len, 1],
                                                   flag='train', features='MS', scale=True, freq=iwata_params.freq, 
                                                   hasTargetDF=True, targetDF=target_df, seed=iwata_params.seed_end,
                                                   columns=iwata_params.columns)
            iwata_params.seed_end += 1 # sample differently for each stock
            data_loader = DataLoader(
                        iwata_stck_ds,
                        batch_size=1, # only works with one as they are sampled already from Q_N, S_N
                        shuffle=False,
                        drop_last=True)
            train_steps = len(data_loader)

            train_loss = []
            epoch_time = time.time()
            iter_count = 0
            for s_seq_x, q_seq_x, q_seq_y in data_loader:
                s_seq_x = s_seq_x.float().squeeze(0).to(device) # support set
                q_seq_x = q_seq_x.float().to(device) # query set 
                q_seq_y = q_seq_y.float().to(device) # query set label 

                optimizer.zero_grad()
                # print("input", s_seq_x.shape)
                # print("input", s_seq_x.dtype)
                # print("input", s_seq_x)
                # print("q_seq_y", q_seq_y)
                # print("q_seq_x", q_seq_x.shape)
                # print("q_seq_x", q_seq_x.dtype)
                # print("q_seq_x", q_seq_x)
                output = model(s_seq_x, q_seq_x)
                # print("Output", output)
                loss = criterion(output, q_seq_y)
                # print("Loss", loss)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                iter_count += 1
                if (iter_count) % 100 == 0: 
                    print("Stock {} / {}".format(i, num_of_stocks))
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(iter_count, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_epochs = total_epochs - (epoch * num_of_stocks + i)
                    left_time = speed*(left_epochs*train_steps - iter_count)
                    print('\t(raw estimate) speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            train_loss = np.average(train_loss)
            print(
                    ">>>>> FINISHED Training on stock {}/{} | epoch {}".format(
                        i, num_of_stocks, epoch + 1
                    )
                )
            print('Epoch {}/{} \t Time: {:.2f}s \t Loss: {:.4f}'.format(epoch+1, epochs, time.time() - epoch_time, train_loss))
    
    torch.cuda.empty_cache()

    # Save model 
    stringify_dict = lambda d: '_'.join([k + '_' + str(v) for k, v in d.items()])
    model_name = 'iwata_simple_STOCK' + stringify_dict(iwata_params) + '.pt'
    dir_name = 'iwata_checkpoints'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    model_path = os.path.join(dir_name, model_name)
    # torch.save(model.state_dict(), model_path)


    # Evaluate
    model.eval()
    test_loss = []
    mse = []
    mae = []
    r2 = []
    p_r = []

    with torch.no_grad():
        for i, df in enumerate(data_lst):
            
            if len(df) < min_len:
                print("SKIPPINT TRAINING stock {} / {} because it is too short".format(i, num_of_stocks))
                print(f'Min length: {min_len} | Stock length {len(df)}')
                continue

            print(f'Evaluating on stock {i}/{num_of_stocks}')
            
            predictions = []
            y = []
            # Load Test DS
            iwata_stck_ds = Iwata_Dataset_DB_Stock(conn, iwata_params.S_N, iwata_params.Q_N, size=[seq_len, seq_len, 1],
                                                flag='test', features='MS', scale=True, freq=iwata_params.freq, 
                                                hasTargetDF=True, targetDF=target_df, seed=iwata_params.seed_end,
                                                columns=iwata_params.columns)
            data_loader = DataLoader(
                        iwata_stck_ds,
                        batch_size=1, # only works with one as they are sampled already from Q_N, S_N
                        shuffle=False,
                        drop_last=True)

            for s_seq_x, q_seq_x, q_seq_y in data_loader:
                s_seq_x = s_seq_x.float().squeeze(0).to(device)
                q_seq_x = q_seq_x.float().to(device)
                q_seq_y = q_seq_y.float().to(device)

                output = model(s_seq_x, q_seq_x)
                loss = criterion(output, q_seq_y)
                predictions.append(output.item())
                y.append(q_seq_y.item())
                test_loss.append(loss.item())
            
            if i == 0: # first 
                first_predictions = predictions
                first_y = y
            
            # Compute scores
            predictions = np.asarray(predictions)
            y = np.asarray(y)
            mse.append(mean_squared_error(y, predictions))
            mae.append(mean_absolute_error(y, predictions))
            r2.append(r2_score(y, predictions))
            p_r.append(pearson_r(predictions[...,None], y))

            print('\tFinished Evaluating on {}/{} | Test loss: {:.4f}'.format(i, num_of_stocks, np.average(test_loss)))
            print(f'\tMSE: {mse[-1]} | MAE: {mae[-1]} | R2: {r2[-1]} | P_R: {p_r[-1]}')
            print("toni")

    # iwata_params.p_r = np.mean(p_r)
    test_loss = np.average(test_loss)
    print('Test loss: {:.4f}'.format(test_loss))
    # Compute scores 
    print(f'SCORE OVERVIEW ON COMPANYID {target_id}:')

    # Make forecast df 
    points_to_return = min(300, len(predictions))
    forecast = pd.DataFrame({"y": first_y[:points_to_return], "y_hat": first_predictions[:points_to_return]})

    return np.mean(mae), np.mean(mse), np.mean(r2), iwata_params, forecast