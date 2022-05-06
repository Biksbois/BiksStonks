# from curses import window
import pandas as pd
from utils.data_obj import DataObj
import random
import pandas as pd
from psycopg2 import paramstyle
from überLSTM import LSTM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.DatasetAccess as db_access
import utils.preprocess as preprocess
import utils.Pruning as pruning
import utils.arguments as arg
import warnings
import pickle
from datetime import datetime
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import utils.overwrite_arguments as oa

from train_lstm import execute_lstm
from train_lstm2 import execute_lstm2
from train_arima import execute_arima
from train_informer import _train_informer
from train_prophet import execute_prophet
from train_iwata_simple import execute_iwata_simple

import os
import sys

sys.path += ["Informer"]
from Informer.exp.exp_informer import Exp_Informer
from Informer.utils_in.tools import dotdict
from Informer.utils_in.metrics import metric
from Informer.parameters import informer_params

from lib2to3.pytree import convert
import yfinance as yf

def rename_df_columns(dfs):
    new_dfs = []
    for df in dfs:
        df = df.reset_index()
        df = df.rename(columns={"Close": "close"})
        df = df.rename(columns={"Date": "date"})
        print(df.head())
        new_dfs.append(df)
    return new_dfs

def get_yahoo_finance_data(index_to_download="ALL", start="2007-12-13", end="2017-12-12"):
    indeces = ["SPY","HSI","000001.SS"]
    dfs = []

    if index_to_download == "ALL":
        for index in indeces:
            dfs.append(yf.download(index, start, end))
    else:
        dfs.append(yf.download(index_to_download, start, end))

    dfs = rename_df_columns(dfs)
    return dfs

def convert_data_to_csv(mae, mse, r_squared, idx, name="placeholder"): 
    print("mae: ", mae)
    print("mse: ", mse)
    print("r_squared: ", r_squared)
    df =  pd.DataFrame(columns=["mae", "mse", "r_squared"], data=[[mae, mse, r_squared]])
    print(df.head())
    df.to_csv("{}_{}.csv".format(name, idx))

def run_forecast_algorithms_on_dfs(dfs, ws=2):
    for idx, df in enumerate(dfs): 
        #a_mae, a_mse, a_r_squared, a_parameters, a_forecasts = _train_arima(df, ws)
        #convert_data_to_csv(a_mae, a_mse, a_r_squared, idx, name="arima")

        #p_mae, p_mse, p_r_squared, p_parameters, p_forecasts = tp.train_prophet()
        #convert_data_to_csv(p_mae, p_mse, p_r_squared, idx, name="prophet")

        #l_mae, l_mse, l_r_squared, l_parameters, l_forecasts = _train_lstma(["close"],df, ws)
        #convert_data_to_csv(l_mae, l_mse, l_r_squared, idx, name="lstm")
        
        i_mae, i_mse, i_r_squared, i_parameters, i_forecasts =  _train_informer(
                dict(),
                [df],
                ["close"],
                seq_len=2,
                pred_len=1,
                epoch=50,
            )

        #convert_data_to_csv(i_mae, i_mse, i_r_squared, idx, name="informer")



if __name__ == "__main__":
    dfs = get_yahoo_finance_data(start="2007-11-13", end="2007-12-12")
    for df in dfs:
        print(df.head())
    data = run_forecast_algorithms_on_dfs(dfs)