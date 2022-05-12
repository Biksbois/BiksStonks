# from curses import window
import pandas as pd
import train_prophet
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

from train_lstm import _train_lstma
from train_arima import _train_arima
from train_informer import _train_informer
from train_prophet import _train_prophet

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
        new_dfs.append(df)
    return new_dfs

def get_yahoo_finance_data(index_to_download="ALL", start="2007-12-13", end="2017-12-12"):
    indeces = ["SPY","^HSI","^GDAXI","000001.SS"]
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
    df.to_csv("{}_{}.csv".format(name, idx))

def run_forecast_algorithms_on_dfs(dfs, ws=2):
    args = dotdict()
    arguments = arg.get_arguments()
    arguments.timeunit='1D'
    arguments.initial = '365 days'
    arguments.horizon = '7 days'
    arguments.period = '7 days'
    args.primarycategory = None
    # _train_informer(
    #         args,
    #         [dfs[0]],
    #         ["close"],
    #         seq_len=2,
    #         pred_len=10,
    #         epoch=50,
    #         distil=False,
    #         # d_layers = 1,
    #         # e_layers=4,
    # )
    
    #for idx, df in enumerate(dfs): 
        #i_mae, i_mse, i_r_squared, i_parameters, i_forecasts = _train_prophet(arguments, [df], 'Close', 1)
        #convert_data_to_csv(i_mae, i_mse, i_r_squared, idx, name="Prophet")
    

    # get length of dataframe

    print(len(dfs[1])) 
    
    a_mae, a_mse, a_r_squared, a_parameters, a_forecasts = _train_arima(dfs[1], 10)
    convert_data_to_csv(a_mae, a_mse, a_r_squared, 2, name="arima")

    # for idx, df in enumerate(dfs): 
    #     a_mae, a_mse, a_r_squared, a_parameters, a_forecasts = _train_arima(df, 30)
    #     convert_data_to_csv(a_mae, a_mse, a_r_squared, idx, name="arima")

        #p_mae, p_mse, p_r_squared, p_parameters, p_forecasts = tp.train_prophet()
        #convert_data_to_csv(p_mae, p_mse, p_r_squared, idx, name="prophet")

        #l_mae, l_mse, l_r_squared, l_parameters, l_forecasts = _train_lstma(["close"],[df], ws,n_class=1, Output_size=1, yahoo=True)
        #convert_data_to_csv(l_mae, l_mse, l_r_squared, idx, name="lstm")



if __name__ == "__main__":
    dfs = get_yahoo_finance_data("ALL", "2007-12-13", "2017-12-12")
    
    data = run_forecast_algorithms_on_dfs(dfs)


