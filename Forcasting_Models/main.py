# from curses import window
from operator import truediv
from unittest.mock import sentinel
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
from train_informer import execute_informer
from train_prophet import execute_prophet
from train_iwata_simple import execute_iwata_simple

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

def get_data(arguments, connection, from_date, to_date, sentiment_cat=None):
    attribute_name = ""
    attribute_value = ""

    primary_category = db_access.get_primay_category(connection)
    secondary_category = db_access.get_secondary_category(connection)
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

    data = db_access.get_data_for_attribute(
        attribute_name,
        attribute_value,
        connection,
        arguments.timeunit,
        from_time=from_date,
        to_time=to_date,
        sentiment_cat= None if sentiment_cat == 'one' else sentiment_cat
    )

    data = [
        d
        for d in data
        if pruning.is_there_enough_points(
            from_date, to_date, d.data.shape[0], 0.7, arguments.timeunit
        )
    ]

    # data = [d for d in data if len(d.data) > 1000]

    if arguments.limit and len(data) > arguments.limit:
        print(
            "Data is too large. Only the first {} rows will be used.".format(
                arguments.limit
            )
        )
        data = data[: arguments.limit]
    else:
        print(f"Data is {len(data)} rows long.")

    print(f"-----------------------------------------------")
    print(f"------ Introducint {len(data)} companies ------")
    print(f"-----------------------------------------------")

    for company in data:
        print(f"  - {company.name}, ({company.id}, {company.data.shape})")

    return data

def run_experiments_nn(arguments, connection, from_date, to_date):

    if arguments.use_sentiment == 'all':
        data = get_data(arguments, connection, from_date, to_date, sentiment_cat=True)
        data_lst = [d.data for d in data]
    elif arguments.use_sentiment == 'one':
        data = get_data(arguments, connection, from_date, to_date, sentiment_cat='one')
        data_lst = [d.data for d in data]
    else:
        data = get_data(arguments, connection, from_date, to_date)
        data_lst = [d.data for d in data]

    if len(data_lst) > 0:
        if arguments.model == "informer" or arguments.model == "all":
            print("about to train the informer")
            execute_informer(arguments, data_lst, from_date, to_date, data, connection)


        if arguments.model == "lstm" or arguments.model == "all":
            print("about to train the lstma model")
            execute_lstm(arguments, data_lst, from_date, to_date, data, connection)

        if arguments.model == 'iwataSimple' or arguments.model == 'all':
            print("about to train the iwata model")
            execute_iwata_simple(arguments, data_lst, from_date, to_date, data, connection)
            
        if arguments.model == "lstm2" or arguments.model == "all":
            print("about to train the lstma2 model")
            execute_lstm2(arguments, data_lst, from_date, to_date, data, connection)
    else:
        print("No data was found. Exiting...")
    
def run_experiments_stat(arguments, connection, from_date, to_date):
    data = get_data(arguments, connection, from_date, to_date)

    data_lst = [d.data for d in data]

    if len(data_lst) > 0:
        if arguments.model == "arima" or arguments.model == "all":
            print("about to train the arima model")
            execute_arima(data_lst[0], arguments, from_date, to_date, data, connection)

        if arguments.model == "fb" or arguments.model == "all":
            print("about to train the fb prophet model")
            execute_prophet(arguments, data_lst, from_date, to_date, data, connection)

    else:
        print("No data was found. Exiting...")


if __name__ == "__main__":
    arguments = arg.get_arguments()
    connection = db_access.get_connection()

    primary_category = db_access.get_primay_category(connection)
    secondary_category = db_access.get_secondary_category(connection)
    company_id = db_access.get_companyid(connection)

    if arguments.use_args in ["False", "false", '0']:
        print("\n\nrunning without parameters\n\n")
        for granularity in oa.granularities:
            if arguments.model in ['iwataSimple', 'lstm', 'informer', 'all']:
                for column in oa.columns_nn:
                    for company in oa.companies_nn:
                        for period in oa.periods_nn:
                            arguments, from_date,to_date = oa.overwrite_arguments(arguments, granularity, column, period, company)
                            if arguments.use_sentiment == 'all' and not oa.sentiment_col['all'][0] in arguments.columns:
                                arguments.columns.extend(oa.sentiment_col['all'])
                            elif arguments.use_sentiment == 'one' and not "compound" in arguments.columns:
                                arguments.columns.append("compound")
                            run_experiments_nn(arguments, connection, from_date, to_date)
            if arguments.model in ['arima', 'fb', 'all']:
                for column in oa.columns_stat:
                    for company in oa.companies_stat:
                        for period in oa.periods_stat:
                            arguments, from_date,to_date = oa.overwrite_arguments(arguments, granularity, column, period, company)
                            run_experiments_stat(arguments, connection, from_date, to_date)
    else:
        print("\n\nrunning with parameters\n\n")
        from_date = "2021-11-20 00:00:00"
        to_date = "2021-12-31 23:59:59"

        run_experiments_nn(arguments, connection, from_date, to_date)
        run_experiments_stat(arguments, connection, from_date, to_date)
