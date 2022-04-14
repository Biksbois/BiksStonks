from pydoc import describe
from unittest import result
import psycopg2 as pg
from utils.DatabaseConnection import DatabaseConnection
import utils.preprocess as preprocess
import utils.settings_utils as settings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import islice


class DatasetAccess:
    def __init__(self):
        self.conn = DatabaseConnection()

    def getAllcompanies(self):
        AllCompanies = self.conn.query("SELECT * FROM dataset")
        return AllCompanies

    def GetNumberOfCompanies(self):
        return self.conn.query("SELECT count(*) FROM dataset")

    def getNcompanies(self, N):
        AllCompanies = self.conn.query("SELECT * FROM dataset limit " + str(N) + "")
        return AllCompanies

    def getStockFromSymbol(self, StockSymbol, column="*"):
        company = self.conn.query(
            "SELECT * FROM dataset WHERE symbol = '" + StockSymbol + "'"
        )
        self.getStockFromCompany(company, column)
        return company

    def getStockFromCompany(self, companies, column="*"):
        result = []
        for company in companies:
            result.append(
                self.conn.query(
                    "SELECT "
                    + self.convertListToString(column)
                    + " FROM stock WHERE identifier = '"
                    + str(company[0])
                    + "'"
                )
            )
        return result

    def getStockDFFromCompany(self, companies, column="*"):
        result = []
        for company in companies:
            result.append(
                pd.read_sql(
                    "SELECT * FROM stock WHERE identifier = '" + str(company[0]) + "'",
                    self.conn.GetConnector(),
                )
            )
        return result

    def convertListToString(self, column):
        if type(column) != list:
            return column
        result = ""
        for item in column:
            result += item + ", "
        return result[:-2]

    def GetAllStocksAsDF(self):
        PandaStock = pd.read_sql("SELECT * FROM stock", self.conn.GetConnector())
        print(PandaStock)


def GetNumberOfCompanies():
    dbAccess = DatasetAccess()
    return dbAccess.GetNumberOfCompanies()


def GetDF():
    dbaccess = DatasetAccess()
    vestas = pd.read_sql(
        "select * from stock where identifier = 15611 ", dbaccess.conn.GetConnector()
    )
    return vestas


def GetSingleStockDF():
    dbAccess = DatasetAccess()
    comp = dbAccess.getNcompanies(2)
    return dbAccess.getStockDFFromCompany(comp, column="close")


def GetStocks(n):
    dbAccess = DatasetAccess()
    comp = dbAccess.getNcompanies(n)
    return dbAccess.getStockDFFromCompany(comp, column="close")


def GetStocksHourly(n, column="close"):
    dbAccess = DatasetAccess()
    comps = dbAccess.getNcompanies(n)
    result = []
    for comp in comps:
        result.append(
            get_data_for_datasetid(str(comp[0]), dbAccess.conn.GetConnector(), "h")[
                column
            ]
        )
    return result


def extractNumbers(numbers):
    result = []
    for number in numbers:
        result.append(number[0])
    return result


def GetCloseValue(indexes=slice(1)):
    dbAccess = DatasetAccess()
    print(dbAccess.getAllcompanies()[indexes])
    return extractNumbers(
        dbAccess.getStockFromCompany(dbAccess.getAllcompanies()[indexes], "close")[0]
    )


def PlotCloseValue(indexes=slice(1)):
    import matplotlib.pyplot as plt

    plt.plot(GetCloseValue(indexes))
    plt.show()


def GetSingleStockDF():
    dbAccess = DatasetAccess()
    comp = dbAccess.getNcompanies(2)
    return dbAccess.getStockDFFromCompany(comp)


def GetNStockDFs(N):
    dbAccess = DatasetAccess()
    comp = dbAccess.getNcompanies(N)
    return dbAccess.getStockDFFromCompany(comp)


def _get_dataset_ids(conn, where_clause):
    df = pd.read_sql_query(
        f"SELECT identifier, description from dataset WHERE {where_clause}",
        conn,
    )

    return df


def get_data_for_attribute(
    attribute_name,
    attribute_value,
    conn,
    interval,
    from_time="0001-01-01 00:00:00",
    to_time="9999-01-01 00:00:00",
):
    if isinstance(attribute_value, list):
        datasetids = _get_dataset_ids(
            conn, f"{attribute_name} in ({','.join(attribute_value)})"
        )
    else:
        raise Exception(
            f"{attribute_name} must be a list, not a {type(attribute_value)}"
        )

    dfs = []

    print("\n-----------------------")
    print("--- presenting data ---")
    print("-----------------------\n")
    print(f"{len(datasetids)} compaines were found. Including:")

    for i in range(len(datasetids)):
        datasetid = datasetids.iloc[i]["identifier"]
        description = datasetids.iloc[i]["description"]

        dfs.append(
            get_data_for_datasetid(datasetid, conn, interval, from_time, to_time)
        )
        print(f"  - {description} - shape: {dfs[-1].shape}")

    print("\n")
    return dfs


def get_data_for_datasetid(
    datasetid,
    conn,
    interval,
    from_time="0001-01-01 00:00:00",
    to_time="9999-01-01 00:00:00",
):
    df = pd.read_sql_query(
        f"SELECT time AS date, open, high, low, close, volume \
                            FROM stock WHERE identifier = {datasetid} AND time BETWEEN '{from_time}' AND '{to_time}' \
                            ORDER BY time ASC;",
        conn,
    )
    df.set_index(pd.DatetimeIndex(df["date"]), inplace=True)
    # df.drop(['date'], axis=1, inplace=True)

    df = preprocess.resample_data_to_interval(interval, df)

    # resample minutely to hourly

    return df


def get_primay_category(conn):
    df = pd.read_sql_query("SELECT DISTINCT primarycategory FROM dataset", conn)

    df_list = list(df["primarycategory"])
    df_list.remove("UNKNOWN")

    return df_list


def get_secondary_category(conn):
    df = pd.read_sql_query("SELECT DISTINCT secondarycategory FROM dataset", conn)

    df_list = list(df["secondarycategory"])
    df_list.remove("UNKNOWN")

    return df_list


def get_company_name(identifier, conn):
    return pd.read_sql_query(
        f"SELECT description from dataset where identifier = {identifier}", conn
    )


def get_companyid(conn):
    df = pd.read_sql_query("SELECT DISTINCT identifier FROM dataset", conn)

    return list(df["identifier"])


def get_connection():
    return pg.connect(
        database=settings.get_database(),
        user=settings.get_user(),
        password=settings.get_pasword(),
    )


def window1(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def FormatDataForLSTM(stocks, window_size):
    WindowedStocks = []
    for stock in stocks:
        WindowedStocks.append(window1(stock, window_size))
    result = []
    for window in WindowedStocks:
        for i in window:
            result.append(i)
    return result

def GenerateDataset(companies):
    training,targeting = None,None
    for company in companies:
        train,target = company
        if training is None:
            training = train
            targetign = target
        else:
            training = np.concatenate((training,train))
            targeting = np.concatenate((targetign,target))
    return (training,targeting)

def SingleCompany(Company, window_size, Output_size):
    closing_prices = [ x["close"].values for x in Company]
    open_prices = [ x["open"].values for x in Company]

    #print("closing prices: ", closing_prices)
    # print("closing prices shape: ", closing_prices[0].shape)
    close_formatedData = FormatDataForLSTM(closing_prices,window_size)
    open_formatedData = FormatDataForLSTM(open_prices,window_size)

    #print("formatedData: ", formatedData)
    # print("formatedData shape: ", len(close_formatedData[0]))
    close_formatedData = [np.array(x) for x in close_formatedData]
    open_formatedData = [np.array(x) for x in open_formatedData]
    #print("evenMoreFormated: ", evenMoreFormated)
    #print("evenMoreFormated shape: ", np.array(evenMoreFormated).shape)

    closingData = np.array(close_formatedData)
    openData = np.array(open_formatedData)

    closingData = (closingData - closingData.mean()) / closingData.std()

    # print("Fetching opening prices")
    # openData = np.array(
    #     db_access.getBigData("open", n_companies, n_datapoints, window_size)
    # )
    openData = (openData - openData.mean()) / openData.std()
    closing = closingData.reshape(closingData.shape[0], closingData.shape[1], 1)
    opens = openData.reshape(openData.shape[0], openData.shape[1], 1)
    data = torch.concat((torch.FloatTensor(closing), torch.FloatTensor(opens)), 2)

    train = np.array(
        [np.array(d[: window_size - Output_size]) for d in data]
    )  # (number of windows, points, n_class)
    target = np.array(
        [np.array(d[window_size - Output_size :]) for d in data]
    )  # (number of windows, points, n_class)
    return (train,target)

def getBigData(colum, n_company, n_datapoints, window_size):
    stocks = []
    NumberOfCompanies = GetNumberOfCompanies()[0][0]
    if NumberOfCompanies >= n_company:
        for stock in GetStocksHourly(n_company, colum):
            stocks.append(np.array(stock[:n_datapoints]).flatten())

        WindowedStocks = []
        for stock in stocks:
            WindowedStocks.append(window1(stock, window_size))

        result = []
        for window in WindowedStocks:
            for i in window:
                result.append(i)
    else:
        result = []
        print(
            "There are only {NumberOfCompanies} not {n_company} as requested".format(
                NumberOfCompanies=NumberOfCompanies, n_company=n_company
            )
        )
    return result
def SplitData(data, train_size):
    train = data[0][:int(data[0].shape[0]*train_size)]
    test = data[0][int(data[0].shape[0]*train_size):]
    target = data[1][:int(data[0].shape[0]*train_size)]
    test_target = data[1][int(data[0].shape[0]*train_size):]
    return ((train,target)), ((test,test_target))

if __name__ == "__main__":
    print(GetSingleStockDF())
