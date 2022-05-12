from pickle import TRUE
from pydoc import describe
from unittest import result
from webbrowser import get
import psycopg2 as pg
from utils.DatabaseConnection import DatabaseConnection
import utils.preprocess as preprocess
import utils.settings_utils as settings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from utils.data_obj import DataObj
import datetime

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


def _upsert_model(model_name, model_desc, cur):
    cur.callproc(
        "upsert_model",
        (model_name, model_desc),
    )

    model_id = cur.fetchone()[0]

    return model_id


def _upsert_score(
    model_id,
    executed_time,
    mae,
    mse,
    r_squared,
    data_from,
    data_to,
    time_unit,
    forecasted_company,
    metadata,
    used_sentiment,
    used_companies,
    columns,
    cur,
):
    print(type(model_id), model_id)
    print(type(executed_time), executed_time)
    print(type(float(mae)), float(mae))
    print(type(float(mse)), float(mse))
    print(type(float(r_squared)), float(r_squared))
    print(type(data_from), data_from)
    print(type(data_to), data_to)
    print(type(time_unit), time_unit)
    print(type(forecasted_company), forecasted_company)
    print("---")
    print(metadata)
    print(type(metadata))
    print("---")
    print(type(json.dumps(metadata)), json.dumps(metadata))
    print(type(used_sentiment), used_sentiment)
    print(type(used_companies), used_companies)
    print(type(columns), columns)

    cur.callproc(
        "upsert_score",
        (
            model_id,
            executed_time,
            float(mae),
            float(mse),
            float(r_squared),
            data_from,
            data_to,
            time_unit,
            forecasted_company,
            json.dumps(metadata),
            used_sentiment,
            used_companies,
            columns,
        ),
    )

    score_id = cur.fetchone()[0]

    return score_id


def _upsert_graph(score_id, forecasts, cur):
    for _, row in forecasts.iterrows():
        cur.callproc(
            "upsert_graph",
            (
                score_id,
                row["y"] if not pd.isnull(row["y"]) else None,
                row["y_hat"] if not pd.isnull(row["y_hat"]) else None,
                row["time"] if "time" in forecasts.columns and not pd.isnull(row["time"]) else None,
            ),
        )


def upsert_exp_data(
    model_name,
    model_desc,
    mae,
    mse,
    r_squared,
    data_from,
    data_to,
    time_unit,
    forecasted_company,
    metadata,
    used_sentiment,
    used_companies,
    columns,
    forecasts,
    connection,
):
    with connection as conn:
        cur = conn.cursor()
        executed_time = datetime.datetime.now()
        print("upserting company")
        model_id = _upsert_model(model_name, model_desc, cur)
        print(f"model id: {model_id}")
        score_id = _upsert_score(
            model_id,
            executed_time,
            mae,
            mse,
            r_squared,
            data_from,
            data_to,
            time_unit,
            forecasted_company,
            metadata,
            "True" if used_sentiment in ['all', 'one'] else used_sentiment,
            used_companies,
            columns,
            cur,
        )
        _upsert_graph(score_id, forecasts, cur)
        cur.close()


def get_data_for_attribute(
    attribute_name,
    attribute_value,
    conn,
    interval,
    from_time="0001-01-01 00:00:00",
    to_time="9999-01-01 00:00:00",
    use_sentiment=True,
    sentiment_cat=None,
):

    if isinstance(attribute_value, list):
        datasetids = _get_dataset_ids(
            conn, f"{attribute_name} in ({','.join(attribute_value)})"
        )
    else:
        raise Exception(
            f"{attribute_name} must be a list, not a {type(attribute_value)}"
        )

    company_data = []
    if use_sentiment in [True, "True", "true", "TRUE"] and sentiment_cat is None:
        sentiment = get_sentiment(from_time, to_time, conn, interval)
    else:
        sentiments = get_sentiments(from_time, to_time, conn, interval)


    for i in range(len(datasetids)):
        datasetid = datasetids.iloc[i]["identifier"]
        description = datasetids.iloc[i]["description"]

        data = get_data_for_datasetid(datasetid , conn, interval, from_time, to_time)
        company_data.append(DataObj(data, datasetid, description))

    for company in company_data:
        if use_sentiment in [True, "True", "true", "TRUE"] and sentiment_cat is None:
            company.data = company.data.merge(sentiment, how="left", on="date")
            company.data = company.data.fillna(method="ffill")
            company.data = company.data.fillna(0.0)

        else:
            for cat in sentiments:
                date = company.data['date'].tolist()[0]
                PastDates = [x for x in cat['date'].tolist() if x < date]
                if len(PastDates) > 0:
                    nearest = get_nearest(PastDates, date)
                    value = cat[cat['date'] == nearest][cat.columns[1]].tolist()[0]
                    row = [date, value]
                    cat.loc[-1] = row
                company.data = company.data.merge(cat, how="left", on="date")
                company.data = company.data.fillna(method="ffill")
                company.data = company.data.fillna(0.0)
    return company_data

def get_nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def get_graph_for_id(id, conn):
    df = pd.read_sql_query(
        f"SELECT time, y, y_hat FROM graph where score_id = {id}",
        conn,
    )

    return df


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

def get_sentiment(from_time, to_time, conn, interval):
    df = pd.read_sql_query(f"select release_date, compound from sentiment where release_date between '{from_time}' and '{to_time}'", conn)
    df["release_date"] = pd.to_datetime(df['release_date'])
    df=df.rename(columns={"release_date":"date"})
    datetime_index = pd.DatetimeIndex(df.date)
    df=df.set_index(datetime_index)
    df = preprocess.resample_data_to_interval(interval, df, {"compound": "mean"})
    return df

def get_sentiments(from_time, to_time, conn, interval, category = None, cutoff = 2000):
    if category is None:
        category = get_sentiment_categories(conn)
    category = category["category"].tolist()
        
    result = []
    for cat in category:
        if get_sentiment_count(conn, cat) < cutoff:
            print(f"cutoff reached: {cat}")
            continue
        result.append(get_sentiment_from_cat(from_time,to_time,conn,interval,cat).rename(columns={"release_date":"date", "compound":cat}))
    return result


def get_sentiment_categories(conn):
    df = pd.read_sql_query("select category from sentiment_dataset", conn)
    return df

def get_sentiment_count(conn, category):
    df = pd.read_sql_query(f"select count(*) from sentiment where sentiment.datasetid = (select id from sentiment_dataset where category = '{category}')", conn)
    return df["count"].iloc[0]

def get_sentiment_from_cat(from_time, to_time,conn, interval, category):
    from_time = datetime.datetime.strptime(from_time, "%Y-%m-%d %H:%M:%S")- datetime.timedelta(days=365)
    from_time = from_time.strftime("%Y-%m-%d %H:%M:%S")
    df = pd.read_sql_query(f"select release_date, compound from sentiment where sentiment.datasetid = (select id from sentiment_dataset where category = '{category}')", conn)
    df["release_date"] = pd.to_datetime(df['release_date'])
    df=df.rename(columns={"release_date":"date"})
    datetime_index = pd.DatetimeIndex(df.date)
    df=df.set_index(datetime_index)
    df = preprocess.resample_data_to_interval(interval, df, {"compound": "mean"})
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
    training, targeting = None, None
    for company in companies:
        train, target = company
        if training is None:
            training = train
            targeting = target
        else:
            training = np.concatenate((training, train))
            targeting = np.concatenate((targeting, target))
    return (training, targeting)


def GenerateDatasets(companies):
    training, targeting, testing, test_targeting = None, None, None, None
    for company in companies:
        tr, te = SplitData(company, 0.8)
        train, target = tr
        test, test_target = te
        if training is None:
            training = train
            targeting = target
            testing = test
            test_targeting = test_target
        else:
            training = np.concatenate((training, train))
            targeting = np.concatenate((targeting, target))
            testing = np.concatenate((testing, test))
            test_targeting = np.concatenate((test_targeting, test_target))
    return ((training, targeting), (testing, test_targeting))


def SingleCompany(Company, window_size, Output_size, columns):
    column_data = []
    for column in columns:
        column_data.append(
            ProccessData([x[column].values for x in Company], window_size)
        )
    data = None
    for d in column_data:
        if data is None:
            data = d
        else:
            data = torch.concat((torch.FloatTensor(data), torch.FloatTensor(d)), 2)
    train = np.array(
        [np.array(d[: window_size - Output_size]) for d in data]
    )  # (number of windows, points, n_class)
    target = np.array(
        [np.array(d[window_size - Output_size :]) for d in data]
    )  # (number of windows, points, n_class)
    return (train, target)


def ProccessData(values, window_size):
    valuesFormated = FormatDataForLSTM(values, window_size)
    valuesFormated = [np.array(x) for x in valuesFormated]
    ValuesData = np.array(valuesFormated)
    ValuesData = (ValuesData - ValuesData.mean()) / ValuesData.std()
    return ValuesData.reshape(ValuesData.shape[0], ValuesData.shape[1], 1)


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
    train = data[0][: int(data[0].shape[0] * train_size)]
    test = data[0][int(data[0].shape[0] * train_size) :]
    target = data[1][: int(data[0].shape[0] * train_size)]
    test_target = data[1][int(data[0].shape[0] * train_size) :]
    return ((train, target)), ((test, test_target))


if __name__ == "__main__":
    print(GetSingleStockDF())
