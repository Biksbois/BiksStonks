from unittest import result
import psycopg2 as pg
from utils.DatabaseConnection import DatabaseConnection
import utils.preprocess as preprocess
import utils.settings_utils as settings
import pandas as pd


class DatasetAccess:
    def __init__(self):
        self.conn = DatabaseConnection()

    def getAllcompanies(self):
        AllCompanies = self.conn.query("SELECT * FROM dataset")
        return AllCompanies

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


def get_data_for_datasetid(datasetid, conn, interval, time="0001-01-01 00:00:00"):
    df = pd.read_sql_query(
        f"SELECT time AS date, open, high, low, close, volume \
                            FROM stock WHERE identifier = {datasetid} AND time > '{time}' \
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

def get_company_name(identifier,conn):
    return pd.read_sql_query(f"SELECT description from dataset where identifier = {identifier}", conn)



def get_companyid(conn):
    df = pd.read_sql_query("SELECT DISTINCT identifier FROM dataset", conn)

    return list(df["identifier"])


def get_connection():
    return pg.connect(
        database=settings.get_database(),
        user=settings.get_user(),
        password=settings.get_pasword(),
    )


if __name__ == "__main__":
    print(GetSingleStockDF())
