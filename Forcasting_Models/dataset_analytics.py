import pandas as pd 
import psycopg2 as pg
import utils.settings_utils as settings
import warnings
import matplotlib.pyplot as plt
from utils.DatabaseConnection import DatabaseConnection
import utils.DatasetAccess as db_access
warnings.filterwarnings("ignore")

connection = db_access.get_connection()


def get_data(connection):
    df = pd.read_sql_query(f"SELECT time AS date, open, high, low, close, volume \
                     FROM stock;", connection)
    return df

def get_stats(df):
    return df.describe()

def get_correlation(df):
    return df.corr()

def get_histogram(df):
    df.hist(bins=50, figsize=(20,15))
    plt.show()

def get_boxplot(df):
    df.boxplot(figsize=(20,15))
    plt.show()

def get_scatterplot(df):
    df.plot.scatter(x='Close', y='Volume')
    plt.show()

def get_stock(connection):
    df = pd.read_sql_query(f"Select time as date, close, high, low, open, volume from stock;", connection)
    print(df.head(10))

get_stock(connection)
# print("Getting data...")
# data = get_data(connection)
# print("Data retrieved.")
# print("Getting stats...")
# stats = get_stats(data)
# print("Stats retrieved.")
# print("Getting correlation...")
# correlation = get_correlation(data)
# print("Correlation retrieved.")

# # write stats and correlation to csv
# stats.to_csv('stats.csv')
# correlation.to_csv('correlation.csv')
# # histogram = get_histogram(data)
# # boxplot = get_boxplot(data)
# # scatterplot = get_scatterplot(data)

