import pandas as pd



def resample_data_to_interval(interval, df):
    df_d = df.resample(interval).agg(
            {'open': 'first',
            'high': 'max', 
            'low': 'min', 
            'close': 'last', 
            'volume': 'sum'}).dropna()
    df_d.reset_index(inplace=True)
    # df_d = df_d[["date","close"]]

    # df_d["date"] = pd.to_datetime(df_d['date'])
    return df_d

def rename_dataset_columns(df_d):
    df_d["date"] = pd.to_datetime(df_d['date'])
    df_d=df_d.rename(columns={"date":"ds", "close":"y"})
    df_d.reset_index(inplace=True)
    df_d.drop(['index'], axis=1, inplace=True)
    return df_d

def std_on_column(df, col_name):
    return df[col_name].std()

def avg_on_column(df, col_name):
    return df[col_name].mean()

def min_on_column(df, col_name):
    return df[col_name].min()

def max_on_column(df, col_name):
    return df[col_name].max()