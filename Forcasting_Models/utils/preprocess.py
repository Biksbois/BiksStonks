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

def rename_dataset_columns(df_d, column):
    df_d["date"] = pd.to_datetime(df_d['date'])
    df_d=df_d.rename(columns={"date":"ds", column:"y"})
    df_d.reset_index(inplace=True)
    df_d.drop(['index'], axis=1, inplace=True)
    return df_d

def get_split_data(df, col_name="ds"):
    split_date = df[col_name].iloc[-1] - ((df[col_name].iloc[-1] - df[col_name].iloc[0]) * 0.3)
    training = df[df[col_name] <= split_date]# data[:split_date].iloc[:-1]
    testing = df[df[col_name] > split_date]
    return training, testing

def std_on_column(df, col_name):
    return df[col_name].std()

def avg_on_column(df, col_name):
    return df[col_name].mean()

def min_on_column(df, col_name):
    return df[col_name].min()

def max_on_column(df, col_name):
    return df[col_name].max()