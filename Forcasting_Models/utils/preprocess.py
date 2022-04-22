import pandas as pd

def add_to_parameters(arguments, parameters, is_fb_or_arima=False):
    if arguments.primarycategory:
        parameters["primarycategory"] = arguments.primarycategory
    elif arguments.secondarycategory:
        parameters["secondarycategory"] = arguments.secondarycategory
    else:
        parameters["companyid"] = arguments.companyid

    if arguments.limit:
        parameters["limit"] = arguments.limit
    
    if is_fb_or_arima:
        parameters["columns"] = arguments.columns[0]
    else:
        parameters["columns"] = arguments.columns



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
    split_date = df[col_name].iloc[-1] - ((df[col_name].iloc[-1] - df[col_name].iloc[0]) * 0.2)
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

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean