from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt


from utils.metrics import metric

convert_to_datetime = lambda x: datetime(year=2021, month=x[0], day=x[1], hour=x[3])

def plot_preds_with_date(trues, preds, data_set, idx, **kwargs):
    """
    Plots the true and predicted values for the given data set.

    Args:
        trues: The true values for the given data set.
        preds: The predicted values for the given data set.
        data_set: The data set to be plotted, used to find the dates
        idx: The index of the data set to be plotted.
        **kwargs: Additional arguments to be passed to the plot function.
    """
    get_dates = lambda x: data_set[x][-1][-len(preds):]
    df_eval = pd.DataFrame(index=pd.to_datetime([convert_to_datetime(d) for d in get_dates(idx)]))
    df_eval['trues'] = trues.reshape(-1)
    df_eval['preds'] = preds.reshape(-1)
    sns.set(style="darkgrid")
    df_eval.plot(**kwargs)

# Predict for any set of data
# load data
def predict_and_metrics(model, exp, flag):
    model = exp.model
    data_set, data_loader = exp._get_data(flag)

    model.eval()

    preds = []
    trues = []

    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(data_loader):
        pred, true = exp._process_one_batch(
            data_set, batch_x, batch_y, batch_x_mark, batch_y_mark)
        preds.append(pred.detach().cpu().numpy())
        trues.append(true.detach().cpu().numpy())
    
    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)
    metrics = [mae, mse, rmse, mape, mspe] = metric(preds, trues)

    return preds, trues, metrics

def plot_sample(exp, flag, idx, dates=False, **kwargs):
    """
    Plots the given sample.

    Args:
        sample: The sample to be plotted.
        idx: The index of the sample to be plotted.
        dates: Whether to plot the dates. Not implemented yet
    """
    exp.model.eval()
    args = exp.args

    data_obj, _ = exp._get_data(flag)
    x, y, x_mark, y_mark = data_obj[idx]

    
    x = torch.FloatTensor(x).unsqueeze(0)
    x_mark = torch.FloatTensor(x_mark).unsqueeze(0)
    y = torch.FloatTensor(y).unsqueeze(0)
    y_mark = torch.FloatTensor(y_mark).unsqueeze(0)
    y_hat, y_true = exp._process_one_batch(data_obj, x, y, x_mark, y_mark)
    y_hat = y_hat.detach().cpu().numpy().flatten()
    y_true = y_true.cpu().numpy().flatten()

    if dates:
        prev_timeenc = exp.args.embed 
        exp.args.embed = 0
        data_set, _ = exp._get_data(flag)
        exp.args.embed = prev_timeenc
        _, _, x_mark, y_mark = data_set[idx]
        dates = np.concatenate([x_mark, y_mark[-args.pred_len:]], axis=0)
        df_eval = pd.DataFrame(index=pd.to_datetime([convert_to_datetime(d) for d in dates]))
        df_eval['Input Sequence'] = np.nan
        df_eval['Predictions'] = np.nan
        df_eval['Ground truth'] = np.nan
        df_eval['Input Sequence'][:args.seq_len] = x[0,:,-1].detach().cpu().numpy()
        df_eval['Predictions'][-args.pred_len:] = y_hat
        df_eval['Ground truth'][-args.pred_len:] = y_true
        df_eval.plot(**kwargs)
    else:
        x_seq = np.arange(args.seq_len)
        x_preds = np.arange(args.seq_len, args.seq_len+args.pred_len)

        sns.set(style="darkgrid")
        fig, ax = plt.subplots(**kwargs)
        ax.plot(x_seq, x[0,:,-1] , label='Input sequence')
        ax.plot(x_preds, y_hat, label='Prediction')
        ax.plot(x_preds, y_true, label='GroundTruth')
        ax.legend()
        fig.show()
        
def stock_id_to_df(stock_id, pg_conn, agg, sample_freq):
    query = f'SELECT time AS date, open, high, low, volume, close \
                        FROM stock WHERE identifier = {stock_id} \
                        ORDER BY time ASC;'
    df = pd.read_sql(query, pg_conn)
    df.set_index(pd.DatetimeIndex(df['date']), inplace=True)
    # resample 
    df = df.resample(sample_freq).agg(agg).dropna()
    df.reset_index(inplace=True)
    return df