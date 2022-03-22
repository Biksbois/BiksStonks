from datetime import datetime
import pandas as pd
import seaborn as sns

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
    convert_to_datetime = lambda x: datetime(year=2021, month=x[0], day=x[1], hour=x[3])
    df_eval = pd.DataFrame(index=pd.to_datetime([convert_to_datetime(d) for d in get_dates(idx)]))
    df_eval['trues'] = trues.reshape(-1)
    df_eval['preds'] = preds.reshape(-1)
    sns.set(style="darkgrid")
    df_eval.plot(**kwargs)

        