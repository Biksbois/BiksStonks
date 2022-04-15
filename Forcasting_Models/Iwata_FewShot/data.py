from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from utils import StandardScaler


class Iwata_Dataset_DB_Stock(Dataset):
    def __init__(self, conn, S_N, Q_N, sup_stck_ids=None, q_stck_ids=None, flag='train', size=None, 
                 features='M', data_path='ETTh1.csv', 
                 target='close', scale=True, inverse=False,):
        # size [seq_len, label_len, pred_len]
        # info
        assert len(size)==3
        # init
        assert flag in ['train', 'test']
        type_map = {'train':0, 'test':1}
        self.set_type = type_map[flag]
        
        self.agg = {'open': 'first',
        'high': 'max', 
        'low': 'min', 
        'close': 'last',
        'volume': 'sum'}
        self.freq = '1T'
        self.S_N = S_N
        self.Q_N = Q_N
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2] 
        assert self.pred_len == 1 # for now 

        self.__read_data__(conn)

    def __stock_id_to_df__(self, stock_id, pg_conn, sample_freq):
        query = f'SELECT time AS date, open, high, low, volume, close \
                            FROM stock WHERE identifier = {stock_id} \
                            ORDER BY time ASC;'
        df = pd.read_sql(query, pg_conn)
        df.set_index(pd.DatetimeIndex(df['date']), inplace=True)
        # resample 
        df = df.resample(sample_freq).agg(self.agg).dropna()
        df.reset_index(inplace=True)
        return df

    def __read_data__(self, conn):
        """Reads Q_N df from the db as the query set, and S_N df from the db as the support set.
           - Only works for random selection of stocks currently."""

        # sample stock ids 
        stock_meta = pd.read_sql('SELECT dataset.identifier as id, description as name, count(close) as points\
                                    FROM stock JOIN dataset ON stock.identifier=dataset.identifier\
                                    GROUP BY dataset.identifier, dataset.description\
                                    HAVING COUNT(close) > 10000;', conn)
        S_stock_meta = stock_meta.sample(n=self.S_N)
        Q_stock_meta = stock_meta.sample(n=self.Q_N)

        # read data
        '''
        df.columns: ['date', ...(other features), target feature]
        '''
        self.S_dfs = []
        self.Q_dfs = []
        for s_id in S_stock_meta.id:
            self.S_dfs.append(self.__stock_id_to_df__(s_id, conn, self.freq))

        for q_id in Q_stock_meta.id:
            self.Q_dfs.append(self.__stock_id_to_df__(q_id, conn, self.freq))
        
        # only tuned for Q_N = 1
        cols = list(self.Q_dfs[0].columns); cols.remove(self.target); cols.remove('date')
        df_raw = self.Q_dfs[0][['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = len(df_raw) - num_train
        border1s = [0, num_train-self.seq_len]
        border2s = [num_train, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
        elif self.features=='S':
            cols_data = [self.target]

        self.S_dfs_x = []
        self.Q_dfs_x = []
        if self.scale:
            self.s_scalers = [StandardScaler() for i in range(self.S_N)]
            self.q_scalers = [StandardScaler() for i in range(self.Q_N)]
            for i in range(self.S_N):
                train_data = self.S_dfs[i][cols_data][border1s[0]:border2s[0]]
                self.s_scalers[i].fit(train_data)
                self.S_dfs_x.append(self.s_scalers[i].transform(self.S_dfs[i][cols_data][border1:border2]).values)
            for i in range(self.Q_N):
                train_data = self.Q_dfs[i][cols_data][border1s[0]:border2s[0]]
                self.q_scalers[i].fit(train_data)
                self.Q_dfs_x.append(self.q_scalers[i].transform(self.Q_dfs[i][cols_data][border1:border2]).values)
        else:
            for i in range(self.S_N):
                self.S_dfs_x.append(self.S_dfs[i][cols_data][border1:border2].values)
            for i in range(self.Q_N):
                self.Q_dfs_x.append(self.Q_dfs[i][cols_data][border1:border2].values)
        
        self.data_x = self.Q_dfs_x[0]
        self.data_y = self.Q_dfs_x[0][:, -1]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len        

        # check date for all support set is < query index data
        # query_date = self.Q_dfs[0].iloc[index]['date']
        # for i in range(self.S_N):
        #     s_date = self.S_dfs[i].iloc[0]['date']
        #     # check support set date - 

        q_seq_x = self.Q_dfs_x[0][s_begin:s_end]
        if self.features == 'MS':
            q_seq_y = self.Q_dfs_x[0][r_begin:r_end][-1][-1] # last value of the target
        else:
            q_seq_y = self.Q_dfs_x[0][r_begin:r_end][-1]
        
        # randomly sample support sequence for each support timeseries 
        # (only works for Q_N = 1)
        s_seq_x = []
        for i in range(self.S_N):
            end = len(self.S_dfs_x[i]) - self.label_len - self.pred_len
            s_begin = np.random.randint(0, end)
            s_end = s_begin + self.seq_len
            r_begin = s_end
            r_end = r_begin + self.pred_len
            s_seq_x.append(self.S_dfs_x[i][s_begin:s_end][:-1])

        s_seq_x = np.array(s_seq_x)
            
        return s_seq_x, q_seq_x, q_seq_y
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)