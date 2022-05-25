from traceback import print_tb
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
import os 

from utils_in.tools import StandardScaler
from utils.DatasetAccess import get_sentiment, get_sentiments

class IWATA_Classification_Sampler(Dataset):
    def __init__(self, root_path, seq_len, pred_len, num_samples, flag, 
                 split=[55,10,24], S_N=16, Q_N=1, series_len=100, seed=0):
        assert sum(split) == 89 # only 89 datasets 
        self.num_samples = num_samples
        self.split = split
        self.root_path = root_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.S_N = S_N # support set size
        self.Q_N = Q_N # query set size
        self.series_len = series_len
        self.data_list = os.listdir(root_path)
        self.flag = flag
        random.seed(seed) # important to ensure that train_tasks, val_tasks, target_tasks are the same
        self.train_tasks, self.val_tasks, self.target_tasks = self.__split_data(self.data_list)
        self.support_tasks, self.query_tasks = self.__read_data()
        print(len(self.train_tasks), len(self.val_tasks), len(self.target_tasks))
    
    def __split_data(self, data_list):
        train_tasks, val_tasks, target_tasks = [], [], []
        data_bowl = self.data_list.copy()
        # sample train_tasks
        for i in range(self.split[0]):
            train_tasks.append(random.choice(data_bowl))
            data_bowl.remove(train_tasks[-1])
        # sample val_tasks
        for i in range(self.split[1]):
            val_tasks.append(random.choice(data_bowl))
            data_bowl.remove(val_tasks[-1])
        target_tasks = data_bowl
        return train_tasks, val_tasks, target_tasks

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx_end = self.series_len - self.seq_len - self.pred_len
        # sample indexes for train_tasks
        train_idx = random.sample(range(idx_end), self.S_N)
        # sample train_tasks for support set
        sup_tasks = random.sample(range(self.split[0]), self.S_N)
        if self.flag == 'test':
            # sample indexes for target tasks
            target_idx = random.sample(range(idx_end), self.Q_N)
            # sample target tasks for query set
            query_tasks = random.sample(range(self.split[2]), self.Q_N)
        else: 
            # sample target tasks from train_tasks
            query_tasks = random.sample(range(self.split[0]), self.Q_N)
            # sample indexes for target tasks
            target_idx = random.sample(range(idx_end), self.Q_N)

        sup_x, _ = self.__tasks_to_arrays(train_idx, support_set=True)
        query_x, query_y = self.__tasks_to_arrays(target_idx, support_set=self.flag == 'TRAIN') # if 'TEST' use query_tasks 
        # univariate add extra dimension
        sup_x = sup_x[...,None]
        query_x = query_x[...,None]
        

        return sup_x, query_x, query_y
        

    def __tasks_to_arrays(self, idxs, support_set=True):
        x, y = [], []
        sample_set = self.support_tasks if support_set else self.query_tasks
        
        for idx in idxs:
            task = random.choice(self.support_tasks)
            timeseries = task.sample(1).values[0][0].to_numpy() # each task contains multiple timeseries (classification data)
            x.append(timeseries[idx:idx+self.seq_len])
            y.append(timeseries[idx+self.seq_len:idx+self.seq_len+self.pred_len])
        
        return np.array(x), np.array(y).reshape(-1)[0]
            
    
    def __read_data(self):
        from sktime.datasets import load_from_tsfile_to_dataframe
        support_tasks = []
        query_tasks = None
        for data in self.train_tasks:
            root_path = 'preprocessed_iwata_ds/'
            path = os.path.join(root_path, data, data + '_TRAIN.ts')
            x, y = load_from_tsfile_to_dataframe(path)
            support_tasks.append(x)

        if self.flag == 'test':
            query_tasks = []
            for data in self.target_tasks:
                root_path = 'preprocessed_iwata_ds/'
                path = os.path.join(root_path, data, data + '_TRAIN.ts')
                x, y = load_from_tsfile_to_dataframe(path)
                query_tasks.append(x)

        return support_tasks, query_tasks         


class Iwata_Dataset_DB_Stock(Dataset):
    def __init__(self, conn, S_N, Q_N, sup_stck_ids=None, q_stck_ids=None, flag='train', size=None, 
                 features='M', data_path='ETTh1.csv', target='close', scale=True, inverse=False,
                 companyID=None, hasTargetDF=False, targetDF=None, seed=0, freq='1T', columns=None):
        # size [seq_len, label_len, pred_len]
        # info
        assert len(size)==3
        # init
        assert flag in ['train', 'test']
        self.flag = flag
        self.type_map = {'train':0, 'test':1}
        self.set_type = self.type_map[flag]

        # seed 
        random.seed(seed)
        np.random.seed(seed) # also works for pandas
        
        self.agg = {'open': 'first',
        'high': 'max', 
        'low': 'min', 
        'close': 'last',
        'volume': 'sum'}
        self.freq = freq
        self.S_N = S_N
        self.Q_N = Q_N
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.companyID = companyID
        self.hasTargetDF = hasTargetDF
        self.targetDF = targetDF
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2] 
        self.columns = columns
        assert self.pred_len == 1 # for now 

        self.__read_data__(conn)

    def __stock_id_to_df__(self, stock_id, pg_conn, sample_freq, sentiment = None, use_sentiment = False):
        query = f'SELECT time AS date, open, high, low, volume, close \
                            FROM stock WHERE identifier = {stock_id} \
                            ORDER BY time ASC;'
        df = pd.read_sql(query, pg_conn)
        df.set_index(pd.DatetimeIndex(df['date']), inplace=True)
        # resample 
        df = df.resample(sample_freq).agg(self.agg).dropna()
        df.reset_index(inplace=True)
        if use_sentiment:
            if isinstance(sentiment, list):
                for sent in sentiment: # list of dfs for each cat?
                    df = df.merge(sent, how="left", on="date")
                    df = df.fillna(method="ffill")
                    df = df.fillna(1)
                df.reset_index(inplace=True)
            else:
                df = df.merge(sentiment, how="left", on="date")
                df = df.fillna(method="ffill")
                df = df.fillna(1)
                df.reset_index(inplace=True)

        df = df.dropna()
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
        if self.companyID:
            pass 
            Q_stock_meta = pd.read_sql(f'SELECT dataset.identifier as id, description as name, count(close) as points\
                  FROM stock JOIN dataset ON stock.identifier=dataset.identifier\
                  WHERE stock.identifier = {int(self.companyID)}\
                  GROUP BY dataset.identifier, dataset.description', conn)
        else:
            Q_stock_meta = stock_meta.sample(n=self.Q_N)

        # read data
        '''
        df.columns: ['date', ...(other features), target feature]
        '''
        sentiment = None
        use_sentiment = False
        if 'compound' in self.columns: #  <----------------------------- here oh I see, hmm
            sentiment = get_sentiment(self.targetDF['date'].iloc[0],self.targetDF['date'].iloc[-1],conn, self.freq)
            use_sentiment = True
            
        elif 'Markedsberetninger' in self.columns: # read here all sentiment categories 
            # assume it is always all categories  
            sentiment = get_sentiments(str(self.targetDF['date'].iloc[0]),str(self.targetDF['date'].iloc[-1]),conn, 
                                           self.freq, category=None)
            use_sentiment = True                                    

        self.S_dfs = []
        self.Q_dfs = []
        for s_id in S_stock_meta.id:
            self.S_dfs.append(self.__stock_id_to_df__(s_id, conn, self.freq, sentiment,use_sentiment))
            
        for q_id in Q_stock_meta.id:
            if self.hasTargetDF: 
                self.Q_dfs.append(self.targetDF)
            else:
                self.Q_dfs.append(self.__stock_id_to_df__(q_id, conn, self.freq,sentiment,use_sentiment))
        
        # only tuned for Q_N = 1
        cols = list(self.Q_dfs[0].columns); cols.remove(self.target); cols.remove('date')
        df_raw = self.Q_dfs[0][['date']+cols+[self.target]]
        
        # not used we get it from outside 
        if not self.columns:
            if self.features=='M' or self.features=='MS': 
                cols_data = df_raw.columns[1:]
            elif self.features=='S':
                cols_data = [self.target]
        else:
            cols_data = self.columns[1:] + [self.target]

        

        self.S_dfs_x = []
        self.Q_dfs_x = []
        if self.scale:
            self.s_scalers = [StandardScaler() for i in range(self.S_N)]
            self.q_scalers = [StandardScaler() for i in range(self.Q_N)]
            for i in range(self.S_N):
                b1, b2 = self.__compute_borders(self.S_dfs[i], flag='train')
                train_data = self.S_dfs[i][cols_data][b1:b2]
                self.s_scalers[i].fit(train_data)
                
                
                self.S_dfs_x.append(self.s_scalers[i].transform(self.S_dfs[i][cols_data][b1:b2]).values)
                
                # if np.isnan(np.min(self.S_dfs_x[-1])):
                #     print("CORRUPT")
                #     print(self.S_dfs_x[-1])
            for i in range(self.Q_N):
                b1, b2 = self.__compute_borders(self.Q_dfs[i], flag=self.flag)
                train_data = self.Q_dfs[i][cols_data][b1:b2]
                self.q_scalers[i].fit(train_data)
                self.Q_dfs_x.append(self.q_scalers[i].transform(self.Q_dfs[i][cols_data][b1:b2]).values)
                
                n = len(self.Q_dfs[i][cols_data][b1:b2])
                
        else:
            for i in range(self.S_N):
                self.S_dfs_x.append(self.S_dfs[i][cols_data][b1:b2].values)
            for i in range(self.Q_N):
                self.Q_dfs_x.append(self.Q_dfs[i][cols_data][b1:b2].values)

        # exit()
        
        
        
        
        
        self.data_x = self.Q_dfs_x[0]
        self.data_y = self.Q_dfs_x[0][:, -1]
    
    def __compute_borders(self, df, flag='train'):
        num_train = int(len(df)*0.7)
        num_test = len(df) - num_train
        border1s = [0, num_train-self.seq_len]
        border2s = [num_train, len(df)]
        border1 = border1s[self.type_map[flag]]
        border2 = border2s[self.type_map[flag]]
        return border1, border2

    def __getitem__(self, index):    
        # check date for all support set is < query index data
        # query_date = self.Q_dfs[0].iloc[index]['date']
        # for i in range(self.S_N):
        #     s_date = self.S_dfs[i].iloc[0]['date']
        #     # check support set date - 

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len        
        Q_dfs_x = self.Q_dfs_x[0]
        q_seq_x = Q_dfs_x[s_begin:s_end]
        if self.features == 'MS':
            q_seq_y = Q_dfs_x[r_begin:r_end][-1][-1] # last value of the target
        else:
            q_seq_y = Q_dfs_x[0][r_begin:r_end][-1]
        
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