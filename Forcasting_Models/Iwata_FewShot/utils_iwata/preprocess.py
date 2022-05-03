import os
from sktime.datasets import load_from_tsfile_to_dataframe


def preprocess_ts(filename, x, y, dataset_name, seq_len=100):
    with open(filename, 'w') as f:
        f.write(f'@problemName {dataset_name}\n')
        f.write(f'@timeStamps false\n')
        f.write(f'@missing false\n')
        f.write(f'@univariate true\n')
        f.write(f'@equalLength true\n')
        f.write(f'@seriesLength 100\n')
        f.write(f'@classLabel true 0 1 2 3 4 5 6 7 8 9\n')
        f.write(f'@data\n')
        for row in x.itertuples():
            f.writelines(','.join(map(str, row.dim_0.to_list()[:seq_len])))
            f.write(f':{y[row.Index]}\n')


def preprocess_iwata(data_list_path='data_list.csv', 
                     out_ds_path='preprocessed_iwata_ds', 
                     original_ds_path='Univariate_ts'):
    with open(data_list_path, 'r') as f:
        data_list = f.read().splitlines()
    # makedir if not exists 
    root_path = out_ds_path
    DATA_PATH = original_ds_path
    for dataset_name in data_list:
        new_dir_path = os.path.join(root_path, dataset_name)
        if not os.path.exists(new_dir_path):
            os.makedirs(f"{new_dir_path}")
        for flag in ['TRAIN', 'TEST']:
            x, y = load_from_tsfile_to_dataframe(
                os.path.join(DATA_PATH, f"{dataset_name}/{dataset_name}_{flag}.ts")
                )
            
            filename = os.path.join(new_dir_path, f"{dataset_name}_{flag}.ts") 
            x = normalize_x(x)
            preprocess_ts(filename, x, y, dataset_name, seq_len=100)

def normalize_x(x):
    for i, row in enumerate(x['dim_0']):
        x['dim_0'][i] = (row - row.mean())/row.std()
    return x