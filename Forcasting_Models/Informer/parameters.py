import torch

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


informer_params = dotdict()

informer_params.model = 'informer' # model of experiment, options: [informer, informerstack, informerlight(TBD)]

informer_params.data = 'custom' # data
informer_params.root_path = 'data' # root path of data file
informer_params.data_path = 'AAPL_h.csv' # 'dbank_h.csv' # data file

informer_params.use_df = True # use dataframe
# informer_params.df = df # dataframe

informer_params.features = 'MS' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
informer_params.target = 'close' # target feature in S or MS task
informer_params.freq = 'h' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
informer_params.checkpoints = './informer_checkpoints' # location of model checkpoints



informer_params.seq_len = 60 # input sequence length of Informer encoder
informer_params.label_len = 48 # start token length of Informer decoder
informer_params.pred_len = 30 # prediction sequence length
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

informer_params.enc_in = 5 # encoder input size # ohlc + volume + [trade_count, vwap]
informer_params.dec_in = 5 # decoder input size # ohlc + volume + [trade_count, vwap]
informer_params.c_out = 1 # output size # 1 univariate prediction for close price
informer_params.factor = 5 # probsparse attn factor
informer_params.d_model = 512 # dimension of model
informer_params.n_heads = 8 # num of heads
informer_params.e_layers = 2 # num of encoder layers
informer_params.d_layers = 1 # num of decoder layers
informer_params.d_ff = 2048 # dimension of fcn in model
informer_params.dropout = 0.05 # dropout
informer_params.attn = 'prob' # attention used in encoder, options:[prob, full]
informer_params.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
informer_params.activation = 'gelu' # activation
informer_params.distil = True # whether to use distilling in encoder
informer_params.output_attention = False # whether to output attention in ecoder
informer_params.mix = True
informer_params.padding = 0
informer_params.freq = 'h'
informer_params.detail_freq = informer_params.freq
informer_params.inverse = False

informer_params.batch_size = 32 
informer_params.learning_rate = 0.0001
informer_params.loss = 'mse'
informer_params.lradj = 'type1'
informer_params.use_amp = False # whether to use automatic mixed precision training

informer_params.num_workers = 0
informer_params.itr = 1
informer_params.train_epochs = 10
informer_params.patience = 100
informer_params.des = 'exp'

informer_params.use_gpu = True if torch.cuda.is_available() else False
informer_params.gpu = 0

informer_params.use_multi_gpu = False
informer_params.devices = '0,1,2,3'