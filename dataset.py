import pandas as pd
import numpy as np
import torch
#import scilkit-learn standard scaler


def candle_return_logs(data_path):
    
    df = pd.read_csv(data_path)

    #Logarithmic returns
    df['open_'] = np.log(df['Open']/df['Close'].shift(1))
    df['high_'] = np.log(df['High']/df['Open'])
    df['low_'] = np.log(df['Low']/df['Open'])
    df['close_'] = np.log(df['Close']/df['Open'])

    df.fillna(0, inplace=True)
    
    #calculate EMA200 and EMA1440    
    df['ema_200'] = np.log(df['Close']/df['Close'].ewm(span=200, adjust=False).mean())
    df['ema_1440'] = np.log(df['Close']/df['Close'].ewm(span=1440, adjust=False).mean())
    
    return torch.tensor(df[['open_', 'high_', 'low_', 'close_', 'ema_200', 'ema_1440']].values, dtype=torch.float32)

#function to scale the data
def scale_data(data: torch.Tensor):
    data_mean = data.mean(dim=0, keepdim=True)
    data_std = data.std(dim=0, keepdim=True)
    return (data - data_mean) / data_std, data_mean, data_std

#function to unscale the data
def unscale_data(data: torch.Tensor, data_mean, data_std):
    return data * data_std + data_mean



class BatchIterator:
    def __init__(self, batch_size, num_batches, inputs, targets, shuffle=True):
        self.batch_size = batch_size
        self.num_batches = num_batches - 1
        self.inputs = inputs
        self.targets = targets
        self.shuffle = shuffle
        self.dataset_len = batch_size * num_batches

    def __iter__(self):
        self.current_batch = 0
        if self.shuffle:
            self.perm = torch.randperm(self.inputs.shape[0])[:self.dataset_len]
        else:
            self.perm = torch.arange(self.inputs.shape[0])[:self.dataset_len]
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        s = self.current_batch * self.batch_size
        ids = self.perm[s : s + self.batch_size]
        batch_inputs = self.inputs[ids]
        batch_targets = self.targets[ids]
        self.current_batch += 1

        return self.current_batch, batch_inputs, batch_targets
    

def prepare_data(data_path, seq_len, batch_size, shuffle=False):
    
    data = candle_return_logs(data_path)
    #scale the data
    data, data_mean, data_std = scale_data(data)

    data = data.unfold(0, seq_len, 1).permute(0, 2, 1)

    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    train_inputs = train_data[:, :-1]
    train_targets = train_data[:, 1:, :4]
    test_inputs = test_data[:, :-1]
    test_targets = test_data[:, 1:, :4]

    train_iter = BatchIterator(batch_size, len(train_inputs) // batch_size, train_inputs, train_targets, shuffle)
    test_iter = BatchIterator(batch_size, len(test_inputs) // batch_size, test_inputs, test_targets, shuffle=False)

    return train_iter, test_iter, {'mean': data_mean, 'std': data_std}


def restore_candle(candle_ln, init_price):
    

    open = candle_ln[0]*init_price
    high = candle_ln[1]*open
    low = candle_ln[2]*open
    close = candle_ln[3]*open

    open = round(open, 3)
    high = round(high, 3)
    low = round(low, 3)
    close = round(close, 3)

    return open, high, low, close


def restore_candles(candles_ln, init_price, init_timestamp = None):

    open = []
    high = []
    low = []
    close = []
    index = [] if init_timestamp is not None else range(len(candles_ln))

    candles_ln = np.exp(candles_ln)

    for candle in candles_ln:
        o, h, l, c = restore_candle(candle, init_price)
        open.append(o)
        high.append(h)
        low.append(l)
        close.append(c)
        init_price = c
 
        if init_timestamp is not None:
            new_timestamps = init_timestamp + pd.to_timedelta(unit='m', value=1)
            index.append(new_timestamps)
            init_timestamp = new_timestamps
        
    #create a dataframe
    candles = pd.DataFrame({'open': open, 'high': high, 'low': low, 'close': close}, index=index)

    return candles
