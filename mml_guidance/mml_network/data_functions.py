import torch
from sklearn.preprocessing import MinMaxScaler

class DataManager():
    def __init__(self, source, input_window, output_window, split):
        self.source = source
        self.input_window = input_window
        self.output_window = output_window
        self.split = split

def create_inout_sequences(input_data, input_window, output_window=1):
    inout_seq = []
    L = len(input_data)
    for i in range(L-input_window-output_window):
        train_seq = input_data[i:i+input_window]
        train_label = input_data[i+output_window:i+input_window+output_window]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)

def get_data(data, split, input_window, output_window = 1, device='cpu', scale_data = False):
    """Split ratio of training data"""
    if scale_data:
        orig_shape = data.shape
        scaler = MinMaxScaler(feature_range=(-1, 1)) 
        data = scaler.fit_transform(data.reshape(-1,1)).reshape(orig_shape)
    series = data
    
    split = round(split*len(series))
    train_data = series[:split]
    test_data = series[split:]

    train_data = train_data # Training data augmentation, increase amplitude for the model to better generalize.(Scaling by 2 is aribitrary)
                              # Similar to image transformation to allow model to train on wider data sets

    train_sequence = create_inout_sequences(train_data,input_window, output_window)
    train_sequence = train_sequence[:-output_window]

    test_data = create_inout_sequences(test_data,input_window, output_window)
    test_data = test_data[:-output_window]

    return train_sequence.to(device), test_data.to(device)

def get_batch(source, i, batch_size, input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return torch.squeeze(input), torch.squeeze(target)