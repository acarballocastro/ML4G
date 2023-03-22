import base64
import copy
import io
import math
import os
import pickle
import random
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from balanced_loss import Loss

# pip install numpy pandas requests tqdm torch sklearn

batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_batch(source: Tuple, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tuple[Tensor, Tensor], shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    x, y = source
    seq_len = min(batch_size, len(x) - 1 - i)
    data = x[i:i + seq_len]
    target = y[i:i + seq_len]
    return data.to(device), target.to(device)


def train_epoch(model: nn.Module, X, y, criterion, optimizer, scheduler, epoch) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(X) // batch_size
    for batch, i in enumerate(pbar := tqdm(range(0, X.size(0) - 1, batch_size))):
        data, targets = get_batch((X, y), i)
        output = model(data)
        output = output.view(-1)
        loss = criterion(output, targets)
        pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model: nn.Module, eval_data: Tuple, criterion) -> (float, float, float, float):
    X, y = eval_data
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    tgts, preds = [], []
    with torch.no_grad():
        for i in range(0, X.size(0) - 1, batch_size):
            data, targets = get_batch((X, y), i)
            seq_len = data.size(0)
            # if seq_len != bptt:
            #     src_mask = src_mask[:seq_len, :seq_len]
            output = model(data)
            output = output.view(-1)
            preds.extend(output.tolist())
            tgts.extend(targets.tolist())
            total_loss += seq_len * criterion(output, targets).item()
    tgts = np.array(tgts)
    preds = np.array(preds)
    # metrics for regression
    mse = np.mean((tgts - preds) ** 2)

    #spearman correlation between targets and predictions
    spearman = stats.spearmanr(tgts, preds)[0]

    return total_loss / (len(eval_data) - 1), mse, spearman


class TransformerRegressor:

    def __str__(self):
        return 'TransformerRegressor'

    def __init__(self):
        n_features = 5  #
        emsize = 200  # embedding dimension. It must be divisible by nhead.
        d_hid = 300  # dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 4  # number of heads in nn.MultiheadAttention
        dropout = 0.2  # dropout probability
        self.model = TransformerModel(n_features, emsize, nhead, d_hid, nlayers, dropout).to(device)
        self.x_val = None
        self.y_val = None

    def set_validation_data(self, x_test, y_test):
        self.x_val = x_test
        self.y_val = y_test

    def get_polynomial_decay_schedule_with_warmup(
            self, optimizer, num_warmup_steps, num_training_steps, lr_end=1e-6, power=1.0, last_epoch=-1
    ):

        lr_init = optimizer.defaults["lr"]
        if not (lr_init > lr_end):
            raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif current_step > num_training_steps:
                return lr_end / lr_init  # as LambdaLR multiplies by lr_init
            else:
                lr_range = lr_init - lr_end
                decay_steps = num_training_steps - num_warmup_steps
                pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
                decay = lr_range * pct_remaining ** power + lr_end
                return decay / lr_init  # as LambdaLR multiplies by lr_init

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def fit(self, X, y):
        best_mse = float('inf')
        epochs = 20

        mean_squared_error = nn.MSELoss()

        criterion = mean_squared_error
        initial_lr = 1e-2  # learning rate
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=initial_lr)
        scheduler = self.get_polynomial_decay_schedule_with_warmup(optimizer, 128,
                                                                   128 * 300)  # torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
        best_model = None

        try:
            for epoch in range(1, epochs + 1):
                epoch_start_time = time.time()
                train_epoch(self.model, X, y, criterion, optimizer, scheduler, epoch)
                assert self.x_val is not None
                assert self.y_val is not None
                val_loss, mse, spearman = evaluate(self.model, (self.x_val, self.y_val), criterion)
                elapsed = time.time() - epoch_start_time
                print('-' * 89)
                print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                      f'valid loss {val_loss:5.2f}  | mse {mse:5.2f} | lr {scheduler.get_last_lr()[0]:02.6f} | '
                      f'spearman {spearman:5.2f}')
                print('-' * 89)

                if mse < best_mse:
                    best_mse = mse
                    best_model = copy.deepcopy(self.model)

                scheduler.step()
        except KeyboardInterrupt:
            print("Finishing training...")
        self.model = best_model

    def predict(self, X):
        return self.model(X)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, dim_input: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(dim_input, d_model)
        self.d_model = d_model
        self.regressor = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.regressor.bias.data.zero_()
        self.regressor.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # exchange batch and seq dim
        src_key_padding_mask = src[:, :, 0].isnan()  # this mask is over elements of the sequence,
        # replace nans with 0s
        src = src.nan_to_num()
        # not over the sequence itself. Shape: [batch_size, seq_len]
        src = src.permute(1, 0, 2) # [batch_size, seq_len, dim_input] -> [seq_len, batch_size, dim_input]
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.regressor(output[0, :, :])  # take the CLS  and project it through regressor
        output = self.relu(output)
        return output


def main():
    # load data/X_train.pickle, data/X_val.pickle, data/X3_test.pickle

    X_train = pickle.load(open("data/X_train.pickle", "rb"))
    X_val = pickle.load(open("data/X_val.pickle", "rb"))
    X_test = pickle.load(open("data/X3_test.pickle", "rb"))

    train = list(X_train.items())
    validation = list(X_val.items())
    test = list(X_test.items())

    # random shuffle train
    random.shuffle(train)

    # Prepare train tensor
    train_names = [x[0] for x in train]
    train_features = [x[1][0] for x in train]
    train_labels = [x[1][1] for x in train]
    train_features_nparray = np.stack([i for i in train_features], axis=0)

    # to torch tensor
    x_data_train = torch.from_numpy(train_features_nparray.astype('float32'))
    y_data_train = torch.from_numpy(np.array(train_labels).astype('float32'))

    # Prepare validation tensor
    val_names = [x[0] for x in validation]
    val_features = [x[1][0] for x in validation]
    val_labels = [x[1][1] for x in validation]
    val_features_nparray = np.stack([i for i in val_features], axis=0)

    x_data_val = torch.from_numpy(val_features_nparray.astype('float32'))  # batch x seq_len x dim
    y_data_val = torch.from_numpy(np.array(val_labels).astype('float32'))

    # Prepare test tensor
    test_names = [x[0] for x in test]
    test_features = [x[1][0] for x in test]
    test_features_nparray = np.stack([i for i in test_features], axis=0)

    x_data_test = torch.from_numpy(test_features_nparray.astype('float32'))

    """
    
    #Create a nan tensor to add to x_data

    nan_tensor= torch.full((x_data.size(dim=0), 1, x_data.size(dim=2)), np.nan)

    #add extra column to x_data in dimension 1 with nans

    #x_data = torch.cat((x_data, nan_tensor), dim=1)
    #x_data = torch.cat((nan_tensor, x_data), dim=1)

    # check that x_data has nans in first position of dimension 1 (last vector of every sequence)
    assert torch.isnan(x_data[:, 0, :]).any()

    # shift data to the right in dimension 1
    #x_data = torch.roll(x_data, 1, 1)
    x_data[:, 0, :] = -1000 * torch.ones(seq_length)  # set first vector of every sequence to -1000

    """

    # Alternative method: Create a -1000 tensor and append to the first column of x_data in the dimension 1
    initial_tensor = torch.full((x_data_train.size(dim=0), 1, x_data_train.size(dim=2)), -1000)
    x_data_train = torch.cat((initial_tensor, x_data_train), dim=1)

    # check that x_data has -1000 in first position of dimension 1
    assert torch.all(x_data_train[:, 0, :] == -1000)

    x_data_train.to(device)
    y_data_train.to(device)

    # split data
    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
    # load pytorch model from file 'model.pt'

    # create model
    model = TransformerRegressor()
    # if file exists load
    if os.path.isfile('model_transformer.pt'):
        model.model.load_state_dict(torch.load('model_transformer.pt'))

    model.set_validation_data(x_data_val, y_data_val)
    # fit model
    model.fit(x_data_train, y_data_train)
    # detach from gpu and save model
    model.model.cpu()
    torch.save(model.model.state_dict(), 'model_nomedian.pt')


if __name__ == '__main__':
    main()
