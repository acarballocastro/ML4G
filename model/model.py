import copy
import math
import os
import pickle
import random
import time
from typing import Tuple

import numpy as np
import torch
from scipy import stats
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from sklearn import metrics


# pip install numpy pandas requests tqdm torch sklearn

batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_batch(source: Tuple, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tuple[Tensor, Tensor], shape [full_seq_len, embedding_size]. This is the whole dataset, with all the bins and the embedding size
        i: int. Batch size

    Returns:
        tuple (data, target), where data has shape [seq_len, embedding_size] and
        target has shape [seq_len * embedding_size]
    """
    x, y = source
    seq_len = min(batch_size, len(x) - 1 - i)
    data = x[i:i + seq_len]
    target = y[i:i + seq_len]
    return data.to(device), target.to(device)


def train_epoch(model: nn.Module, X, y, criterion, optimizer, epoch) -> None:
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
        pbar.set_postfix(loss=loss.item(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        #scheduler.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            #lr = scheduler.get_last_lr()[0]
            lr= 0.000411
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model: nn.Module, eval_data: Tuple, criterion):
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
    mse = metrics.mean_squared_error(tgts, preds)

    #spearman correlation between targets and predictions
    print(preds)
    spearman = stats.spearmanr(tgts, preds)[0]

    return total_loss / (len(eval_data) - 1), mse, spearman


class TransformerRegressor:

    def __str__(self):
        return 'TransformerRegressor'

    def __init__(self):

        #Here I have the parameters of the transformer model

        n_features = 5  #Initial nº of histones
        emsize = 30  # embedding dimension. It must be divisible by nhead.
        nhead = 3  # number of heads in nn.MultiheadAttention
        d_hid = 166  # dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 10  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        dropout = 0.1525  # dropout probability
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
        epochs = 100

        mean_squared_error = nn.MSELoss()

        criterion = mean_squared_error
        initial_lr = 1e-4  # learning rate #TODO: changed it to 1e-3 -> 1e-4
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=initial_lr)
        #scheduler = self.get_polynomial_decay_schedule_with_warmup(optimizer, 128,128 * 300)  # torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
        best_model = None

        try:
            for epoch in range(1, epochs + 1):
                epoch_start_time = time.time()
                train_epoch(self.model, X, y, criterion, optimizer,  epoch)
                assert self.x_val is not None
                assert self.y_val is not None
                val_loss, mse, spearman = evaluate(self.model, (self.x_val, self.y_val), criterion)
                elapsed = time.time() - epoch_start_time
                print('-' * 89)
                print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                      f'valid loss {val_loss:5.2f}  | mse {mse:5.2f} |'# lr {scheduler.get_last_lr()[0]:02.6f} | '
                      f'spearman {spearman:5.2f}')
                print('-' * 89)

                if mse < best_mse:
                    best_mse = mse
                    best_model = copy.deepcopy(self.model)

                #scheduler.step()
        except KeyboardInterrupt:
            print("Finishing training...")
        self.model = best_model

    def predict(self, X):
        return self.model(X)


class PositionalEncoding(nn.Module): #The positional encoder is using sine and cosine, which we saw in class that is not ideal for DNA


    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term) #TODO: change this
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
                 nlayers: int, dropout: float = 0.1): #TODO: I am changing dropout 0.5 -> 0.1
        super().__init__()
        self.model_type = 'Transformer'
        self.encoder1 = nn.Linear(dim_input, 2*d_model) #This is the embedding layer
        self.relu1 = nn.ReLU()
        self.encoder2 = nn.Linear(2*d_model, 2*d_model)
        self.relu2 = nn.ReLU()
        self.embeddings =nn.Linear(2*d_model, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout) #This is with sine cosine so not ideal. Check relative encoding
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.regressor = nn.Linear(d_model, 1) #Fully connected layer that gives the output
        self.relu = nn.ReLU() #Add a relu so that the output is always positive

        self.init_weights()



    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder1.weight.data.uniform_(-initrange, initrange)
        self.encoder2.weight.data.uniform_(-initrange, initrange)
        self.embeddings.weight.data.uniform_(-initrange, initrange)
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
        src = self.encoder1(src) * math.sqrt(2*self.d_model) #Normalization of the variance of the input. Encoding layer
        src = self.relu1(src)
        src = self.encoder2(src)
        src = self.relu2(src)
        src = self.embeddings(src) 
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.regressor(output[0, :, :])  # take the CLS  and project it through regressor
        output = self.relu(output)

        return output


def main():

    # fix random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # load data

    X_train = pickle.load(open("../data/X_train.pickle", "rb"))
    X_val = pickle.load(open("../data/X_val.pickle", "rb"))

    train = list(X_train.items())
    validation = list(X_val.items())
    

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


    # Create a -1000 tensor and append to the first column of x_data in the dimension 1
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
    #if os.path.isfile('model_transformer.pt'):
    #   model.model.load_state_dict(torch.load('model_transformer.pt'))

    model.set_validation_data(x_data_val, y_data_val)
    # fit model
    model.fit(x_data_train, y_data_train)
    # detach from gpu and save model
    model.model.cpu()
    torch.save(model.model.state_dict(), 'model_improved_100epochs_seed.pt')

def test_saved_model():
    global device
    device = 'cpu'
    X_test = pickle.load(open("../data/X3_test.pickle", "rb"))
    test = list(X_test.items())

    # Prepare test tensor
    test_names = [x[0] for x in test]
    test_features = [x[1][0] for x in test]
    test_features_nparray = np.stack([i for i in test_features], axis=0)

    x_data_test = torch.from_numpy(test_features_nparray.astype('float32'))

    # Create a -1000 tensor and append to the first column of x_data in the dimension 1
    initial_tensor = torch.full((x_data_test.size(dim=0), 1, x_data_test.size(dim=2)), -1000)
    x_data_test = torch.cat((initial_tensor, x_data_test), dim=1)

    # check that x_data has -1000 in first position of dimension 1
    assert torch.all(x_data_test[:, 0, :] == -1000)

    #x_data_test.to(device)

    # load pytorch model from file 'model_improved_300epochs.pt'
    model = TransformerRegressor()
    model.model.load_state_dict(torch.load('model_improved_100epochs_seed.pt'))
    model.model.eval()

    # predict
    y_pred = model.predict(x_data_test)
    y_numpy = y_pred.detach().numpy()
    y_predict_numpy_original = np.power(2, y_numpy) - 1


    # save predictions, header: index,gene_name,gex_predicted
    with open('gex_predicted.csv', 'w') as f:
        f.write(',gene_name,gex_predicted' + '\n')
        for i in range(len(y_predict_numpy_original)):
            f.write(str(i) + ',' + str(test_names[i]).split('_')[0] + ',' + str(y_predict_numpy_original[i][0]) + '\n')


def test_val():
    global device
    device = 'cpu'
    X_val = pickle.load(open("../data/X_val.pickle", "rb"))
    validation = list(X_val.items())
    val_features = [x[1][0] for x in validation]
    val_features_nparray = np.stack([i for i in val_features], axis=0)
    val_labels = [x[1][1] for x in validation]
    val_labels_nparray = np.array(val_labels).astype('float32')
    val_labels_original = np.power(2, val_labels_nparray) - 1
    x_data_val = torch.from_numpy(val_features_nparray.astype('float32'))  # batch x seq_len x dim
    y_data_val = torch.from_numpy(val_labels_original)

    # Create a -1000 tensor and append to the first column of x_data in the dimension 1
    initial_tensor = torch.full((x_data_val.size(dim=0), 1, x_data_val.size(dim=2)), -1000)
    x_data_val = torch.cat((initial_tensor, x_data_val), dim=1).to(device)

    # check that x_data has -1000 in first position of dimension 1
    assert torch.all(x_data_val[:, 0, :] == -1000)

    # load pytorch model from file 'model_improved_300epochs.pt'
    model = TransformerRegressor()
    model.model.load_state_dict(torch.load('model_improved_100epochs_seed.pt'))
    model.model.to(device)
    model.model.eval()

    # predict
    y_pred = model.predict(x_data_val)
    y_numpy = y_pred.detach().numpy()
    y_predict_numpy_original = np.power(2, y_numpy) - 1

    # calculate spearman correlation
    spearman = stats.spearmanr(val_labels_original, y_predict_numpy_original)[0]
    print(spearman)



if __name__ == '__main__':
    #main()
    test_saved_model()
    #test_val()
