"""
The implementation of a Autoencoder model for anomaly detection.

Authors:
    Pauline Koch

"""
import numpy
import numpy as np
import pandas as pd
import torch.optim
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..utils import metrics
from torch import nn
from torch.nn import Linear, ReLU
import matplotlib.pyplot as plt

# GREEN = [0,255,0]
# RED = [255, 0, 0]
# BLUE = [0, 0, 255]
# GREY = [128, 128, 128]

GREEN = 'g'
RED = 'r'
BLUE = 'b'
GREY = 0.75


class Encoder(nn.Module):

    def __init__(self, batch_size, bottleneck_size, hidden_size, seq_len, dropout=0, num_layers=2):
        super(Encoder, self).__init__()
        self.feature_count = 1
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.bottleneck_size = bottleneck_size
        self.hidden_size = hidden_size
        self.lstm_1 = nn.LSTM(self.feature_count, hidden_size,num_layers=num_layers, batch_first=True, dropout=dropout)
        self.lstm_2 = nn.LSTM(hidden_size, bottleneck_size, batch_first=True)

    def forward(self, x):
        x = x.reshape((-1, self.seq_len, self.feature_count))
        x, _ = self.lstm_1.forward(x)
        x, (hidden, _) = self.lstm_2.forward(x)
        return hidden.reshape(-1, self.bottleneck_size, self.feature_count)


class Decoder(nn.Module):

    def __init__(self, batch_size, bottleneck_size, hidden_size, seq_len,dropout=0, num_layers=2):
        super(Decoder, self).__init__()
        self.feature_count = 1
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.bottleneck_size = bottleneck_size
        self.hidden_size = hidden_size
        self.lstm_1 = nn.LSTM(self.bottleneck_size, self.bottleneck_size, batch_first=True, dropout=dropout)
        self.lstm_2 = nn.LSTM(bottleneck_size, hidden_size,num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.feature_count)

    def forward(self, x):
        # x = x.repeat(self.seq_len, self.feature_count)
        x = x.repeat(self.batch_size, self.seq_len, self.feature_count)
        x = x.reshape((-1, self.seq_len, self.bottleneck_size))
        x, _ = self.lstm_1.forward(x)
        x, _ = self.lstm_2.forward(x)
        return self.out(x).flatten()


class AutoencoderLSTM(nn.Module):

    def __init__(self, batch_size,
                 bottleneck_size,
                 encoder_size,
                 learning_rate=1e-4,
                 decay=5e-5,
                 device="cpu",
                 percentile=0.97,
                 num_directions=2,
                 dropout = 0,
                 num_layers = 2,
                 model_name='model_autoencoder'):
        super(AutoencoderLSTM, self).__init__()
        input_size = 1
        self.num_directions = num_directions
        self.input_size = input_size
        self.device = self.set_device(device)
        self.decay = decay
        self.learning_rate = learning_rate
        self.threshold = 0
        self.percentile = percentile
        self.model_name = model_name
        self.seq_size = batch_size
        self.batch_size = 1
        self.encoder_size = encoder_size
        self.bottleneck_size = bottleneck_size

        self.encoder = Encoder(self.batch_size, bottleneck_size, encoder_size, self.seq_size, dropout, num_layers).to(device)
        self.decoder = Decoder(self.batch_size, bottleneck_size, encoder_size, self.seq_size, dropout, num_layers).to(device)

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def init_hidden(self):
        h0 = torch.zeros(self.num_directions, self.batch_size, self.encoder_size).to(self.device)
        c0 = torch.zeros(self.num_directions, self.batch_size, self.encoder_size).to(self.device)
        return (h0, c0)

    def calculate_threshold(self, train_loader, file_name=None):
        X = np.array([])
        preds = list()
        x_tensor = torch.from_numpy(train_loader).float().to(self.device)
        with torch.no_grad():
            preds = self(x_tensor)
        # for data in train_loader:
        #     x_tensor = torch.from_numpy(data).float().to(self.device)
        #     with torch.no_grad():
        #         pred = self(x_tensor.reshape(self.batch_size, self.seq_size, 1))
        #     preds.append(np.array(pred).reshape(self.seq_size))
        # preds = np.array(preds)

        # X = np.reshape(X,(-1, self.input_size))
        preds = np.array(preds.reshape(-1, self.seq_size))
        mse = np.mean(np.power(preds - train_loader, 2), axis=1)
        # mse = np.delete(mse, mse.argmax())
        # mse = np.delete(mse, mse.argmax())
        error_df = pd.DataFrame({'reconstruction_error': mse})

        # Set threshold at the 99th quartile.
        threshold = error_df.quantile(self.percentile)['reconstruction_error']
        print(f'{threshold=}')
        print(error_df.describe())

        figF1, axF1 = plt.subplots()
        axF1.set_title('Training MSEs')
        axF1.scatter(error_df.index, error_df.values)
        if file_name:
            plt.savefig(file_name + '-train.png')
            with open(file_name + '.txt', 'w') as result:
                result.write(error_df.describe().to_string())
        else:
            plt.show()

        self.threshold = threshold

        return threshold

    def fit(self, train_loader, epochs=10, file_name=None):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.decay, amsgrad=True)
        progressbar = tqdm.tqdm(range(epochs), total=epochs)
        dataset = TensorDataset(torch.tensor(train_loader, dtype=torch.float))
        dataloader = DataLoader(dataset, batch_size=2048, pin_memory=True)
        writer = SummaryWriter()
        for epoch in progressbar:
            epoch_loss = 0
            batch_cnt = 0
            # X = train_loader
            # for X in train_loader:
            for step, (X) in enumerate(dataloader):
                x_tensor = X[0]
                predicted = self.forward(x_tensor)
                loss = criterion(predicted.reshape(-1, self.seq_size), x_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1


            epoch_loss = epoch_loss / batch_cnt
            writer.add_scalar('train_loss', epoch_loss, epoch + 1)
            txt = 'epoch [{}/{}], loss:{:.20f}'.format(epoch + 1, epochs, epoch_loss)
            progressbar.set_description(txt)
            print(txt)


        writer.close()
        torch.save(self.state_dict(), './{}.pth'.format(self.model_name))
        threshold = self.calculate_threshold(train_loader, file_name)

        return threshold

    def set_device(self, gpu=-1):
        if gpu != -1 and torch.cuda.is_available():
            device = torch.device('cuda: ' + str(gpu))
        else:
            device = torch.device('cpu')
        return device

    def evaluate(self, X, y_true, file_name=None, session_ids=None, sample=1):
        self.eval()
        print('====== Evaluation summary ======')
        with torch.no_grad():
            x_tensor = torch.from_numpy(X).float().to(self.device)
            preds = self.forward(x_tensor)
        mses = np.power(preds.reshape(-1, self.seq_size) - X, 2).mean(axis=1)
        # mses = np.delete(mses, mses.argmax())
        # mses = np.delete(mses, mses.argmax())
        y_pred = np.zeros_like(y_true)
        y_pred[mses > self.threshold] = 1

        # precision, recall, f1 = metrics(y_pred, y_true)
        # print(f'threshold={self.threshold}')
        # print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))

        # error_df = pd.DataFrame({'reconstruction_error': mses})
        error_df = pd.DataFrame({'reconstruction_error': [mse.item() for mse in mses], 'label': y_true, 'pred': y_pred})
        if session_ids is not None:
            error_df['session_ids'] = session_ids
            session_df = error_df.groupby('session_ids', as_index=False).max()
            figF1, axF1 = plt.subplots()
            axF1.set_title('Validation Session MSEs')
            point_size = 6
            # axF1.scatter(error_df.index, error_df.values, c=colors, s=sizes)
            TN = session_df[session_df['label'] == 0][session_df['pred'] == 0]['reconstruction_error'].sample(frac=sample)
            axF1.scatter(TN.index, TN.values, c=['b'] * len(TN.values), s=point_size, zorder=1, label='TN')
            TP = session_df[session_df['label'] == 1][session_df['pred'] == 1]['reconstruction_error'].sample(frac=sample)
            axF1.scatter(TP.index, TP.values, c=['g'] * len(TP.values), s=point_size, zorder=2, label='TP')
            FN = session_df[session_df['label'] != 0][session_df['pred'] == 0]['reconstruction_error'].sample(frac=sample)
            axF1.scatter(FN.index, FN.values, c=[0.75] * len(FN.values), s=point_size, zorder=3, label='FN')
            FP = session_df[session_df['label'] != 1][session_df['pred'] == 1]['reconstruction_error'].sample(frac=sample)
            axF1.scatter(FP.index, FP.values, c=['r'] * len(FP.values), s=point_size, zorder=4, label='FP')

            plt.plot([self.threshold] * len(TN.index), linestyle='dashed', label='Grenzwert')
            plt.yscale('log')
            plt.ylabel('MSE')
            plt.legend()
            precision, recall, f1 = metrics(session_df['pred'], session_df['label'])
            print('Session: Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
            TP = len(TP)
            FP = len(FP)
            total = session_df['label'].sum()
            TN = len(TN)
            FN = len(FN)
            print(f'Session: {TP=} {FP=} {FN=} {TN=}, {total=}')
            if file_name is None:
                plt.show()
            else:
                plt.savefig(file_name + '.session.png')
                with open(file_name + '.session.txt', 'w') as session_stats:
                    session_stats.write(f'{precision=}, {recall=}, {f1=}')

        figF1, axF1 = plt.subplots()
        axF1.set_title('Validation MSEs')
        point_size = 6
        # axF1.scatter(error_df.index, error_df.values, c=colors, s=sizes)
        TN = error_df[error_df['label'] == 0][error_df['pred'] == 0]['reconstruction_error']
        axF1.scatter(TN.index, TN.values, c=['b'] * len(TN.values), s=point_size, zorder=1, label='TN')
        TP = error_df[error_df['label'] == 1][error_df['pred'] == 1]['reconstruction_error']
        axF1.scatter(TP.index, TP.values, c=['g'] * len(TP.values), s=point_size, zorder=2, label='TP')
        FN = error_df[error_df['label'] != 0][error_df['pred'] == 0]['reconstruction_error']
        axF1.scatter(FN.index, FN.values, c=[0.75] * len(FN.values), s=point_size, zorder=3, label='FN')
        FP = error_df[error_df['label'] != 1][error_df['pred'] == 1]['reconstruction_error']
        axF1.scatter(FP.index, FP.values, c=['r'] * len(FP.values), s=point_size, zorder=4, label='FP')

        plt.plot([self.threshold] * len(error_df.index), linestyle='dashed', label='Grenzwert')
        plt.legend()
        plt.yscale('log')
        plt.ylabel('MSE')

        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)

        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        TP = len(TP)
        FP = len(FP)
        total = sum(y_true)
        TN = len(TN)
        FN = len(FN)
        print(f'{TP=} {FP=} {FN=} {TN=}, {total=}')

        return precision, recall, f1
