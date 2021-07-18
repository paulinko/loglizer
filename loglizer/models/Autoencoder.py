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
class Autoencoder(nn.Module):

    def __init__(self, input_size, bottleneck_size, encoder_size, learning_rate=1e-4, decay=5e-5, device="cpu", percentile=0.97, model_name='model_autoencoder'):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            Linear(input_size, encoder_size).double(),
            ReLU(True),
            Linear(encoder_size, encoder_size).double(),
            ReLU(True),
            Linear(encoder_size, encoder_size).double(),
            ReLU(True),
            Linear(encoder_size, bottleneck_size).double(),
        )
        self.decoder = nn.Sequential(
            Linear(bottleneck_size, encoder_size).double(),
            ReLU(True),
            Linear(encoder_size, encoder_size).double(),
            ReLU(True),
            Linear(encoder_size, encoder_size).double(),
            ReLU(True),
            Linear(encoder_size, input_size).double(),
        )
        self.input_size = input_size
        self.device = self.set_device(device)
        self.decay = decay
        self.learning_rate = learning_rate
        self.threshold = 0
        self.percentile = percentile
        self.model_name = model_name


    def forward(self, inputs):
        # inputs = torch.from_numpy(X).double().to(self.device)
        bottleneck = self.encoder.forward(inputs)
        return self.decoder.forward(bottleneck)

    def calculate_threshold(self, train_loader, file_name):
        X = np.array([])
        preds = list()
        for data in train_loader:
            x_tensor = torch.from_numpy(data).double().to(self.device)
            with torch.no_grad():
                pred = self(x_tensor)
            preds.append(np.array(pred))
        preds = np.array(preds)

        # X = np.reshape(X,(-1, self.input_size))
        mse = np.mean(np.power(preds-train_loader, 2), axis=1)
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
            plt.savefig(file_name + '.png')
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
        for epoch in progressbar:
            epoch_loss = 0
            batch_cnt = 0
            for X in train_loader:
                x_tensor = torch.from_numpy(X).double().to(self.device)
                predicted = self.forward(x_tensor)
                loss = criterion(predicted, x_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1

            epoch_loss = epoch_loss / batch_cnt

            txt = 'epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, epoch_loss)
            progressbar.set_description(txt)
            print(txt)

        torch.save(self.state_dict(), './{}.pth'.format(self.model_name))
        threshold = self.calculate_threshold(train_loader, file_name)

        return threshold

    def set_device(self, gpu=-1):
        if gpu != -1 and torch.cuda.is_available():
            device = torch.device('cuda: ' + str(gpu))
        else:
            device = torch.device('cpu')
        return device

    def evaluate(self, X, y_true, file_name=None, session_ids = None):
        self.eval()
        print('====== Evaluation summary ======')
        with torch.no_grad():
            x_tensor = torch.from_numpy(X).double().to(self.device)
            preds = self.forward(x_tensor)
        mses = np.power(preds - X, 2).mean(axis=1)
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
            TN = session_df[session_df['label'] == 0][session_df['pred'] == 0]['reconstruction_error']
            axF1.scatter(TN.index, TN.values, c=['b'] * len(TN.values), s=point_size, zorder=1, label='TN')
            TP = session_df[session_df['label'] == 1][session_df['pred'] == 1]['reconstruction_error']
            axF1.scatter(TP.index, TP.values, c=['g'] * len(TP.values), s=point_size, zorder=2, label='TP')
            FN = session_df[session_df['label'] != 0][session_df['pred'] == 0]['reconstruction_error']
            axF1.scatter(FN.index, FN.values, c=[0.75] * len(FN.values), s=point_size, zorder=3, label='FN')
            FP = session_df[session_df['label'] != 1][session_df['pred'] == 1]['reconstruction_error']
            axF1.scatter(FP.index, FP.values, c=['r'] * len(FP.values), s=point_size, zorder=4, label='FP')

            plt.plot([self.threshold] * len(error_df.index), linestyle='dashed', label='Grenzwert')
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
        axF1.scatter(TN.index, TN.values, c=['b']*len(TN.values), s=point_size, zorder=1, label='TN')
        TP = error_df[error_df['label'] == 1][error_df['pred'] == 1]['reconstruction_error']
        axF1.scatter(TP.index, TP.values, c=['g']*len(TP.values), s=point_size, zorder=2, label='TP')
        FN = error_df[error_df['label'] != 0][error_df['pred'] == 0]['reconstruction_error']
        axF1.scatter(FN.index, FN.values, c=[0.75]*len(FN.values), s=point_size, zorder=3, label='FN')
        FP = error_df[error_df['label'] != 1][error_df['pred'] == 1]['reconstruction_error']
        axF1.scatter(FP.index, FP.values, c=['r']*len(FP.values), s=point_size, zorder=4, label='FP')

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


