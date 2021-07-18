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

    def calculate_threshold(self, train_loader):
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
        axF1.set_title('Training mses')
        axF1.scatter(error_df.index, error_df.values)
        plt.show()

        self.threshold = threshold

        return threshold



    def fit(self, train_loader, epochs=10):
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
        threshold = self.calculate_threshold(train_loader)

        return threshold

    def set_device(self, gpu=-1):
        if gpu != -1 and torch.cuda.is_available():
            device = torch.device('cuda: ' + str(gpu))
        else:
            device = torch.device('cpu')
        return device

    def evaluate(self, X, y_true, file_name=None):
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

        precision, recall, f1 = metrics(y_pred, y_true)
        print(f'threshold={self.threshold}')
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))

        colors = []
        sizes = []
        for pred, true in zip(y_pred, y_true):
            if pred == 1 and true == 1:
                colors.append(GREEN)
                sizes.append(1)
            elif pred == 1 and true == 0:
                colors.append(RED)
                sizes.append(2)
            elif pred == 0 and pred == 0:
                colors.append(BLUE)
                sizes.append(0.5)
            elif pred == 0 and pred == 1:
                colors.append(GREY)
                sizes.append(2)

        # error_df = pd.DataFrame({'reconstruction_error': mses})
        error_df = pd.DataFrame({'reconstruction_error': mses, 'label': y_true, 'pred': y_pred})
        figF1, axF1 = plt.subplots()
        axF1.set_title('Validation mses')
        point_size = 6
        # axF1.scatter(error_df.index, error_df.values, c=colors, s=sizes)
        TN = error_df[error_df['label'] == 0][error_df['pred'] == 0]['reconstruction_error']
        axF1.scatter(TN.index, TN.values, c=['b']*len(TN.values), s=point_size, zorder=1)
        TP = error_df[error_df['label'] == 1][error_df['pred'] == 1]['reconstruction_error']
        axF1.scatter(TP.index, TP.values, c=['g']*len(TP.values), s=point_size, zorder=2)
        FN = error_df[error_df['label'] != 0][error_df['pred'] == 0]['reconstruction_error']
        axF1.scatter(FN.index, FN.values, c=[0.75]*len(FN.values), s=point_size, zorder=3)
        FP = error_df[error_df['label'] != 1][error_df['pred'] == 1]['reconstruction_error']
        axF1.scatter(FP.index, FP.values, c=['r']*len(FP.values), s=point_size, zorder=4)

        plt.plot([self.threshold] * len(error_df.index), linestyle='dashed')
        plt.yscale('log')

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


