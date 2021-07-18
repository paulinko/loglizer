"""
The implementation of a Autoencoder model for anomaly detection.

Authors:
    Pauline Koch

"""
import numpy
import numpy as np
import pandas as pd
import torch.optim
import matplotlib.pyplot as plt

from ..utils import metrics
from torch import nn
from torch.nn import Linear, ReLU
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):

    def __init__(self, input_size, bottleneck_size, encoder_size, learning_rate=1e-4, decay=5e-5, device="cpu", percentile=0.97):
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
        mse = np.delete(mse, mse.argmax())
        mse = np.delete(mse, mse.argmax())
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.decay)
        for epoch in range(epochs):
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
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, epochs, epoch_loss))

        torch.save(self.state_dict(), './log_autoencoder.pth')
        threshold = self.calculate_threshold(train_loader)

        return threshold

    def set_device(self, gpu=-1):
        if gpu != -1 and torch.cuda.is_available():
            device = torch.device('cuda: ' + str(gpu))
        else:
            device = torch.device('cpu')
        return device

    def evaluate(self, X, y_true):
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

        error_df = pd.DataFrame({'reconstruction_error': mses})
        figF1, axF1 = plt.subplots()
        axF1.scatter(error_df.index, error_df.values)
        plt.show()

        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        TP = sum(y_pred[y_true == 1])
        FP = sum(y_pred[y_true != 1])
        TN = 0 # len(y_pred[y_true == 0])
        FN = 0 # len(y_pred[y_true != 0])
        print(f'{TP=} {FP=} {FN=} {TN=}')

        return precision, recall, f1


