"""
The interface for data preprocessing.

Authors:
    LogPAI Team

"""

import pandas as pd
import os
import numpy as np
import re
from collections import Counter
import torch.nn.functional as F

import torch
from scipy.special import expit
from itertools import compress
from torch.utils.data import DataLoader, Dataset


class Iterator(Dataset):
    def __init__(self, data_dict, batch_size=32, shuffle=False, num_workers=1):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.iter = DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __getitem__(self, index):
        return {k: self.data_dict[k][index] for k in self.keys}

    def __len__(self):
        return self.data_dict["SessionId"].shape[0]


NUM_CLASSES = 36 + 2


class Vectorizer(object):

    def fit_transform(self, x_train, window_y_train, y_train, normalize=False, use_onehot = True):
        if len(window_y_train) > 0:
            self.label_mapping = {eid: idx for idx, eid in enumerate(window_y_train.unique(), 2)}
        else:
            unique_events = np.unique(np.concatenate(x_train['EventSequence']))
            self.label_mapping = {eid: idx for idx, eid in enumerate(unique_events, 2)}
        self.label_mapping["#OOV"] = 0
        self.label_mapping["#Pad"] = 1
        self.use_onehot = use_onehot
        self.num_labels = len(self.label_mapping)
        self.normalize = normalize
        return self.transform(x_train, window_y_train, y_train)

    @staticmethod
    def to_onehot(l):
        return F.one_hot(torch.tensor(l), num_classes=NUM_CLASSES).cpu().detach().numpy()

    def transform(self, x, window_y, y):
        if self.normalize:
            x["EventSequence"] = x["EventSequence"].map(
                lambda x: [self.label_mapping.get(item, 0) / self.num_labels for item in x])
            window_y = window_y.map(lambda x: self.label_mapping.get(x, 0) / self.num_labels)
        else:
            x["EventSequence"] = x["EventSequence"].map(lambda x: [self.label_mapping.get(item, 0) for item in x])
            window_y = window_y.map(lambda x: self.label_mapping.get(x, 0))
        y = y
        if self.use_onehot:
            data_dict = {"SessionId": x["SessionId"].values, "window_y": window_y.values, "y": y.values,
                     "x": np.array(list(map(self.to_onehot, x["EventSequence"].tolist())))}
        else:
            data_dict = {"SessionId": x["SessionId"].values, "window_y": window_y.values, "y": y.values,
                         "x": np.array(x["EventSequence"].tolist())}
        return data_dict


class FeatureExtractor(object):

    def __init__(self):
        self.idf_vec = None
        self.mean_vec = None
        self.events = None
        self.term_weighting = None
        self.normalization = None
        self.oov = None

    def fit_transform(self, X_seq, term_weighting=None, normalization=None, oov=False, min_count=1):
        """ Fit and transform the data matrix

        Arguments
        ---------
            X_seq: ndarray, log sequences matrix
            term_weighting: None or `tf-idf`
            normalization: None or `zero-mean`
            oov: bool, whether to use OOV event
            min_count: int, the minimal occurrence of events (default 0), only valid when oov=True.

        Returns
        -------
            X_new: The transformed data matrix
        """
        print('====== Transformed train data summary ======')
        self.term_weighting = term_weighting
        self.normalization = normalization
        self.oov = oov

        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        self.events = X_df.columns
        X = X_df.values
        if self.oov:
            oov_vec = np.zeros(X.shape[0])
            if min_count > 1:
                idx = np.sum(X > 0, axis=0) >= min_count
                oov_vec = np.sum(X[:, ~idx] > 0, axis=1)
                X = X[:, idx]
                self.events = np.array(X_df.columns)[idx].tolist()
            X = np.hstack([X, oov_vec.reshape(X.shape[0], 1)])

        num_instance, num_event = X.shape
        if self.term_weighting == 'tf-idf':
            df_vec = np.sum(X > 0, axis=0)
            self.idf_vec = np.log(num_instance / (df_vec + 1e-8))
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
            X = idf_matrix
        if self.normalization == 'zero-mean':
            mean_vec = X.mean(axis=0)
            self.mean_vec = mean_vec.reshape(1, num_event)
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'positive-mean-vec':
            self.min_vec = X.min(axis=0)
            X = X - self.min_vec
            self.max_vec = X.max(axis=0)
            self.max_vec[self.max_vec == 0] = 1
            X = X / self.max_vec
        elif self.normalization == 'positive-mean':
            self.min_vec = X.min()
            X = X - self.min_vec
            self.max_vec = X.max()
            X = X / self.max_vec
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X

        print('Train data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1]))
        return X_new

    def transform(self, X_seq):
        """ Transform the data matrix with trained parameters

        Arguments
        ---------
            X: log sequences matrix
            term_weighting: None or `tf-idf`

        Returns
        -------
            X_new: The transformed data matrix
        """
        print('====== Transformed test data summary ======')
        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        empty_events = set(self.events) - set(X_df.columns)
        for event in empty_events:
            X_df[event] = [0] * len(X_df)
        X = X_df[self.events].values
        if self.oov:
            oov_vec = np.sum(X_df[X_df.columns.difference(self.events)].values > 0, axis=1)
            X = np.hstack([X, oov_vec.reshape(X.shape[0], 1)])

        num_instance, num_event = X.shape
        if self.term_weighting == 'tf-idf':
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
            X = idf_matrix
        if self.normalization == 'zero-mean':
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'positive-mean' or self.normalization == 'positive-mean-vec':
            X = X - self.min_vec
            X = X / self.max_vec
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X

        print('Test data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1]))

        return X_new
