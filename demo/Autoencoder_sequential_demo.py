#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import pandas
import torch

sys.path.append('../')
from loglizer.models import Autoencoder
from loglizer import dataloader, preprocessing
import pickle
import numpy as np
import os

struct_log = '../data/HDFS/HDFS_100k.log_structured.csv'  # The structured log file
label_file = '../data/HDFS/anomaly_label.csv'  # The anomaly label file

window_size = 300
generate = True
MODEL_PATH = 'log_autoencoder'
EPOCHS = 10
PERCENTILE = 0.99
load = False and not generate
#
if __name__ == '__main__':
    if generate:
        (x_train, _, y_train), (x_test, _, y_test) = dataloader.load_HDFS(struct_log,
                                                                          label_file=label_file,
                                                                          window_size=window_size,
                                                                          train_ratio=0.5,
                                                                          split_type='uniform',
                                                                          zero_positive=True)
        with open(f'WIP_ws{window_size}', 'wb') as pickle_file:
            pickle.dump({'train': (x_train, y_train), 'test': (x_test, y_test)}, pickle_file)
    else:
        with open(f'WIP_ws{window_size}', 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            x_train, y_train = data['train']
            x_test, y_test = data['test']

    feature_extractor = preprocessing.Vectorizer()
    unique_events = np.unique(np.concatenate(x_train['EventSequence']))
    train_dataset = feature_extractor.fit_transform(x_train, pandas.Series([], dtype=pandas.Float64Dtype), y_train,
                                                    normalize=True)
    x_train, y_train = train_dataset['x'], train_dataset['y']
    test_dataset = feature_extractor.transform(x_test, pandas.Series([], dtype=pandas.Float64Dtype), y_test)
    x_test, y_test = test_dataset['x'], test_dataset['y']

    bottleneck_size = int(x_test.shape[1] // 2)
    encoder_sizer = x_test.shape[1] * 1
    model_name = f'{MODEL_PATH}_ws{window_size}_epoch{EPOCHS}'
    model = Autoencoder(input_size=x_test.shape[1],
                        bottleneck_size=bottleneck_size,
                        encoder_size=encoder_sizer,
                        learning_rate=1e-2,
                        decay=1e-3,
                        percentile=PERCENTILE,
                        model_name=model_name
                        )

    if load:
        model.load_state_dict(torch.load(f'{model_name}.pth'))
        with open(model_name + '.meta', 'rb') as pickle_file:
            meta = pickle.load(pickle_file)
        model.threshold = meta['threshold']
    else:
        threshold = model.fit(x_train, epochs=EPOCHS)
        with open(model_name + '.meta', 'wb') as pickle_file:
            pickle.dump({'threshold': threshold}, pickle_file)

    print('Test validation:')
    file_name = f'validation{MODEL_PATH}_ws{window_size}_ep{EPOCHS}_pct{PERCENTILE}_bn{bottleneck_size}_ec{encoder_sizer}'
    file_name = os.path.join('results', file_name)
    precision, recall, f1 = model.evaluate(x_test, y_test, file_name + '.png')
    with open(file_name + '.txt', 'w') as result_file:
        result_file.write(f'{precision=}, {recall=}, {f1=}')
    print(f'{precision=}, {recall=}, {f1=}')
