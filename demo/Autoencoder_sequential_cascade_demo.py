#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import sys
from datetime import datetime

import pandas
import torch

sys.path.append('../')
from loglizer.models import Autoencoder, AutoencoderCascade
from loglizer import dataloader, preprocessing
import pickle
import numpy as np
import os
import time

struct_log = '../data/HDFS/HDFS.npz'  # The structured log file
# struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured
label_file = '../data/HDFS/anomaly_label.csv'  # The anomaly label file
typ = ''
if struct_log.endswith('.npz'):
    typ = '_npz'

window_size = 10
generate = True
MODEL_PATH = 'log_autoencoder_seq_casc'
EPOCHS = 50
PERCENTILE = 0.9999
lr = 1e-3
decay = 1e-5
load = True and not generate
ratio = 0.01

batch_size = 0

#
if __name__ == '__main__':
    if generate:
        (x_train, _, y_train), (x_test, _, y_test) = dataloader.load_HDFS(struct_log,
                                                                          label_file=label_file,
                                                                          window_size=window_size,
                                                                          train_ratio=ratio,
                                                                          split_type='uniform',
                                                                          zero_positive=True)
        with open(f'WIP_ws{window_size}{typ}', 'wb') as pickle_file:
            pickle.dump({'train': (x_train, y_train), 'test': (x_test, y_test)}, pickle_file)
    else:
        with open(f'WIP_ws{window_size}{typ}', 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            x_train, y_train = data['train']
            x_test, y_test = data['test']

    feature_extractor = preprocessing.Vectorizer()
    unique_events = np.unique(np.concatenate(x_train['EventSequence']))
    train_dataset = feature_extractor.fit_transform(x_train, pandas.Series([], dtype=pandas.Float64Dtype), y_train,
                                                    normalize=True, use_onehot=False)
    x_train, y_train = train_dataset['x'], train_dataset['y']
    test_dataset = feature_extractor.transform(x_test, pandas.Series([], dtype=pandas.Float64Dtype), y_test)
    x_test, y_test = test_dataset['x'], test_dataset['y']

    bottleneck_size = int(x_test.shape[1] // 2.7)
    encoder_sizer = int(x_test.shape[1] * 3)
    model_name = f'{MODEL_PATH}_ws{window_size}_epoch{EPOCHS}'
    model = AutoencoderCascade(input_size=x_test.shape[1],
                        bottleneck_size=bottleneck_size,
                        encoder_size=encoder_sizer,
                        learning_rate=lr,
                        decay=decay,
                        percentile=PERCENTILE,
                        model_name=model_name
                        )

    if load:
        model.load_state_dict(torch.load(f'{model_name}.pth'))
        with open(model_name + '.meta', 'rb') as pickle_file:
            meta = pickle.load(pickle_file)
        model.threshold = meta['threshold']
        train_time = 'NA'
    else:
        file_name = f'train_{MODEL_PATH}_ws{window_size}_ep{EPOCHS}_pct{PERCENTILE}_bn{bottleneck_size}_ec{encoder_sizer}_d{decay}_lr{lr}{typ}_bs{batch_size}_r{ratio}'
        file_name = os.path.join('results', file_name)
        train_start = time.time()
        threshold = model.fit(x_train, epochs=EPOCHS, file_name=file_name, batch_size=batch_size)
        train_time = time.time() - train_start
        with open(model_name + '.meta', 'wb') as pickle_file:
            pickle.dump({'threshold': threshold}, pickle_file)

    print('Test validation:')
    file_name = f'validation{MODEL_PATH}_ws{window_size}_ep{EPOCHS}_pct{PERCENTILE}_bn{bottleneck_size}_ec{encoder_sizer}_d{decay}_lr{lr}{typ}_bs{batch_size}_r{ratio}'
    file_name = os.path.join('results', file_name)
    eval_start = time.time()
    precision, recall, f1 = model.evaluate(x_test, y_test, file_name + '.png', test_dataset['SessionId'], sample=0.1)
    eval_time = time.time() - eval_start
    with open(file_name + '.txt', 'w') as result_file:
        result_file.write(f'{precision=}, {recall=}, {f1=}')
    print(f'{precision=}, {recall=}, {f1=}')
    print('result_file', file_name)

    log_name = os.path.join('important', 'result_log.csv')

    if not os.path.isfile(log_name):
        with open(log_name, 'w') as result_log:
            w = csv.writer(result_log)
            w.writerow(['time', 'architecture','percentile','lr', 'decay', 'bn','epochs', 'encoder_size', 'window_size', 'total_val', 'total_train', 'precision', 'recall', 'f1', 'result_file', 'eval_time', 'train_time'])
    with open(log_name, 'a') as result_log:
        w = csv.writer(result_log)
        time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        w.writerow(
            [time, MODEL_PATH, PERCENTILE, lr, decay, bottleneck_size, EPOCHS, encoder_sizer, window_size, len(x_test),len(x_train), precision, recall,
             f1, file_name, eval_time, train_time])

