#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import sys
from datetime import datetime
import time

import pandas
import torch

sys.path.append('../')
from loglizer.models import AutoencoderLSTM
from loglizer import dataloader, preprocessing
import pickle
import numpy as np
import os

struct_log = '../data/HDFS/HDFS_100k.log_structured.csv'  # The structured log file
label_file = '../data/HDFS/anomaly_label.csv'  # The anomaly label file

window_size = 10
generate = False
MODEL_PATH = 'log_autoencoderlstm'
EPOCHS = 5
PERCENTILE = 0.99
load = False and not generate
load_threshold = False
lr = 1e-1
decay = 1e-3

if __name__ == '__main__':
    if generate:
        (x_train, _, y_train), (x_test, _, y_test) = dataloader.load_HDFS(struct_log,
                                                                          label_file=label_file,
                                                                          window_size=window_size,
                                                                          train_ratio=0.1,
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
    model = AutoencoderLSTM(batch_size=x_test.shape[1],
                        bottleneck_size=bottleneck_size,
                        encoder_size=encoder_sizer,
                        learning_rate=lr,
                        decay=decay,
                        percentile=PERCENTILE,
                        num_directions=1,
                        model_name=model_name
                        )
    print(model)

    if load:
        model.load_state_dict(torch.load(f'{model_name}.pth'))
        if load_threshold:
            with open(model_name + '.meta', 'rb') as pickle_file:
                meta = pickle.load(pickle_file)
            model.threshold = meta['threshold']
        else:
            model.percentile = PERCENTILE
            model.calculate_threshold(train_loader=x_train)

        train_time = 0
    else:
        file_name = f'train_{MODEL_PATH}_ws{window_size}_ep{EPOCHS}_pct{PERCENTILE}_bn{bottleneck_size}_ec{encoder_sizer}_d{decay}_lr{lr}'
        file_name = os.path.join('results', file_name)
        start_train = time.time()
        threshold = model.fit(x_train, epochs=EPOCHS, file_name=file_name)
        train_time = time.time() - start_train
        with open(model_name + '.meta', 'wb') as pickle_file:
            pickle.dump({'threshold': threshold}, pickle_file)

    print('Test validation:')
    file_name = f'validation{MODEL_PATH}_ws{window_size}_ep{EPOCHS}_pct{PERCENTILE}_bn{bottleneck_size}_ec{encoder_sizer}_d{decay}_lr{lr}'
    file_name = os.path.join('results', file_name)
    start_eval = time.time()
    precision, recall, f1 = model.evaluate(x_test, y_test, file_name + '.png', test_dataset['SessionId'])
    eval_time = time.time() -start_eval
    with open(file_name + '.txt', 'w') as result_file:
        result_file.write(f'{precision=}, {recall=}, {f1=}')
        result_file.write(str(model))
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
            [time, MODEL_PATH, PERCENTILE, lr, decay, bottleneck_size, EPOCHS, encoder_sizer, 'session', len(x_test),len(x_train), precision, recall,
             f1, file_name, eval_time, train_time])


