#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from datetime import datetime

sys.path.append('../')
from loglizer.models import Autoencoder
from loglizer import dataloader, preprocessing
import numpy as np
import os
import csv
import time

struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file
MODEL_PATH = 'quantitative_autoencoder'
EPOCHS = 10
PERCENTILE = 0.97
decay=1e-4
lr=1e-3


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='uniform',
                                                                zero_positive=True)
    normalization = 'positive-mean'
    # normalization = 'zero-mean'
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting=None,
                                              normalization=normalization)
    # x_train = np.delete(x_train, x_train.argmax())
    x_test = feature_extractor.transform(x_test)

    bottleneck_size = 5 # int(x_test.shape[1] // )
    encoder_sizer = x_test.shape[1]
    model_name = f'{MODEL_PATH}_{normalization}_ws_epoch{EPOCHS}'
    model = Autoencoder(input_size=x_test.shape[1],
                        bottleneck_size=bottleneck_size,
                        encoder_size=encoder_sizer,
                        percentile=PERCENTILE,
                        learning_rate=lr,
                        decay=decay
                        )
    train_start = time.time()
    threshold = model.fit(x_train, epochs=EPOCHS)
    train_time = time.time() - train_start

    # print('Train validation:')
    # # precision, recall, f1 = model.evaluate(x_train, y_train)
    #
    # print('Test validation:')
    # precision, recall, f1 = model.evaluate(x_test, y_test)
    # print(f'{precision=}, {recall=}, {f1=}')

    print('Test validation:')
    file_name = f'validation{MODEL_PATH}_{normalization}_ep{EPOCHS}_pct{PERCENTILE}_bn{bottleneck_size}_ec{encoder_sizer}_d{decay}_lr{lr}'
    file_name = os.path.join('results', file_name)

    evaluation_start = time.time()
    precision, recall, f1 = model.evaluate(x_test, y_test, file_name + '.png')
    eval_time = time.time() - evaluation_start
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
            [time, MODEL_PATH, PERCENTILE, lr, decay, bottleneck_size, EPOCHS, encoder_sizer, 'session', len(x_test),len(x_train), precision, recall,
             f1, file_name, eval_time, train_time])





