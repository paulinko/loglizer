#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import Autoencoder
from loglizer import dataloader, preprocessing
import numpy as np

struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='uniform',
                                                                zero_positive=True)
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting=None,
                                              normalization='zero-mean')
    # x_train = np.delete(x_train, x_train.argmax())
    x_test = feature_extractor.transform(x_test)

    model = Autoencoder(input_size=x_test.shape[1],
                        bottleneck_size=int(x_test.shape[1] // 1.),
                        encoder_size=x_test.shape[1] * 5,
                        )
    threshold = model.fit(x_train, epochs=20)

    print('Train validation:')
    # precision, recall, f1 = model.evaluate(x_train, y_train)
    
    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)
    print(f'{precision=}, {recall=}, {f1=}')


