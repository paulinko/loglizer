#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time

sys.path.append('../')
import pandas as pd
from loglizer.models import *
from loglizer import dataloader, preprocessing

# run_models = ['AutoencoderCascade']
run_models = ['InvariantsMiner']

struct_log = '../data/HDFS/HDFS.npz' # The benchmark dataset

EPOCHS = 10
PERCENTILE = 0.97
decay=1e-4
lr=1e-3

if __name__ == '__main__':
    (x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS(struct_log,
                                                           window='session', 
                                                           train_ratio=0.05,
                                                           zero_positive=False,
                                                           split_type='uniform')
    benchmark_results = []
    for _model in run_models:
        print('Evaluating {} on HDFS:'.format(_model))
        if _model == 'PCA':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf', 
                                                      normalization='zero-mean')
            model = PCA()
            model.fit(x_train)

        elif _model == 'Autoencoder':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting=None,
                                              normalization='zero-mean')
            bottleneck_size = int(x_train.shape[1] // 1)
            encoder_sizer = x_train.shape[1] * 5
            model = Autoencoder(input_size=x_train.shape[1],
                                bottleneck_size=bottleneck_size,
                                encoder_size=encoder_sizer,
                                percentile=PERCENTILE,
                                learning_rate=lr,
                                decay=decay
                                )
            model.fit(x_train, epochs=EPOCHS)
        elif _model == 'AutoencoderCascade':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting=None,
                                              normalization='zero-mean')
            bottleneck_size = int(x_train.shape[1] // 1)
            encoder_sizer = x_train.shape[1] * 5
            model = AutoencoderCascade(input_size=x_train.shape[1],
                                bottleneck_size=bottleneck_size,
                                encoder_size=encoder_sizer,
                                percentile=PERCENTILE,
                                learning_rate=lr,
                                decay=decay
                                )
            model.fit(x_train, epochs=EPOCHS)


        elif _model == 'InvariantsMiner':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr)
            model = InvariantsMiner(epsilon=0.5, percentage=0.99)
            train_start = time.time()
            model.fit(x_train)
            train_time  = time.time() - train_start
            print(f'{train_time=}')

        elif _model == 'LogClustering':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = LogClustering(max_dist=0.3, anomaly_threshold=0.3)
            model.fit(x_train[y_train == 0, :]) # Use only normal samples for training

        elif _model == 'IsolationForest':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr)
            model = IsolationForest(random_state=2019, max_samples=0.9999, contamination=0.03, 
                                    n_jobs=4)
            model.fit(x_train)

        elif _model == 'LR':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = LR()
            model.fit(x_train, y_train)

        elif _model == 'SVM':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = SVM()
            model.fit(x_train, y_train)

        elif _model == 'DecisionTree':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = DecisionTree()
            model.fit(x_train, y_train)
        
        x_test = feature_extractor.transform(x_te)
        print('Train accuracy:')
        precision, recall, f1 = model.evaluate(x_train, y_train)
        benchmark_results.append([_model + '-train', precision, recall, f1])
        print('Test accuracy:')
        # precision, recall, f1 = model.evaluate(x_test, y_test, file_name='benchmark_eval_result',sample=0.1)

        evaluation_start = time.time()
        precision, recall, f1 = model.evaluate(x_test, y_test)
        evaluation_time = time.time() - evaluation_start
        print(f'{evaluation_time=}')
        benchmark_results.append([_model + '-test', precision, recall, f1])

    pd.DataFrame(benchmark_results, columns=['Model', 'Precision', 'Recall', 'F1']) \
      .to_csv('benchmark_result.csv', index=False)
