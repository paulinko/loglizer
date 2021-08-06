#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time

sys.path.append('../')
import pandas as pd
from loglizer.models import *
from loglizer import dataloader, preprocessing

run_models = ['AutoencoderCascade']
# run_models = ['InvariantsMiner']

struct_log = '../data/HDFS/HDFS.npz' # The benchmark dataset

EPOCHS = 30
PERCENTILE = 0.994
decay=1e-4
lr=1e-3

window_size = 10
run_models = ['AutoencoderLSTM']

train_time=0
frac_train = 0.01
randomize = False

if __name__ == '__main__':
    if window_size == 0:
        (x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS(struct_log,
                                                               window='session',
                                                               train_ratio=frac_train,
                                                               zero_positive=True,
                                                               split_type='uniform',
                                                               randomize=randomize)
    else:
        (x_train, _, y_train), (x_test, _, y_test) = dataloader.load_HDFS(struct_log,
                                                                          window_size=window_size,
                                                                          train_ratio=frac_train,
                                                                          randomize=randomize,
                                                                          split_type='uniform',
                                                                          zero_positive=True)

    benchmark_results = []
    ts = time.time()
    for _model in run_models:
        evaluation_file = False
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
                                decay=decay,
                                cmap='OrRd'
                                )
            model.fit(x_train, epochs=EPOCHS)
            evaluation_file = True
        elif _model == 'AutoencoderCascade':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting=None,
                                              normalization='zero-mean')
            bottleneck_size = int(x_train.shape[1] // 1.5)
            encoder_sizer = x_train.shape[1] * 5
            model = AutoencoderCascade(input_size=x_train.shape[1],
                                bottleneck_size=bottleneck_size,
                                encoder_size=encoder_sizer,
                                percentile=PERCENTILE,
                                learning_rate=lr,
                                decay=decay,
                                cmap='coolwarm'
                                )
            model.fit(x_train, epochs=EPOCHS, anim_dir='/home/pauline/Documents/git/loglizer/anim')
            evaluation_file = True

        elif _model == 'AutoencoderLSTM':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting=None,
                                              normalization='zero-mean')
            bottleneck_size = int(x_train.shape[1] // 1.5)
            encoder_sizer = x_train.shape[1] * 5
            model = AutoencoderLSTM(batch_size=x_test.shape[1],
                                    bottleneck_size=bottleneck_size,
                                    encoder_size=encoder_sizer,
                                    learning_rate=lr,
                                    decay=decay,
                                    percentile=PERCENTILE,
                                    num_directions=1,
                                    model_name=model_name
                                    )
            model.fit(x_train, epochs=EPOCHS, anim_dir='/home/pauline/Documents/git/loglizer/anim')
            evaluation_file = True

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
        # precision, recall, f1 = model.evaluate(x_train, y_train)
        # benchmark_results.append([_model + '-train', precision, recall, f1])
        print('Test accuracy:')

        evaluation_start = time.time()
        if evaluation_file:
            precision, recall, f1 = model.evaluate(x_test, y_test, file_name=f'{ts}_{_model}_benchmark_eval_result.png',sample=0.1)
        else:
            precision, recall, f1 = model.evaluate(x_test, y_test)
        evaluation_time = time.time() - evaluation_start
        print(f'{evaluation_time=}')
        benchmark_results.append([_model + '-test', precision, recall, f1])


    pd.DataFrame(benchmark_results, columns=['Model', 'Precision', 'Recall', 'F1']) \
      .to_csv(f'{ts}_benchmark_result.csv', index=False)

    with open(f'{ts}_benchmark_result.csv', 'a') as result_file:
        result_file.write(f'{lr=}, {decay=}, {PERCENTILE=}, {frac_train} {train_time=} {randomize=} {evaluation_time=} {train_time=}')
        result_file.write(str(model))