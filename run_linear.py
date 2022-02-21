from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from utils.data_utils import load_data_cross_validation
from model.fl_model import VerticalFLModel
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel
from model.simple_fl_model import PCAVerticalFLModel
from datetime import datetime
from utils.split_train_test import split_train_test
from utils.data_utils import load_data_train_test
from model.fl_model import VerticalFLModel
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel

from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

import os.path
import wget
import bz2
import shutil
import numpy as np

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from joblib import Parallel, delayed
import argparse



num_parties = 1
x_scaler_wrapper = []
y_scaler_wrapper = []
x_normalizer_wrapper = []
cross_valid_data = load_data_cross_validation("covtype.binary.train", num_parties=num_parties,
                                              file_type='libsvm', n_fold=5, use_cache=False,
                                              x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                              x_normalizer_wrapper=x_normalizer_wrapper)
xs_test, y_test = load_data_cross_validation("covtype.binary.test", num_parties=num_parties,
                                             file_type='libsvm', n_fold=1, use_cache=False,
                                             x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                             x_normalizer_wrapper=x_normalizer_wrapper)[0]
xs_train, y_train, _, _ = cross_valid_data[0]
classifier = LogisticRegression(max_iter=1000)
classifier.fit(xs_train[0], y_train)
acc = classifier.score(xs_test[0], y_test)
print("LogisticReg on covtype, acc = {}".format(acc))



num_parties = 1
x_scaler_wrapper = []
y_scaler_wrapper = []
x_normalizer_wrapper = []
cross_valid_data = load_data_cross_validation("gisette_scale.train", num_parties=num_parties,
                                              file_type='libsvm', n_fold=5, use_cache=False,
                                              x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                              x_normalizer_wrapper=x_normalizer_wrapper)
xs_test, y_test = load_data_cross_validation("gisette_scale.test", num_parties=num_parties,
                                             file_type='libsvm', n_fold=1, use_cache=False,
                                             x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                             x_normalizer_wrapper=x_normalizer_wrapper)[0]
xs_train, y_train, _, _ = cross_valid_data[0]
classifier = LogisticRegression(max_iter=1000)
classifier.fit(xs_train[0], y_train)
acc = classifier.score(xs_test[0], y_test)
print("LogisticReg on gisette, acc = {}".format(acc))


num_parties = 1
x_scaler_wrapper = []
y_scaler_wrapper = []
x_normalizer_wrapper = []
cross_valid_data = load_data_cross_validation("phishing.train", num_parties=num_parties,
                                              file_type='libsvm', n_fold=5, use_cache=False,
                                              x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                              x_normalizer_wrapper=x_normalizer_wrapper)
xs_test, y_test = load_data_cross_validation("phishing.test", num_parties=num_parties,
                                             file_type='libsvm', n_fold=1, use_cache=False,
                                             x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                             x_normalizer_wrapper=x_normalizer_wrapper)[0]
xs_train, y_train, _, _ = cross_valid_data[0]
classifier = LogisticRegression(max_iter=1000)
classifier.fit(xs_train[0], y_train)
acc = classifier.score(xs_test[0], y_test)
print("RidgeCls on phishing, acc = {}".format(acc))




num_parties = 1
xs_train_val, y_train_val, xs_test, y_test, _, _ = load_data_train_test("mnist", "mnist",
                                                                        num_parties=num_parties, file_type='torch')
train_idx, val_idx = next(KFold(n_splits=5, shuffle=True).split(y_train_val))
x_train = [data[train_idx] for data in xs_train_val][0]
y_train = y_train_val[train_idx]
x_val = [data[val_idx] for data in xs_train_val][0]
y_val = y_train_val[val_idx]
x_train = x_train.reshape(x_train.shape[0], -1)
x_val = x_val.reshape(x_val.shape[0], -1)
x_test = xs_test[0].reshape(xs_test[0].shape[0], -1)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("LogisticReg on MNIST, acc = {}".format(acc))


num_parties = 1
xs_train_val, y_train_val, xs_test, y_test, _, _ = load_data_train_test("kmnist", "kmnist",
                                                                        num_parties=num_parties, file_type='torch')
train_idx, val_idx = next(KFold(n_splits=5, shuffle=True).split(y_train_val))
x_train = [data[train_idx] for data in xs_train_val][0]
y_train = y_train_val[train_idx]
x_val = [data[val_idx] for data in xs_train_val][0]
y_val = y_train_val[val_idx]
x_train = x_train.reshape(x_train.shape[0], -1)
x_val = x_val.reshape(x_val.shape[0], -1)
x_test = xs_test[0].reshape(xs_test[0].shape[0], -1)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("LogisticReg on KMNIST, acc = {}".format(acc))


num_parties = 1
xs_train_val, y_train_val, xs_test, y_test, _, _ = load_data_train_test("fashion_mnist", "fashion_mnist",
                                                                        num_parties=num_parties, file_type='torch')
train_idx, val_idx = next(KFold(n_splits=5, shuffle=True).split(y_train_val))
x_train = [data[train_idx] for data in xs_train_val][0]
y_train = y_train_val[train_idx]
x_val = [data[val_idx] for data in xs_train_val][0]
y_val = y_train_val[val_idx]
x_train = x_train.reshape(x_train.shape[0], -1)
x_val = x_val.reshape(x_val.shape[0], -1)
x_test = xs_test[0].reshape(xs_test[0].shape[0], -1)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("LogisticReg on Fashion MNIST, acc = {}".format(acc))



num_parties = 1
x_scaler_wrapper = []
y_scaler_wrapper = []
x_normalizer_wrapper = []
cross_valid_data = load_data_cross_validation("UJIndoorLoc/trainingData.csv.train", num_parties=num_parties,
                                              file_type='csv', n_fold=5, use_cache=False,
                                              x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                              x_normalizer_wrapper=x_normalizer_wrapper)
xs_test, y_test = load_data_cross_validation("UJIndoorLoc/trainingData.csv.test", num_parties=num_parties,
                                             file_type='csv', n_fold=1, use_cache=False,
                                             x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                             x_normalizer_wrapper=x_normalizer_wrapper)[0]
xs_train, y_train, _, _ = cross_valid_data[0]
model = Ridge()
model.fit(xs_train[0], y_train)
y_predict = model.predict(xs_test[0])
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print("Ridge on UJ, RMSE = {}".format(rmse))


num_parties = 1
x_scaler_wrapper = []
y_scaler_wrapper = []
x_normalizer_wrapper = []
cross_valid_data = load_data_cross_validation("superconduct/train.csv.train", num_parties=num_parties,
                                              file_type='csv', n_fold=5, use_cache=False,
                                              x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                              x_normalizer_wrapper=x_normalizer_wrapper)
xs_test, y_test = load_data_cross_validation("superconduct/train.csv.test", num_parties=num_parties,
                                             file_type='csv', n_fold=1, use_cache=False,
                                             x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                             x_normalizer_wrapper=x_normalizer_wrapper)[0]
xs_train, y_train, _, _ = cross_valid_data[0]
model = Ridge()
model.fit(xs_train[0], y_train)
y_predict = model.predict(xs_test[0])
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print("Ridge on superconduct, RMSE = {}".format(rmse))