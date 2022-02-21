from utils.data_utils import load_data_cross_validation
from model.fl_model import VerticalFLModel
from model.simple_fl_model import PCAVerticalFLModel
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel
from utils.split_train_test import split_train_test

import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

import os.path
import wget
import bz2
import shutil
import numpy as np
import xgboost as xgb
import argparse


torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0, help='Index of GPU')
parser.add_argument('--num-parties', '-p', type=int, default=10, help='Number of parties')
args = parser.parse_args()

if not os.path.isdir("data"):
    os.mkdir("data")
if not os.path.isfile("data/gisette_scale"):
    if not os.path.isfile("data/gisette_scale.bz2"):
        print("Downloading gisette data")
        wget.download("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
                      "data/gisette_scale.bz2")
    with bz2.BZ2File("data/gisette_scale.bz2") as f_read, open("data/gisette_scale", 'wb') as f_write:
        shutil.copyfileobj(f_read, f_write)

if not os.path.isfile("data/gisette_scale.train") or not os.path.isfile("data/gisette_scale.test"):
    split_train_test("gisette_scale", file_type='libsvm', test_rate=0.1)


# FedOnce
num_parties = args.num_parties
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
acc_summary = []
f1_summary = []
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    acc_list = []
    f1_list = []
    i = 0
    xs_train, y_train, xs_val, y_val = cross_valid_data[i]
    start = datetime.now()
    print("Cross Validation Fold {}".format(i))
    print("Active Party is {}".format(active_party))
    model_name = "fedonce_gisette_party_{}_fold_{}".format(num_parties, i)
    name = "{}_active_{}".format(model_name, active_party)
    writer = SummaryWriter("runs/{}".format(name))
    aggregate_model = VerticalFLModel(
        num_parties=num_parties,
        active_party_id=active_party,
        name=model_name,
        full_name=name,
        num_epochs=100,
        num_local_rounds=100,
        local_lr=1e-4,
        local_hidden_layers=[50, 50],
        local_batch_size=100,
        local_weight_decay=1e-5,
        local_output_size=3,
        num_agg_rounds=1,
        agg_lr=1e-4,
        agg_hidden_layers=[30, 30],
        agg_batch_size=100,
        agg_weight_decay=1e-4,
        writer=writer,
        device='cuda:{}'.format(args.gpu),
        update_target_freq=1,
        task='binary_classification',
        n_classes=10,
        test_batch_size=1000,
        test_freq=1,
        cuda_parallel=False,
        n_channels=1,
        model_type='fc',
        optimizer='adam',
        privacy=None,
        batches_per_lot=5,
        epsilon=1,
        delta=1.0 / xs_train[0].shape[0]
    )
    val_acc, val_f1, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, use_cache=True)
    print("Fold {} finished training.".format(i))
    y_score_test = aggregate_model.predict_agg(xs_test)
    y_pred_test = np.where(y_score_test > 0.5, 1, 0)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    acc_list.append(test_acc)
    f1_list.append(test_f1)
    print(aggregate_model.params)
    time_min = (datetime.now() - start).seconds / 60
    print("Time(min) {}: ".format(time_min))

    f1_summary.append(f1_list)
    acc_summary.append(acc_list)
    print("Active party {} finished training.".format(active_party))
    print("Accuracy for party {}".format(active_party) + str(acc_list))
    print("F1 score for party {}".format(active_party, str(f1_list)))
    print("-------------------------------------------------")

print("Accuracy summary: " + repr(acc_summary))
print("F1 score summary: " + repr(f1_summary))
for i, result in enumerate(f1_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: Accuracy mean={}, std={}".format(i, mean, std))
for i, result in enumerate(acc_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: F1-score mean={}, std={}".format(i, mean, std))


# Solo
num_parties = args.num_parties
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
acc_summary = []
f1_summary = []
for party_id in range(num_parties):
    print("Party {} starts training".format(party_id))
    acc_list = []
    f1_list = []
    i = 0
    xs_train, y_train, xs_val, y_val = cross_valid_data[i]
    print("Cross Validation Fold {}".format(i))
    name = "single_gisette_party_{}_single_{}_fold_{}".format(num_parties, party_id, i)
    writer = SummaryWriter("runs/{}".format(name))
    single_model = SingleParty(
        party_id=party_id,
        num_epochs=100,
        lr=1e-4,
        hidden_layers=[50, 50],
        batch_size=128,
        weight_decay=1e-4,
        writer=writer,
        device='cuda:{}'.format(args.gpu),
        task="binary_classification",
        n_classes=10,
        test_batch_size=1000,
        test_freq=1,
        n_channels=1,
        model_type='fc',
        optimizer='adam',
        cuda_parallel=False
    )
    x_train = xs_train[party_id]
    x_test = xs_test[party_id]
    x_val = xs_val[party_id]
    acc, f1, _, _ = single_model.train(x_train, y_train, x_val, y_val)
    y_predict_test, y_score_test = single_model.predict(x_test)
    test_acc = accuracy_score(y_test, y_predict_test)
    test_f1 = f1_score(y_test, y_predict_test)
    acc_list.append(test_acc)
    f1_list.append(test_f1)
    print(single_model.params)

    f1_summary.append(f1_list)
    acc_summary.append(acc_list)
    print("Accuracy for party {}".format(party_id) + str(acc_list))
    print("F1 score for party {}".format(party_id, str(f1_list)))
    print("-------------------------------------------------")
print("Accuracy summary: " + repr(acc_summary))
print("F1 score summary: " + repr(f1_summary))
for i, result in enumerate(acc_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: Accuracy mean={}, std={}".format(i, mean, std))
for i, result in enumerate(f1_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: F1-score mean={}, std={}".format(i, mean, std))


# SplitNN without DP
num_parties = args.num_parties
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
acc_summary = []
f1_summary = []
print("Starts training")
acc_list = []
f1_list = []
i = 0
xs_train, y_train, xs_val, y_val = cross_valid_data[i]
# for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
start = datetime.now()
print("Cross Validation Fold {}".format(i))
model_name = "splitnn_gisette_party_{}_fold_{}".format(num_parties, i)
name = "{}/".format(model_name)
writer = SummaryWriter("runs/{}".format(name))
aggregate_model = SplitNNModel(
    num_parties=num_parties,
    name=model_name,
    num_epochs=50,
    local_hidden_layers=[50, 50],
    local_output_size=3,
    lr=1e-4,
    agg_hidden_layers=[30, 30],
    batch_size=100,
    weight_decay=1e-5,
    writer=writer,
    device='cuda:{}'.format(args.gpu),
    update_target_freq=1,
    task='binary_classification',
    n_classes=10,
    test_batch_size=1000,
    test_freq=1,
    cuda_parallel=False,
    n_channels=1,
    model_type='fc',
    optimizer='adam',
    privacy=None,
    batches_per_lot=5,
    epsilon=1,
    delta=1.0 / xs_train[0].shape[0],
    num_workers=4
)
val_acc, val_f1, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, xs_test, y_test, use_cache=False)
print("Fold {} finished training.".format(i))
y_predict_test, y_score_test = aggregate_model.predict(xs_test)
test_acc = accuracy_score(y_test, y_predict_test)
test_f1 = f1_score(y_test, y_predict_test)
acc_list.append(test_acc)
f1_list.append(test_f1)
print(aggregate_model.params)
time_min = (datetime.now() - start).seconds / 60
print("Time(min) {}: ".format(time_min))

f1_summary.append(f1_list)
acc_summary.append(acc_list)
print("Finished training.")
print("Accuracy: " + str(acc_list))
print("F1 score: " + str(f1_list))
print("-------------------------------------------------")
print("Accuracy summary: " + repr(acc_summary))
print("F1 score summary: " + repr(f1_summary))
for i, result in enumerate(acc_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: Accuracy mean={}, std={}".format(i, mean, std))
for i, result in enumerate(f1_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: F1-score mean={}, std={}".format(i, mean, std))
