from utils.data_utils import load_data_cross_validation
from model.fl_model import VerticalFLModel
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel

from torch.utils.tensorboard import SummaryWriter

import os.path
import wget
import zipfile
import numpy as np
from datetime import datetime

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

from utils.split_train_test import split_train_test
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0, help='Index of GPU')
args = parser.parse_args()


if not os.path.isdir("data"):
    os.mkdir("data")
if not (os.path.isdir("data/superconduct") and os.path.isfile("data/superconduct/train.csv")):
    print("Downloading superconduct data")
    wget.download("https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip",
                  "data/superconduct.zip")
    os.mkdir("data/superconduct/")
    with zipfile.ZipFile("data/superconduct.zip", 'r') as zip_ref:
        zip_ref.extractall("data/superconduct/")


if not os.path.isfile("data/superconduct/train.csv.train") or \
        not os.path.isfile("data/superconduct/train.csv.test"):
    split_train_test("superconduct/train.csv", file_type='csv', test_rate=0.1)


# SecureBoost with DP (use XGBoost instead since SecureBoost is lossless)
num_parties = 10
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

print("Starts training XGBoost superconduct")
rmse_list = []
for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
    print("Cross Validation Fold {}".format(i))
    x_train = np.concatenate(xs_train, axis=1)
    x_val = np.concatenate(xs_val, axis=1)
    x_test = np.concatenate(xs_test, axis=1)
    xg_cls = xgb.XGBRegressor(learning_rate=0.1,
                              max_depth=6,
                              n_estimators=100,
                              reg_alpha=10,
                              verbosity=2)
    xg_cls.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='rmse')
    y_pred = xg_cls.predict(x_test)
    rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
print("Finished training.")
print("-------------------------------------------------")
print("RMSE: " + str(rmse_list))
mean = np.mean(rmse_list)
std = np.std(rmse_list)
print("RMSE mean={}, std={}".format(mean, std))


#
# SplitNN without DP
num_parties = 10
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
rmse_summary = []
print("Starts training")
rmse_list = []
for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
    start = datetime.now()
    print("Cross Validation Fold {}".format(i))
    model_name = "splitnn_superconduct_party_{}_fold_{}".format(num_parties, i)
    name = "{}/".format(model_name)
    writer = SummaryWriter("runs/{}".format(name))
    aggregate_model = SplitNNModel(
        num_parties=num_parties,
        name=model_name,
        num_epochs=600,
        local_hidden_layers=[30, 30],
        local_output_size=3,
        lr=1e-3,
        agg_hidden_layers=[30, 10],
        batch_size=100,
        weight_decay=1e-5,
        writer=writer,
        device='cuda:{}'.format(args.gpu),
        update_target_freq=1,
        task='regression',
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
        delta=1.0/xs_train[0].shape[0],
        num_workers=0
    )
    _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, xs_test, y_test, use_cache=False)
    print("Fold {} finished training.".format(i))
    y_pred_test, y_score_test = aggregate_model.predict(xs_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_score_test))
    rmse_list.append(test_rmse)
    print(aggregate_model.params)
    time_min = (datetime.now() - start).seconds / 60
    print("Time(min) {}: ".format(time_min))

rmse_summary.append(rmse_list)
print("Finished training.")
print("RMSE: " + str(rmse_summary))
print("-------------------------------------------------")
print("Accuracy summary: " + repr(rmse_summary))
for i, result in enumerate(rmse_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: Accuracy mean={}, std={}".format(i, mean, std))




# FedOnce
num_parties = 10
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
rmse_summary = []
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    rmse_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        start = datetime.now()
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "fedonce_superconduct_party_{}_fold_{}".format(num_parties, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            full_name=name,
            num_epochs=600,
            num_local_rounds=500,
            local_lr=3e-4,
            local_hidden_layers=[30, 30],
            local_batch_size=100,
            local_weight_decay=1e-4,
            local_output_size=3,
            num_agg_rounds=1,
            agg_lr=3e-4,
            agg_hidden_layers=[30, 10],
            agg_batch_size=128,
            agg_weight_decay=1e-4,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            update_target_freq=1,
            task='regression',
            n_classes=10,
            test_batch_size=1000,
            test_freq=5,
            cuda_parallel=False,
            n_channels=1,
            model_type='fc',
            optimizer='adam',
            privacy=None,
            batches_per_lot=5,
            epsilon=1,
            delta=1.0/xs_train[0].shape[0],
            num_workers=0
        )
        _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, use_cache=True)
        print("Active party {} finished training.".format(active_party))
        y_test_score = aggregate_model.predict_agg(xs_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_score))
        rmse_list.append(test_rmse)
        print(aggregate_model.params)
        time_min = (datetime.now() - start).seconds / 60
        print("Time(min) {}: ".format(time_min))
        break
    rmse_summary.append(rmse_list)
    print("RMSE for active party {}".format(active_party) + str(rmse_list))
    print("-------------------------------------------------")

print("RMSE summary: " + repr(rmse_summary))
for i, result in enumerate(rmse_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: RMSE mean={}, std={}".format(i, mean, std))


# Solo
num_parties = 10
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
rmse_summary = []
for party_id in range(num_parties):
    print("Party {} starts training".format(party_id))
    rmse_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        name = "single_superconduct_party_{}_single_{}_fold_{}".format(num_parties, party_id, i)
        writer = SummaryWriter("runs/{}".format(name))
        single_model = SingleParty(
            party_id=party_id,
            num_epochs=600,
            lr=1e-3,
            hidden_layers=[30, 30, 10],
            batch_size=128,
            weight_decay=1e-5,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            task="regression",
            n_classes=10,
            test_batch_size=1000,
            test_freq=5,
            n_channels=1,
            model_type='fc',
            optimizer='adam',
            cuda_parallel=False
        )
        x_train = xs_train[party_id]
        x_val = xs_val[party_id]
        x_test = xs_test[party_id]
        _, _, rmse, _ = single_model.train(x_train, y_train, x_val, y_val)
        y_pred_test, y_score_test = single_model.predict(x_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_score_test))
        rmse_list.append(test_rmse)
        print(single_model.params)

    rmse_summary.append(rmse_list)
    print("RMSE for party {}".format(party_id) + str(rmse_list))
    print("-------------------------------------------------")
print("RMSE summary: " + repr(rmse_summary))
for i, result in enumerate(rmse_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: RMSE mean={}, std={}".format(i, mean, std))


# combine
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
rmse_summary = []
for party_id in range(num_parties):
    print("Party {} starts training".format(party_id))
    rmse_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data, 1):
        print("Cross Validation Fold {}".format(i))
        name = "combine_superconduct_fold_{}".format(i)
        writer = SummaryWriter("runs/{}".format(name))
        single_model = SingleParty(
            party_id=party_id,
            num_epochs=1000,
            lr=3e-4,
            hidden_layers=[80, 80, 50],
            batch_size=128,
            weight_decay=1e-5,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            task="regression",
            n_classes=10,
            test_batch_size=1000,
            test_freq=1,
            n_channels=1,
            model_type='fc',
            optimizer='adam',
            cuda_parallel=False
        )
        x_train = xs_train[party_id]
        x_val = xs_val[party_id]
        x_test = xs_test[party_id]
        _, _, rmse, _ = single_model.train(x_train, y_train, x_val, y_val)
        y_pred_test, y_score_test = single_model.predict(x_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_score_test))
        rmse_list.append(test_rmse)
        print(single_model.params)

    rmse_summary.append(rmse_list)
    print("RMSE for party {}".format(party_id) + str(rmse_list))
    print("-------------------------------------------------")
print("RMSE summary: " + repr(rmse_summary))
for i, result in enumerate(rmse_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: RMSE mean={}, std={}".format(i, mean, std))




# # post FedOnce
num_parties = 10
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
rmse_summary = []
print("Starts training")
rmse_list = []
for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
    print("Cross Validation Fold {}".format(i))
    model_name = "post_fedonce_superconduct_party_{}_fold_{}".format(num_parties, i)
    name = "{}/".format(model_name)
    active_party = 0
    cache_local_name = "fedonce_superconduct_party_{}_fold_{}".format(num_parties, i)
    cache_agg_name = "{}_active_{}".format(cache_local_name, active_party)
    writer = SummaryWriter("runs/{}".format(name))
    aggregate_model = SplitNNModel(
        num_parties=num_parties,
        name=model_name,
        num_epochs=600,
        local_hidden_layers=[30, 30],
        local_output_size=3,
        lr=1e-3,
        agg_hidden_layers=[30, 10],
        batch_size=100,
        weight_decay=1e-5,
        writer=writer,
        device='cuda:{}'.format(args.gpu),
        update_target_freq=1,
        task='regression',
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
        delta=1.0/xs_train[0].shape[0],
        num_workers=2,
        cache_local_name=cache_local_name,
        cache_agg_name=cache_agg_name,
        active_party=active_party
    )
    _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, xs_test, y_test, use_cache=True)
    print("Fold {} finished training.".format(i))
    y_pred_test, y_score_test = aggregate_model.predict(xs_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_score_test))
    rmse_list.append(test_rmse)
    print(aggregate_model.params)

rmse_summary.append(rmse_list)
print("Finished training.")
print("RMSE: " + str(rmse_summary))
print("-------------------------------------------------")
print("Accuracy summary: " + repr(rmse_summary))
for i, result in enumerate(rmse_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: Accuracy mean={}, std={}".format(i, mean, std))
