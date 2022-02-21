from utils.data_utils import load_data_cross_validation
from model.fl_model import VerticalFLModel
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

import os.path
import wget
import bz2
import shutil
import zipfile
import numpy as np

import xgboost as xgb
from utils.split_train_test import split_train_test
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0, help='Index of GPU')
args = parser.parse_args()


if not os.path.isdir("data"):
    os.mkdir("data")
if not (os.path.isdir("data/UJIndoorLoc") and os.path.isfile("data/UJIndoorLoc/trainingData.csv")):
    print("Downloading UJ data")
    wget.download("https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIndoorLoc.zip",
                  "data/UJIndoorLoc.zip")
    with zipfile.ZipFile("data/UJIndoorLoc.zip", 'r') as zip_ref:
        zip_ref.extractall("data/")

if not os.path.isfile("data/UJIndoorLoc/trainingData.csv.train") or \
        not os.path.isfile("data/UJIndoorLoc/trainingData.csv.test"):
    split_train_test("UJIndoorLoc/trainingData.csv", file_type='csv', test_rate=0.1)


################################## FedOnce-L1 #########################################
#
num_parties = 4
eps = 2
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
rmse_summary = []
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    rmse_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "dp_fedonce_uj_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=10,
            num_local_rounds=10,
            local_lr=0.4,
            local_hidden_layers=[50, 50],
            local_batch_size=128,
            local_weight_decay=0,
            local_output_size=6,
            num_agg_rounds=1,
            agg_lr=0.5,
            agg_hidden_layers=[20, 20],
            agg_batch_size=128,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.5,
            inter_party_comp_method="MA",
            delta=1e-5
        )
        _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        print("Active party {} finished training.".format(active_party))
        y_test_score = aggregate_model.predict_agg(xs_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_score))
        print("Test set RMSE={}".format(test_rmse))
        rmse_list.append(test_rmse)
        print(aggregate_model.params)
    rmse_summary.append(rmse_list)
    print("RMSE for active party {}".format(active_party) + str(rmse_list))
    print("-------------------------------------------------")
print("RMSE summary: " + repr(rmse_summary))
for i, result in enumerate(rmse_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: RMSE mean={}, std={}".format(i, mean, std))



num_parties = 4
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
rmse_summary = []
eps = 4
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    rmse_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "dp_fedonce_uj_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=20,
            num_local_rounds=12,
            local_lr=0.4,
            local_hidden_layers=[50, 50],
            local_batch_size=128,
            local_weight_decay=0,
            local_output_size=6,
            num_agg_rounds=1,
            agg_lr=0.5,
            agg_hidden_layers=[20, 20],
            agg_batch_size=128,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.5,
            inter_party_comp_method="MA",
            delta=1e-5
        )
        _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        print("Active party {} finished training.".format(active_party))
        y_test_score = aggregate_model.predict_agg(xs_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_score))
        print("Test set RMSE={}".format(test_rmse))
        rmse_list.append(test_rmse)
        print(aggregate_model.params)
    rmse_summary.append(rmse_list)
    print("RMSE for active party {}".format(active_party) + str(rmse_list))
    print("-------------------------------------------------")
print("RMSE summary: " + repr(rmse_summary))
for i, result in enumerate(rmse_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: RMSE mean={}, std={}".format(i, mean, std))





num_parties = 4
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
rmse_summary = []
eps = 6
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    rmse_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "dp_fedonce_uj_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=30,
            num_local_rounds=12,
            local_lr=0.4,
            local_hidden_layers=[50, 50],
            local_batch_size=128,
            local_weight_decay=0,
            local_output_size=6,
            num_agg_rounds=1,
            agg_lr=0.5,
            agg_hidden_layers=[20, 20],
            agg_batch_size=128,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.5,
            inter_party_comp_method="MA",
            delta=1e-5
        )
        _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        print("Active party {} finished training.".format(active_party))
        y_test_score = aggregate_model.predict_agg(xs_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_score))
        print("Test set RMSE={}".format(test_rmse))
        rmse_list.append(test_rmse)
        print(aggregate_model.params)
    rmse_summary.append(rmse_list)
    print("RMSE for active party {}".format(active_party) + str(rmse_list))
    print("-------------------------------------------------")
print("RMSE summary: " + repr(rmse_summary))
for i, result in enumerate(rmse_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: RMSE mean={}, std={}".format(i, mean, std))




num_parties = 4
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
rmse_summary = []
eps = 8
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    rmse_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "dp_fedonce_uj_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=60,
            num_local_rounds=12,
            local_lr=0.4,
            local_hidden_layers=[50, 50],
            local_batch_size=128,
            local_weight_decay=0,
            local_output_size=6,
            num_agg_rounds=1,
            agg_lr=0.5,
            agg_hidden_layers=[20, 20],
            agg_batch_size=128,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.5,
            inter_party_comp_method="MA",
            delta=1e-5
        )
        _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        print("Active party {} finished training.".format(active_party))
        y_test_score = aggregate_model.predict_agg(xs_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_score))
        print("Test set RMSE={}".format(test_rmse))
        rmse_list.append(test_rmse)
        print(aggregate_model.params)
    rmse_summary.append(rmse_list)
    print("RMSE for active party {}".format(active_party) + str(rmse_list))
    print("-------------------------------------------------")
print("RMSE summary: " + repr(rmse_summary))
for i, result in enumerate(rmse_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: RMSE mean={}, std={}".format(i, mean, std))


###################################### Priv-Baseline #########################################


num_parties = 4
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
rmse_summary = []
eps = 2
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    rmse_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "dp_simple_fedonce_uj_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=10,
            num_local_rounds=10,
            local_lr=0.4,
            local_hidden_layers=[50, 50],
            local_batch_size=128,
            local_weight_decay=0,
            local_output_size=6,
            num_agg_rounds=1,
            agg_lr=0.5,
            agg_hidden_layers=[20, 20],
            agg_batch_size=128,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.5,
            inter_party_comp_method="simple",
            delta=1e-5
        )
        _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        print("Active party {} finished training.".format(active_party))
        y_test_score = aggregate_model.predict_agg(xs_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_score))
        print("Test set RMSE={}".format(test_rmse))
        rmse_list.append(test_rmse)
        print(aggregate_model.params)
    rmse_summary.append(rmse_list)
    print("RMSE for active party {}".format(active_party) + str(rmse_list))
    print("-------------------------------------------------")
print("RMSE summary: " + repr(rmse_summary))
for i, result in enumerate(rmse_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: RMSE mean={}, std={}".format(i, mean, std))



num_parties = 4
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
rmse_summary = []
eps = 4
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    rmse_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "dp_simple_fedonce_uj_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=20,
            num_local_rounds=12,
            local_lr=0.4,
            local_hidden_layers=[50, 50],
            local_batch_size=128,
            local_weight_decay=0,
            local_output_size=6,
            num_agg_rounds=1,
            agg_lr=0.5,
            agg_hidden_layers=[20, 20],
            agg_batch_size=128,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.5,
            inter_party_comp_method="simple",
            delta=1e-5
        )
        _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        print("Active party {} finished training.".format(active_party))
        y_test_score = aggregate_model.predict_agg(xs_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_score))
        print("Test set RMSE={}".format(test_rmse))
        rmse_list.append(test_rmse)
        print(aggregate_model.params)
    rmse_summary.append(rmse_list)
    print("RMSE for active party {}".format(active_party) + str(rmse_list))
    print("-------------------------------------------------")
print("RMSE summary: " + repr(rmse_summary))
for i, result in enumerate(rmse_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: RMSE mean={}, std={}".format(i, mean, std))




num_parties = 4
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
rmse_summary = []
eps = 6
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    rmse_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "dp_simple_fedonce_uj_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=30,
            num_local_rounds=12,
            local_lr=0.4,
            local_hidden_layers=[50, 50],
            local_batch_size=128,
            local_weight_decay=0,
            local_output_size=6,
            num_agg_rounds=1,
            agg_lr=0.5,
            agg_hidden_layers=[20, 20],
            agg_batch_size=128,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.5,
            inter_party_comp_method="simple",
            delta=1e-5
        )
        _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        print("Active party {} finished training.".format(active_party))
        y_test_score = aggregate_model.predict_agg(xs_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_score))
        print("Test set RMSE={}".format(test_rmse))
        rmse_list.append(test_rmse)
        print(aggregate_model.params)
    rmse_summary.append(rmse_list)
    print("RMSE for active party {}".format(active_party) + str(rmse_list))
    print("-------------------------------------------------")
print("RMSE summary: " + repr(rmse_summary))
for i, result in enumerate(rmse_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: RMSE mean={}, std={}".format(i, mean, std))




num_parties = 4
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
rmse_summary = []
eps = 8
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    rmse_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "dp_simple_fedonce_uj_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=30,
            num_local_rounds=12,
            local_lr=0.4,
            local_hidden_layers=[50, 50],
            local_batch_size=128,
            local_weight_decay=0,
            local_output_size=6,
            num_agg_rounds=1,
            agg_lr=0.6,
            agg_hidden_layers=[20, 20],
            agg_batch_size=128,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.5,
            inter_party_comp_method="simple",
            delta=1e-5
        )
        _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        print("Active party {} finished training.".format(active_party))
        y_test_score = aggregate_model.predict_agg(xs_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_score))
        print("Test set RMSE={}".format(test_rmse))
        rmse_list.append(test_rmse)
        print(aggregate_model.params)
    rmse_summary.append(rmse_list)
    print("RMSE for active party {}".format(active_party) + str(rmse_list))
    print("-------------------------------------------------")
print("RMSE summary: " + repr(rmse_summary))
for i, result in enumerate(rmse_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: RMSE mean={}, std={}".format(i, mean, std))


############################# FedOnce-L0 ################################

num_parties = 4
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
rmse_summary = []
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    rmse_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "fedonce_uj_party_{}_fold_{}".format(num_parties, i)
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
            local_hidden_layers=[50, 50, 50, 30],
            local_batch_size=100,
            local_weight_decay=1e-4,
            local_output_size=3,
            num_agg_rounds=1,
            agg_lr=1e-4,
            agg_hidden_layers=[30, 10],
            agg_batch_size=100,
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
            num_workers=8
        )
        _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        print("Active party {} finished training.".format(active_party))
        y_test_score = aggregate_model.predict_agg(xs_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_score))
        print("Test set RMSE={}".format(test_rmse))
        rmse_list.append(test_rmse)
        print(aggregate_model.params)

    rmse_summary.append(rmse_list)
    print("RMSE for active party {}".format(active_party) + str(rmse_list))
    print("-------------------------------------------------")
print("RMSE summary: " + repr(rmse_summary))
for i, result in enumerate(rmse_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: RMSE mean={}, std={}".format(i, mean, std))



############33 FedOnce-L0 w/ same model

num_parties = 4
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
rmse_summary = []
eps = 8
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    rmse_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "dp_simple_fedonce_uj_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=600,
            num_local_rounds=500,
            local_lr=3e-4,
            local_hidden_layers=[50, 50],
            local_batch_size=128,
            local_weight_decay=1e-4,
            local_output_size=6,
            num_agg_rounds=1,
            agg_lr=1e-4,
            agg_hidden_layers=[20, 20],
            agg_batch_size=128,
            agg_weight_decay=1e-4,
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
            momentum=0.9,
            privacy=None,
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.5,
            inter_party_comp_method="simple",
            delta=1e-5
        )
        _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        print("Active party {} finished training.".format(active_party))
        y_test_score = aggregate_model.predict_agg(xs_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_score))
        print("Test set RMSE={}".format(test_rmse))
        rmse_list.append(test_rmse)
        print(aggregate_model.params)
    rmse_summary.append(rmse_list)
    print("RMSE for active party {}".format(active_party) + str(rmse_list))
    print("-------------------------------------------------")
print("RMSE summary: " + repr(rmse_summary))
for i, result in enumerate(rmse_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: RMSE mean={}, std={}".format(i, mean, std))