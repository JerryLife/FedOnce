from utils.data_utils import load_data_cross_validation
from model.fl_model import VerticalFLModel
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
from utils.split_train_test import split_train_test

import os.path
import wget
import bz2
import shutil
import numpy as np

import xgboost as xgb
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0, help='Index of GPU')
args = parser.parse_args()

if not os.path.isdir("data"):
    os.mkdir("data")
if not os.path.isfile("data/covtype.binary"):
    if not os.path.isfile("data/covtype.binary.bz2"):
        print("Downloading covtype data")
        wget.download("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
                      "data/covtype.binary.bz2")
    with bz2.BZ2File("data/covtype.binary.bz2") as f_read, open("data/covtype.binary", 'wb') as f_write:
        shutil.copyfileobj(f_read, f_write)

if not os.path.isfile("data/covtype.binary.train") or not os.path.isfile("data/covtype.binary.test"):
    split_train_test("covtype.binary", file_type='libsvm', test_rate=0.1)

n_fold = 5

########################## FedOnce-L1 #################################################
# vertical FL with DP
num_parties = 4
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
acc_summary = []
f1_summary = []
eps = 2
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    acc_list = []
    f1_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "fedonce_dp_covtype_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=30,
            num_local_rounds=10,
            local_lr=0.3,
            local_hidden_layers=[50],
            local_batch_size=32,
            local_weight_decay=0,
            local_output_size=3,
            num_agg_rounds=1,
            agg_lr=0.3,
            agg_hidden_layers=[30],
            agg_batch_size=32,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.0,
            inter_party_comp_method="MA",
            delta=1e-5,
            num_workers=4
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


# vertical FL with DP
num_parties = 4
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
acc_summary = []
f1_summary = []
eps = 4
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    acc_list = []
    f1_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "fedonce_dp_covtype_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=30,
            num_local_rounds=6,
            local_lr=0.2,
            local_hidden_layers=[50],
            local_batch_size=32,
            local_weight_decay=0,
            local_output_size=3,
            num_agg_rounds=1,
            agg_lr=0.2,
            agg_hidden_layers=[30],
            agg_batch_size=32,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.0,
            inter_party_comp_method="MA",
            delta=1e-5,
            num_workers=4
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


# vertical FL with DP
num_parties = 4
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
acc_summary = []
f1_summary = []
eps = 6
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    acc_list = []
    f1_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        version = 'v1'
        print("Active Party is {}".format(active_party))
        model_name = "fedonce_dp_covtype_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=30,
            num_local_rounds=10,
            local_lr=0.3,
            local_hidden_layers=[50],
            local_batch_size=128,
            local_weight_decay=0,
            local_output_size=3,
            num_agg_rounds=1,
            agg_lr=0.3,
            agg_hidden_layers=[30],
            agg_batch_size=128,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.5,
            inter_party_comp_method="MA",
            delta=1e-5,
            num_workers=4
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



# vertical FL with DP
num_parties = 4
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
acc_summary = []
f1_summary = []
eps = 8
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    acc_list = []
    f1_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        version = 'v1'
        print("Active Party is {}".format(active_party))
        model_name = "fedonce_dp_covtype_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=40,
            num_local_rounds=10,
            local_lr=0.3,
            local_hidden_layers=[50],
            local_batch_size=128,
            local_weight_decay=0,
            local_output_size=3,
            num_agg_rounds=1,
            agg_lr=0.6,
            agg_hidden_layers=[30],
            agg_batch_size=128,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.5,
            inter_party_comp_method="MA",
            delta=1e-5,
            num_workers=4
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

########################## Priv-Baseline ##################################
print("Priv-Baseline")

# vertical FL with DP
num_parties = 4
x_scaler_wrapper = []
y_scaler_wrapper = []
x_normalizer_wrapper = []
cross_valid_data = load_data_cross_validation("covtype.binary.train", num_parties=num_parties,
                                              file_type='libsvm', n_fold=n_fold, use_cache=False,
                                              x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                              x_normalizer_wrapper=x_normalizer_wrapper)
xs_test, y_test = load_data_cross_validation("covtype.binary.test", num_parties=num_parties,
                                             file_type='libsvm', n_fold=1, use_cache=False,
                                             x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                             x_normalizer_wrapper=x_normalizer_wrapper)[0]
acc_summary = []
f1_summary = []
eps = 2
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    acc_list = []
    f1_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "simple_fedonce_dp_covtype_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=10,
            num_local_rounds=10,
            local_lr=0.3,
            local_hidden_layers=[50],
            local_batch_size=32,
            local_weight_decay=0,
            local_output_size=3,
            num_agg_rounds=1,
            agg_lr=0.3,
            agg_hidden_layers=[30],
            agg_batch_size=32,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.0,
            inter_party_comp_method="simple",
            delta=1e-5,
            num_workers=4
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



num_parties = 4
x_scaler_wrapper = []
y_scaler_wrapper = []
x_normalizer_wrapper = []
cross_valid_data = load_data_cross_validation("covtype.binary.train", num_parties=num_parties,
                                              file_type='libsvm', n_fold=n_fold, use_cache=False,
                                              x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                              x_normalizer_wrapper=x_normalizer_wrapper)
xs_test, y_test = load_data_cross_validation("covtype.binary.test", num_parties=num_parties,
                                             file_type='libsvm', n_fold=1, use_cache=False,
                                             x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                             x_normalizer_wrapper=x_normalizer_wrapper)[0]
acc_summary = []
f1_summary = []
eps = 4
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    acc_list = []
    f1_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "simple_fedonce_dp_covtype_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=30,
            num_local_rounds=6,
            local_lr=0.2,
            local_hidden_layers=[50],
            local_batch_size=32,
            local_weight_decay=0,
            local_output_size=3,
            num_agg_rounds=1,
            agg_lr=0.2,
            agg_hidden_layers=[30],
            agg_batch_size=32,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.0,
            inter_party_comp_method="simple",
            delta=1e-5
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



num_parties = 4
x_scaler_wrapper = []
y_scaler_wrapper = []
x_normalizer_wrapper = []
cross_valid_data = load_data_cross_validation("covtype.binary.train", num_parties=num_parties,
                                              file_type='libsvm', n_fold=n_fold, use_cache=False,
                                              x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                              x_normalizer_wrapper=x_normalizer_wrapper)
xs_test, y_test = load_data_cross_validation("covtype.binary.test", num_parties=num_parties,
                                             file_type='libsvm', n_fold=1, use_cache=False,
                                             x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                             x_normalizer_wrapper=x_normalizer_wrapper)[0]
acc_summary = []
f1_summary = []
eps = 6
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    acc_list = []
    f1_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        version = 'v1'
        print("Active Party is {}".format(active_party))
        model_name = "simple_fedonce_dp_covtype_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=40,
            num_local_rounds=10,
            local_lr=0.3,
            local_hidden_layers=[50],
            local_batch_size=128,
            local_weight_decay=0,
            local_output_size=3,
            num_agg_rounds=1,
            agg_lr=0.6,
            agg_hidden_layers=[30],
            agg_batch_size=128,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.5,
            inter_party_comp_method="simple",
            delta=1e-5
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


#
num_parties = 4
x_scaler_wrapper = []
y_scaler_wrapper = []
x_normalizer_wrapper = []
cross_valid_data = load_data_cross_validation("covtype.binary.train", num_parties=num_parties,
                                              file_type='libsvm', n_fold=n_fold, use_cache=False,
                                              x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                              x_normalizer_wrapper=x_normalizer_wrapper)
xs_test, y_test = load_data_cross_validation("covtype.binary.test", num_parties=num_parties,
                                             file_type='libsvm', n_fold=1, use_cache=False,
                                             x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                             x_normalizer_wrapper=x_normalizer_wrapper)[0]
acc_summary = []
f1_summary = []
eps = 8
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    acc_list = []
    f1_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "simple_fedonce_dp_covtype_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=30,
            num_local_rounds=10,
            local_lr=0.3,
            local_hidden_layers=[50],
            local_batch_size=128,
            local_weight_decay=0,
            local_output_size=3,
            num_agg_rounds=1,
            agg_lr=0.6,
            agg_hidden_layers=[30],
            agg_batch_size=128,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.0,
            inter_party_comp_method="simple",
            delta=1e-5
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


################################## FedOnce-L0 ####################################
# FedOnce
num_parties = 4
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
acc_summary = []
f1_summary = []
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    acc_list = []
    f1_list = []
    times = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "fedonce_covtype_party_{}_fold_{}".format(num_parties, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            full_name=name,
            num_epochs=100,
            num_local_rounds=30,
            local_lr=1e-3,
            local_hidden_layers=[200, 100, 100],
            local_batch_size=100,
            local_weight_decay=1e-5,
            local_output_size=3,
            num_agg_rounds=1,
            agg_lr=3e-4,
            agg_hidden_layers=[30, 30],
            agg_batch_size=100,
            agg_weight_decay=1e-5,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            update_target_freq=1,
            task='binary_classification',
            n_classes=10,
            test_batch_size=1000,
            test_freq=1,
            cuda_parallel=False,
            n_channels=1,
            num_workers=8,
            model_type='fc',
            optimizer='adam',
        )
        acc, f1, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, use_cache=True)
        y_score_test = aggregate_model.predict_agg(xs_test)
        y_pred_test = np.where(y_score_test > 0.5, 1, 0)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)
        acc_list.append(test_acc)
        f1_list.append(test_f1)
        print(aggregate_model.params)
        print("Time Avg: {}, Std: {} (Minutes)".format(np.mean(times), np.std(times)))
    f1_summary.append(f1_list)
    acc_summary.append(acc_list)
    print("Active party {} finished training.".format(active_party))
    print("Accuracy for party {}".format(active_party) + str(acc_list))
    print("F1 score for party {}".format(active_party, str(f1_list)))
    print("-------------------------------------------------")
print("Accuracy summary: " + repr(acc_summary))
print("F1 score summary: " + repr(f1_summary))
for i, result in enumerate(acc_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: Accuracy mean={}, std={}".format(i, mean, std))


############## FedOnce-L0 w/ same model
# vertical FL with DP
num_parties = 4
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
acc_summary = []
f1_summary = []
eps = 2
for active_party in range(num_parties):
    print("Active party {} starts training".format(active_party))
    acc_list = []
    f1_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "fedonce_dp_covtype_party_{}_fold_{}_eps_{}".format(num_parties, i, eps)
        name = "{}_active_{}/".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=100,
            num_local_rounds=30,
            local_lr=1e-3,
            local_hidden_layers=[50],
            local_batch_size=100,
            local_weight_decay=1e-5,
            local_output_size=3,
            num_agg_rounds=1,
            agg_lr=3e-4,
            agg_hidden_layers=[30],
            agg_batch_size=100,
            agg_weight_decay=1e-5,
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
            momentum=0,
            privacy=None,
            batches_per_lot=1,
            epsilon=eps,
            grad_norm_C=1.0,
            inter_party_comp_method="MA",
            delta=1e-5,
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
