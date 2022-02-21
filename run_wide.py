from utils.data_utils import load_data_cross_validation, load_nus_wide
from model.fl_model import VerticalFLModel
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel
from model.simple_fl_model import PCAVerticalFLModel

from torch.utils.tensorboard import SummaryWriter
from joblib import Parallel, delayed

import os.path
import wget
import bz2
import shutil
import numpy as np

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from datetime import datetime
from sklearn.model_selection import KFold
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0, help='Index of GPU')
parser.add_argument('--repr-noise', '-r', type=float, default=0.0, help='Noise on representations')
args = parser.parse_args()


label_type = 'sky'
xs_train_val, y_train_val, xs_test, y_test = load_nus_wide("data/nus-wide", use_cache=True, label_type=label_type,
                                                           balance=False, num_parties=1, raw_split=True)
x_train_val = np.concatenate(xs_train_val, axis=1)
x_test = np.concatenate(xs_test, axis=1)
print("Starts training XGBoost wide")
acc_list = []
kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
for i, (train_idx, val_idx) in enumerate(kfold):
    print("Cross Validation Fold {}".format(i))
    x_train = x_train_val[train_idx]
    y_train = y_train_val[train_idx]
    x_val = x_train_val[val_idx]
    y_val = y_train_val[val_idx]

    xg_cls = xgb.XGBClassifier(objective='binary:logistic',
                               learning_rate=0.1,
                               max_depth=6,
                               n_estimators=150,
                               reg_alpha=10,
                               verbosity=2)
    xg_cls.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='error')
    y_pred = xg_cls.predict(x_test)
    y_score = xg_cls.predict_proba(x_test)[:, 1].reshape(-1)
    acc = accuracy_score(y_test, y_pred)
    acc_list.append(acc)
print("Finished training.")
print("-------------------------------------------------")
print("Accuracy=" + str(acc_list))

# SplitNN
num_parties = 5
label_type = 'sky'
xs_train_val, y_train_val, xs_test, y_test = load_nus_wide("data/nus-wide", use_cache=True, label_type=label_type,
                                                           balance=False, num_parties=num_parties, raw_split=True)
start = datetime.now()
print("Starts training")
kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
acc_list = []
for i, (train_idx, val_idx) in enumerate(kfold):
    print("Cross Validation Fold {}".format(i))
    xs_train = [data[train_idx] for data in xs_train_val]
    y_train = y_train_val[train_idx]
    xs_val = [data[val_idx] for data in xs_train_val]
    y_val = y_train_val[val_idx]
    model_name = "splitnn_wide_party_{}_{}_fold_{}".format(num_parties, label_type, i)
    name = "{}/".format(model_name)
    writer = SummaryWriter("runs/{}".format(name))
    aggregate_model = SplitNNModel(
        num_parties=num_parties,
        name=model_name,
        num_epochs=100,
        local_hidden_layers=[60],
        local_output_size=8,
        lr=1e-5,
        agg_hidden_layers=[50],
        batch_size=128,
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
        num_workers=0
    )
    acc, _, _, auc = aggregate_model.train(xs_train, y_train, xs_val, y_val, xs_test, y_test, use_cache=False)
    y_predict_test, y_score_test = aggregate_model.predict(xs_test)
    test_acc = accuracy_score(y_test, y_predict_test)
    acc_list.append(test_acc)
    print(aggregate_model.params)
    time_min = (datetime.now() - start).seconds / 60
    print("Time(min) {}: ".format(time_min))
print("Finished training.")
print("-------------------------------------------------")
print("accuracy=" + str(acc_list))

# FedOnce
num_parties = 5
label_type = 'sky'
xs_train_val, y_train_val, xs_test, y_test = load_nus_wide("data/nus-wide", use_cache=True, label_type=label_type,
                                                           balance=False, num_parties=num_parties, raw_split=True)
auc_summary = []
acc_summary = []
for active_party in range(num_parties):
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        start = datetime.now()
        print("Active Party is {}".format(active_party))
        model_name = "fedonce_wide_party_{}_{}_{}".format(num_parties, label_type, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=name,
            full_name=name,
            num_epochs=50,
            num_local_rounds=50,
            local_lr=3e-4,
            local_hidden_layers=[60],
            local_batch_size=128,
            local_weight_decay=1e-5,
            local_output_size=8,
            num_agg_rounds=1,
            agg_lr=7e-5,
            agg_hidden_layers=[50],
            agg_batch_size=128,
            agg_weight_decay=1e-5,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            update_target_freq=1,
            task='binary_classification',
            n_classes=10,
            test_batch_size=1000,
            test_freq=2,
            cuda_parallel=False,
            n_channels=1,
            model_type='fc',
            optimizer='adam',
            privacy=None,
            batches_per_lot=12,
            epsilon=1,
            delta=1e-5
        )
        acc, _, _, auc = aggregate_model.train(xs_train, y_train, xs_val, y_val, use_cache=False)
        y_score_test = aggregate_model.predict_agg(xs_test)
        y_pred_test = np.where(y_score_test > 0.5, 1, 0)
        test_acc = accuracy_score(y_test, y_pred_test)
        acc_list.append(test_acc)
        print("Active party {} finished training.".format(active_party))
        print(aggregate_model.params)
        print("----------------------------------------------------")
        time_min = (datetime.now() - start).seconds / 60
        print("Time(min) {}: ".format(time_min))
    acc_summary.append(acc_list)
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, result))

# Solo
num_parties = 5
label_type = 'sky'
xs_train_val, y_train_val, xs_test, y_test = load_nus_wide("data/nus-wide", use_cache=True, label_type=label_type,
                                                           balance=False, num_parties=num_parties, raw_split=True)
auc_summary = []
acc_summary = []
def run_single_party(party_id):
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]

        name = "single_wide_party_{}_single_{}_{}_fold_{}".format(num_parties, party_id, label_type, i)
        writer = SummaryWriter("runs/{}".format(name))
        single_model = SingleParty(
            party_id=party_id,
            num_epochs=60,
            lr=1e-4,
            hidden_layers=[60],
            batch_size=128,
            weight_decay=1e-5,
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
        x_val = xs_val[party_id]
        x_test = xs_test[party_id]
        acc, _, _, auc = single_model.train(x_train, y_train, x_val, y_val)
        y_predict_test, y_score_test = single_model.predict(x_test)
        test_acc = accuracy_score(y_test, y_predict_test)
        acc_list.append(test_acc)
        print(single_model.params)
    return acc_list
party_ids = np.arange(num_parties)
results = Parallel(n_jobs=10)(delayed(run_single_party)(party_id) for party_id in party_ids)
print("-------------------------------------------------")
for party_id, acc in zip(party_ids, results):
    print("Party {}: Accuracy={}".format(party_id, acc))


# combine
num_parties = 1
label_type = 'sky'
xs_train_val, y_train_val, xs_test, y_test = load_nus_wide("data/nus-wide", use_cache=True, label_type=label_type,
                                                           balance=False, num_parties=num_parties, raw_split=True)
x_train_val = np.concatenate(xs_train_val, axis=1)
x_test = np.concatenate(xs_test, axis=1)
acc_list = []
kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
for i, (train_idx, val_idx) in enumerate(kfold):
    print("Cross Validation Fold {}".format(i))
    x_train = x_train_val[train_idx]
    y_train = y_train_val[train_idx]
    x_val = x_train_val[val_idx]
    y_val = y_train_val[val_idx]
    name = "combine_wide_{}_fold_{}".format(label_type, i)
    writer = SummaryWriter("runs/{}".format(name))
    single_model = SingleParty(
        party_id=0,
        num_epochs=60,
        lr=1e-4,
        hidden_layers=[100],
        batch_size=128,
        weight_decay=1e-5,
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
    acc, _, _, auc = single_model.train(x_train, y_train, x_test, y_test)
    y_predict_test, y_score_test = single_model.predict(x_test)
    test_acc = accuracy_score(y_test, y_predict_test)
    acc_list.append(test_acc)
    print(single_model.params)
print("-------------------------------------------------")
print("Best accuracy={}".format(acc_list))


# SplitNN
num_parties = 5
label_type = 'sky'
xs_train_val, y_train_val, xs_test, y_test = load_nus_wide("data/nus-wide", use_cache=True, label_type=label_type,
                                                           balance=False, num_parties=num_parties, raw_split=True)
start = datetime.now()
print("Starts training")
kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
acc_list = []
for i, (train_idx, val_idx) in enumerate(kfold):
    print("Cross Validation Fold {}".format(i))
    xs_train = [data[train_idx] for data in xs_train_val]
    y_train = y_train_val[train_idx]
    xs_val = [data[val_idx] for data in xs_train_val]
    y_val = y_train_val[val_idx]
    model_name = "post_fedonce_wide_party_{}_{}_fold_{}".format(num_parties, label_type, i)
    name = "{}/".format(model_name)
    active_party = 0
    cache_local_name = "fedonce_wide_party_{}_{}_{}".format(num_parties, label_type, i)
    cache_agg_name = "{}_active_{}".format(cache_local_name, active_party)
    writer = SummaryWriter("runs/{}".format(name))
    aggregate_model = SplitNNModel(
        num_parties=num_parties,
        name=model_name,
        num_epochs=100,
        local_hidden_layers=[60],
        local_output_size=8,
        lr=1e-5,
        agg_hidden_layers=[50],
        batch_size=128,
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
        num_workers=2,
        cache_local_name=cache_local_name,
        cache_agg_name=cache_agg_name,
        active_party=active_party
    )
    acc, _, _, auc = aggregate_model.train(xs_train, y_train, xs_val, y_val, xs_test, y_test, use_cache=True)
    y_predict_test, y_score_test = aggregate_model.predict(xs_test)
    test_acc = accuracy_score(y_test, y_predict_test)
    acc_list.append(test_acc)
    print(aggregate_model.params)
    time_min = (datetime.now() - start).seconds / 60
    print("Time(min) {}: ".format(time_min))
print("Finished training.")
print("-------------------------------------------------")
print("accuracy=" + str(acc_list))


# FedOnce + PCA
num_parties = 5
label_type = 'sky'
xs_train, y_train, xs_test, y_test = load_nus_wide("data/nus-wide", use_cache=True, label_type=label_type,
                                                   balance=False, num_parties=num_parties)
auc_summary = []
acc_summary = []
for active_party in range(num_parties):
    print("Active Party is {}".format(active_party))
    model_name = "fedonce_wide_party_{}_{}_pca".format(num_parties, label_type)
    name = "{}_active_{}".format(model_name, active_party)
    writer = SummaryWriter("runs/{}".format(name))
    aggregate_model = PCAVerticalFLModel(
        num_parties=num_parties,
        active_party_id=active_party,
        name=model_name,
        num_epochs=50,
        num_local_rounds=50,
        local_lr=3e-4,
        local_hidden_layers=[60],
        local_batch_size=128,
        local_weight_decay=1e-5,
        local_output_size=8,
        num_agg_rounds=1,
        agg_lr=7e-5,
        agg_hidden_layers=[50],
        agg_batch_size=128,
        agg_weight_decay=1e-5,
        writer=writer,
        device='cuda:{}'.format(args.gpu),
        update_target_freq=1,
        task='binary_classification',
        n_classes=10,
        test_batch_size=1000,
        test_freq=2,
        cuda_parallel=False,
        n_channels=1,
        model_type='fc',
        optimizer='adam',
        privacy=None,
        batches_per_lot=12,
        epsilon=1,
        delta=1e-5
    )
    acc, _, _, auc = aggregate_model.train(xs_train, y_train, xs_test, y_test, use_cache=False)
    auc_summary.append(auc)
    acc_summary.append(acc)
    print("Active party {} finished training.".format(active_party))
    print(aggregate_model.params)
    print("----------------------------------------------------")
for i, result in enumerate(auc_summary):
    print("Party {}: AUC={}".format(i, result))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, result))

num_parties = 5
label_type = 'sky'
xs_train_val, y_train_val, xs_test, y_test = load_nus_wide("data/nus-wide", use_cache=True, label_type=label_type,
                                                           balance=False, num_parties=num_parties, raw_split=True)
auc_summary = []
acc_summary = []
for active_party in range(num_parties):
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        start = datetime.now()
        print("Active Party is {}".format(active_party))
        model_name = "fedonce_wide_party_{}_{}_{}".format(num_parties, label_type, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = PCAVerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=50,
            num_local_rounds=50,
            local_lr=3e-4,
            local_hidden_layers=[60],
            local_batch_size=128,
            local_weight_decay=1e-5,
            local_output_size=8,
            num_agg_rounds=1,
            agg_lr=7e-5,
            agg_hidden_layers=[50],
            agg_batch_size=128,
            agg_weight_decay=1e-5,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            update_target_freq=1,
            task='binary_classification',
            n_classes=10,
            test_batch_size=1000,
            test_freq=2,
            cuda_parallel=False,
            n_channels=1,
            model_type='fc',
            optimizer='adam',
            privacy=None,
            batches_per_lot=12,
            epsilon=1,
            delta=1e-5
        )
        acc, _, _, auc = aggregate_model.train(xs_train, y_train, xs_val, y_val, use_cache=True)
        y_score_test = aggregate_model.predict_agg(xs_test)
        y_pred_test = np.where(y_score_test > 0.5, 1, 0)
        test_acc = accuracy_score(y_test, y_pred_test)
        acc_list.append(test_acc)
        print("Active party {} finished training.".format(active_party))
        print(aggregate_model.params)
        print("----------------------------------------------------")
        time_min = (datetime.now() - start).seconds / 60
        print("Time(min) {}: ".format(time_min))
    acc_summary.append(acc_list)
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, result))




# FedOnce noise on representations
num_parties = 5
label_type = 'sky'
xs_train_val, y_train_val, xs_test, y_test = load_nus_wide("data/nus-wide", use_cache=True, label_type=label_type,
                                                           balance=False, num_parties=num_parties, raw_split=True)
auc_summary = []
acc_summary = []
for active_party in range(num_parties):
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        start = datetime.now()
        print("Active Party is {}".format(active_party))
        model_name = "fedonce_wide_party_{}_{}_{}".format(num_parties, label_type, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=name,
            full_name=name,
            num_epochs=50,
            num_local_rounds=50,
            local_lr=3e-4,
            local_hidden_layers=[60],
            local_batch_size=128,
            local_weight_decay=1e-5,
            local_output_size=8,
            num_agg_rounds=1,
            agg_lr=7e-5,
            agg_hidden_layers=[50],
            agg_batch_size=128,
            agg_weight_decay=1e-5,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            update_target_freq=1,
            task='binary_classification',
            n_classes=10,
            test_batch_size=1000,
            test_freq=2,
            cuda_parallel=False,
            n_channels=1,
            model_type='fc',
            optimizer='adam',
            privacy=None,
            batches_per_lot=12,
            epsilon=1,
            delta=1e-5,
            repr_noise=args.repr_noise
        )
        acc, _, _, auc = aggregate_model.train(xs_train, y_train, xs_val, y_val, use_cache=True)
        y_score_test = aggregate_model.predict_agg(xs_test)
        y_pred_test = np.where(y_score_test > 0.5, 1, 0)
        test_acc = accuracy_score(y_test, y_pred_test)
        acc_list.append(test_acc)
        print("Active party {} finished training.".format(active_party))
        print(aggregate_model.params)
        print("----------------------------------------------------")
        time_min = (datetime.now() - start).seconds / 60
        print("Time(min) {}: ".format(time_min))
    acc_summary.append(acc_list)
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, result))

