from utils.data_utils import load_data_cross_validation, load_movielens
from model.fl_model import VerticalFLModel
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import os.path
import wget
import bz2
import shutil
import numpy as np

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import KFold
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0, help='Index of GPU')
parser.add_argument('--repr-noise', '-r', type=float, default=0.0, help='Noise on representations')
args = parser.parse_args()

# SecureBoost without DP (use XGBoost instead since SecureBoost is lossless)
num_parties = 10
xs_train_val, y_train_val, xs_test, y_test, counts = load_movielens("data/movielens/", use_cache=True,
                                                            download=True, test_rate=0.1)
print("Starts training XGBoost movielens")
x_train_val = np.concatenate(xs_train_val, axis=1)
x_test = np.concatenate(xs_test, axis=1)
rmse_list = []
kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
for i, (train_idx, val_idx) in enumerate(kfold):
    print("Cross Validation Fold {}".format(i))
    x_train = x_train_val[train_idx]
    y_train = y_train_val[train_idx]
    x_val = x_train_val[val_idx]
    y_val = y_train_val[val_idx]
    xg_cls = xgb.XGBRegressor(learning_rate=0.1,
                              max_depth=6,
                              n_estimators=200,
                              reg_alpha=10,
                              verbosity=2)
    xg_cls.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='rmse')
    y_pred = xg_cls.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_list.append(rmse)
print("Finished training.")
print("-------------------------------------------------")
print("RMSE=" + str(rmse_list))


# SplitNN
num_parties = 2
xs_train_val, y_train_val, xs_test, y_test, counts = load_movielens("data/movielens/", use_cache=True,
                                                                    download=True, test_rate=0.1,
                                                                    num_parties=num_parties)
rmse_list = []
kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
for i, (train_idx, val_idx) in enumerate(kfold):
    print("Cross Validation Fold {}".format(i))
    xs_train = [data[train_idx] for data in xs_train_val]
    y_train = y_train_val[train_idx]
    xs_val = [data[val_idx] for data in xs_train_val]
    y_val = y_train_val[val_idx]
    start = datetime.now()
    name = "splitnn_movielens_party_{}_fold_{}".format(num_parties, i)
    writer = SummaryWriter("runs/{}".format(name))
    ncf_counts = [counts[:2], counts[2:]]
    embed_dims = [[32, 32], [1, 4, 10, 4, 15, 5]]
    aggregate_model = SplitNNModel(
        num_parties=num_parties,
        name=name,
        num_epochs=100,
        local_hidden_layers=[32, 16],
        local_output_size=3,
        lr=3e-5,
        agg_hidden_layers=[10],
        batch_size=128,
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
        model_type='ncf',
        optimizer='adam',
        privacy=None,
        batches_per_lot=5,
        epsilon=1,
        delta=1.0 / xs_train[0].shape[0],
        num_workers=0,
        ncf_counts=ncf_counts,
        ncf_embed_dims=embed_dims
    )
    _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, xs_test, y_test, use_cache=False)
    y_pred_test, y_score_test = aggregate_model.predict(xs_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_score_test))
    rmse_list.append(test_rmse)
    print(aggregate_model.params)
    print("-------------------------------------------------")
    time_min = (datetime.now() - start).seconds / 60
    print("Time(min) {}: ".format(time_min))
print("Best RMSE={}".format(rmse_list))

# FedOnce
num_parties = 2
xs_train_val, y_train_val, xs_test, y_test, counts = load_movielens("data/movielens/", use_cache=True,
                                                            download=True, test_rate=0.1)
score_summary = []
print("Start training")
for party_id in range(num_parties):
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    rmse_list = []
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        start = datetime.now()
        name = "fedonce_movielens_party_{}_active_{}_fold_{}".format(num_parties, party_id, i)
        writer = SummaryWriter("runs/{}".format(name))
        ncf_counts = [counts[:2], counts[2:]]
        embed_dims = [[32, 32], [1, 4, 10, 4, 15, 5]]
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=party_id,
            name=name,
            full_name=name,
            num_epochs=40 if party_id == 0 else 100,
            num_local_rounds=30 if party_id == 0 else 30,
            local_lr=1e-4,
            local_hidden_layers=[32, 16] if party_id == 0 else [128],
            local_batch_size=64,
            local_weight_decay=1e-5,
            local_output_size=3 if party_id == 0 else 64,
            num_agg_rounds=1,
            agg_lr=1e-4 if party_id == 0 else 5e-4,
            agg_hidden_layers=[10] if party_id == 0 else [32, 32],
            agg_batch_size=64,
            agg_weight_decay=2e-4 if party_id == 0 else 1e-5,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            update_target_freq=1,
            task='regression',
            n_classes=10,
            test_batch_size=1000,
            test_freq=1,
            cuda_parallel=False,
            n_channels=1,
            model_type='ncf',
            optimizer='adam',
            privacy=None,
            ncf_counts=ncf_counts,
            ncf_embed_dims=embed_dims,
            num_workers=0
        )
        _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, use_cache=False)
        y_test_score = aggregate_model.predict_agg(xs_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_score))
        rmse_list.append(test_rmse)
        print(aggregate_model.params)
        time_min = (datetime.now() - start).seconds / 60
        print("Time(min) {}: ".format(time_min))
        break
    score_summary.append(rmse_list)
    print("RMSE for active party {}".format(party_id) + str(rmse_list))
    print("-------------------------------------------------")

for i, result in enumerate(score_summary):
    print("Party {}: RMSE={}".format(i, result))


# Solo
num_parties = 2
xs_train_val, y_train_val, xs_test, y_test, counts = load_movielens("data/movielens/", use_cache=True,
                                                            download=True, test_rate=0.1)
rmse_summary = []
print("Start training")
for party_id in range(num_parties):
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    rmse_list = []
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]

        name = "single_movielens_party_{}_single_{}_fold_{}".format(num_parties, party_id, i)
        writer = SummaryWriter("runs/{}".format(name))
        if party_id == 0:
            ncf_counts = counts[:2]
            embed_dims = [32, 32]
        elif party_id == 1:
            ncf_counts = counts[2:]
            embed_dims = [1, 4, 10, 4, 15, 5]
        else:
            assert False
        single_model = SingleParty(
            party_id=party_id,
            num_epochs=60,
            lr=1e-4,
            hidden_layers=[32, 16],
            batch_size=128,
            weight_decay=1e-5,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            task="regression",
            test_batch_size=1000,
            test_freq=1,
            model_type='ncf',
            optimizer='adam',
            cuda_parallel=False,
            ncf_counts=ncf_counts,
            ncf_embed_dims=embed_dims
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
    print("RMSE for active party {}".format(party_id) + str(rmse_list))
    print("-------------------------------------------------")
for i, result in enumerate(rmse_summary):
    print("Party {}: accuracy={}".format(i, result))


# combine
num_parties = 1
xs_train_val, y_train_val, xs_test, y_test, counts = load_movielens("data/movielens/", use_cache=True,
                                                            download=True, test_rate=0.1)
x_train_val = np.concatenate(xs_train_val, axis=1)
x_test = np.concatenate(xs_test, axis=1)
kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
rmse_list = []
for i, (train_idx, val_idx) in enumerate(kfold):
    print("Cross Validation Fold {}".format(i))
    x_train = x_train_val[train_idx]
    y_train = y_train_val[train_idx]
    x_val = x_train_val[val_idx]
    y_val = y_train_val[val_idx]
    name = "combine_movielens_fold_{}".format(i)
    writer = SummaryWriter("runs/{}".format(name))
    single_model = SingleParty(
        party_id=0,
        num_epochs=60,
        lr=1e-4,
        hidden_layers=[64, 32],
        batch_size=128,
        weight_decay=3e-5,
        writer=writer,
        device='cuda:{}'.format(args.gpu),
        task="regression",
        test_batch_size=1000,
        test_freq=1,
        model_type='ncf',
        optimizer='adam',
        cuda_parallel=False,
        ncf_counts=counts,
        ncf_embed_dims=[32, 32, 1, 4, 10, 4, 15, 5]
    )
    _, _, rmse, _ = single_model.train(x_train, y_train, x_val, y_val)
    y_pred_test, y_score_test = single_model.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_score_test))
    rmse_list.append(test_rmse)
    print(single_model.params)
print("-------------------------------------------------")
print("Best accuracy={}".format(rmse_list))


# Post FedOnce
num_parties = 2
xs_train_val, y_train_val, xs_test, y_test, counts = load_movielens("data/movielens/", use_cache=True,
                                                                    download=True, test_rate=0.1,
                                                                    num_parties=num_parties)
rmse_list = []
kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
for i, (train_idx, val_idx) in enumerate(kfold):
    print("Cross Validation Fold {}".format(i))
    xs_train = [data[train_idx] for data in xs_train_val]
    y_train = y_train_val[train_idx]
    xs_val = [data[val_idx] for data in xs_train_val]
    y_val = y_train_val[val_idx]
    start = datetime.now()
    name = "post_fedonce_movielens_party_{}_fold_{}".format(num_parties, i)
    active_party = 0
    cache_local_name = "fedonce_movielens_party_{}_active_{}_fold_{}".format(num_parties, active_party, i)
    cache_agg_name = "fedonce_movielens_party_{}_active_{}_fold_{}".format(num_parties, active_party, i)
    writer = SummaryWriter("runs/{}".format(name))
    ncf_counts = [counts[:2], counts[2:]]
    embed_dims = [[32, 32], [1, 4, 10, 4, 15, 5]]
    aggregate_model = SplitNNModel(
        num_parties=num_parties,
        name=name,
        num_epochs=100,
        local_hidden_layers=[32, 16],
        local_output_size=3,
        lr=3e-5,
        agg_hidden_layers=[10],
        batch_size=128,
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
        model_type='ncf',
        optimizer='adam',
        privacy=None,
        batches_per_lot=5,
        epsilon=1,
        delta=1.0 / xs_train[0].shape[0],
        num_workers=2,
        ncf_counts=ncf_counts,
        ncf_embed_dims=embed_dims,
        cache_local_name=cache_local_name,
        cache_agg_name=cache_agg_name,
        active_party=active_party
    )
    _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, xs_test, y_test, use_cache=True)
    y_pred_test, y_score_test = aggregate_model.predict(xs_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_score_test))
    rmse_list.append(test_rmse)
    print(aggregate_model.params)
    print("-------------------------------------------------")
    time_min = (datetime.now() - start).seconds / 60
    print("Time(min) {}: ".format(time_min))
print("Best RMSE={}".format(rmse_list))



# FedOnce noise on representations
num_parties = 2
xs_train_val, y_train_val, xs_test, y_test, counts = load_movielens("data/movielens/", use_cache=True,
                                                            download=True, test_rate=0.1)
score_summary = []
print("Start training")
for party_id in range(num_parties)[::-1]:
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    rmse_list = []
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        start = datetime.now()
        name = "fedonce_movielens_party_{}_active_{}_fold_{}".format(num_parties, party_id, i)
        writer = SummaryWriter("runs/{}".format(name))
        ncf_counts = [counts[:2], counts[2:]]
        embed_dims = [[32, 32], [1, 4, 10, 4, 15, 5]]
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=party_id,
            name=name,
            full_name=name,
            num_epochs=40 if party_id == 0 else 100,
            num_local_rounds=30 if party_id == 0 else 30,
            local_lr=1e-4,
            local_hidden_layers=[32, 16] if party_id == 0 else [128],
            local_batch_size=64,
            local_weight_decay=1e-5,
            local_output_size=3 if party_id == 0 else 64,
            num_agg_rounds=1,
            agg_lr=1e-4 if party_id == 0 else 5e-4,
            agg_hidden_layers=[10] if party_id == 0 else [32, 32],
            agg_batch_size=64,
            agg_weight_decay=2e-4 if party_id == 0 else 1e-5,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            update_target_freq=1,
            task='regression',
            n_classes=10,
            test_batch_size=1000,
            test_freq=1,
            cuda_parallel=False,
            n_channels=1,
            model_type='ncf',
            optimizer='adam',
            privacy=None,
            ncf_counts=ncf_counts,
            ncf_embed_dims=embed_dims,
            repr_noise=args.repr_noise
        )
        _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, use_cache=True)
        y_test_score = aggregate_model.predict_agg(xs_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_score))
        rmse_list.append(test_rmse)
        print(aggregate_model.params)
        time_min = (datetime.now() - start).seconds / 60
        print("Time(min) {}: ".format(time_min))
        break
    score_summary.append(rmse_list)
    print("RMSE for active party {}".format(party_id) + str(rmse_list))
    print("-------------------------------------------------")

for i, result in enumerate(score_summary[::-1]):
    print("Party {}: RMSE={}".format(i, result))

