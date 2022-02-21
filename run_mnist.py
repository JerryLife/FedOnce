from utils.data_utils import load_data_train_test
from model.fl_model import VerticalFLModel
from model.simple_fl_model import PCAVerticalFLModel
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score

import os.path
import wget
import bz2
import shutil
import numpy as np
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0, help='Index of GPU')
parser.add_argument('--local-output-dim', '-d', type=int, default=3)
parser.add_argument('--repr-noise', '-r', type=float, default=0.0, help='Noise on representations')
args = parser.parse_args()

# SplitNN without DP
num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
    load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
acc_summary = []
f1_summary = []
acc_list = []
f1_list = []
start = datetime.now()
print("Starts training")
for i, (train_idx, val_idx) in enumerate(KFold(n_splits=5, shuffle=True).split(y_train_val)):
    print("Cross Validation Fold {}".format(i))
    start = datetime.now()
    xs_train = [data[train_idx] for data in xs_train_val]
    y_train = y_train_val[train_idx]
    xs_val = [data[val_idx] for data in xs_train_val]
    y_val = y_train_val[val_idx]
    model_name = "splitnn_mnist_party_{}_fold_{}".format(num_parties, i)
    name = "{}/".format(model_name)
    writer = SummaryWriter("runs/{}".format(name))
    aggregate_model = SplitNNModel(
        num_parties=num_parties,
        name=model_name,
        num_epochs=100,
        local_hidden_layers=[400, 200, 200],
        local_output_size=16,
        lr=3e-4,
        agg_hidden_layers=[128],
        batch_size=128,
        weight_decay=3e-5,
        writer=writer,
        device='cuda:{}'.format(args.gpu),
        update_target_freq=1,
        task='multi_classification',
        n_classes=10,
        test_batch_size=1000,
        test_freq=50,
        cuda_parallel=False,
        n_channels=1,
        model_type='cnn',
        optimizer='adam',
        privacy=None,
        batches_per_lot=5,
        epsilon=1,
        delta=1.0/xs_train[0].shape[0],
        num_workers=0
    )
    acc, f1, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, xs_test, y_test, use_cache=False)
    test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
    acc_list.append(test_acc)
    time_min = (datetime.now() - start).seconds / 60
    print(aggregate_model.params)
    print("Time(min) {}: ".format(time_min))

acc_summary.append(acc_list)
f1_summary.append(f1_list)
print("Finished training.")
print("-------------------------------------------------")
print("Accuracy summary: " + repr(acc_summary))
print("F1 score summary: " + repr(f1_summary))
for i, result in enumerate(acc_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: Accuracy mean={}, std={}".format(i, mean, std))
for i, result in enumerate(acc_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: F1-score mean={}, std={}".format(i, mean, std))


# FedOnce
num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
    load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
acc_summary = []
for active_party in range(num_parties):
    start = datetime.now()
    print("Active Party is {}".format(active_party))
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        start = datetime.now()
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        model_name = "fedonce_mnist_party_{}".format(num_parties, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            full_name=name,
            num_epochs=200,
            num_local_rounds=100,
            local_lr=1e-4,
            local_hidden_layers=[50, 50, 30],
            local_batch_size=128,
            local_weight_decay=1e-5,
            local_output_size=16,
            num_agg_rounds=1,
            agg_lr=1e-3,
            agg_hidden_layers=[128],
            agg_batch_size=128,
            agg_weight_decay=1e-5,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            update_target_freq=3,
            task='multi_classification',
            n_classes=10,
            test_batch_size=1000,
            test_freq=100,
            cuda_parallel=False,
            n_channels=1,
            model_type='cnn',
            optimizer='adam',
            privacy=None,
            num_workers=0
        )
        acc, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, use_cache=True)
        test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
        acc_list.append(test_acc)
        print(aggregate_model.params)
        time_min = (datetime.now() - start).seconds / 60
        print("Time(min) {}: ".format(time_min))
        break
    print("Active party {} finished training.".format(active_party))
    acc_summary.append(acc_list)
    print("----------------------------------------------------")
    break
print("Accuracy summary" + repr(acc_summary))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, result))


# single party
num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, _, _ = load_data_train_test("mnist", "mnist",
                                                                        num_parties=num_parties, file_type='torch')
acc_summary = []
for party_id in range(num_parties):
    acc_list = []
    for i, (train_idx, val_idx) in enumerate(KFold(n_splits=5, shuffle=True).split(y_train_val)):
        print("Cross Validation Fold {}".format(i))
        start = datetime.now()
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        name = "single_mnist_party_{}_single_{}_fold_{}".format(num_parties, party_id, i)
        writer = SummaryWriter("runs/{}".format(name))
        single_model = SingleParty(
            party_id=party_id,
            num_epochs=200,
            lr=1e-3,
            hidden_layers=[300, 200, 100],
            batch_size=128,
            weight_decay=1e-5,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            task="multi_classification",
            n_classes=10,
            test_batch_size=1000,
            test_freq=1,
            n_channels=1,
            model_type='cnn',
            optimizer='adam',
            cuda_parallel=False,
            num_workers=4
        )
        x_train = xs_train[party_id]
        x_val = xs_val[party_id]
        x_test = xs_test[party_id]
        acc, _, _, _ = single_model.train(x_train, y_train, x_val, y_val)
        test_acc = single_model.eval_image(x_test, y_test, ['accuracy'], has_transform=True)
        acc_list.append(test_acc)
        print(single_model.params)
    print("Party {} finished training".format(party_id))
    acc_summary.append(acc_list)
    print("-------------------------------------------------")
print("Accuracy summary" + repr(acc_summary))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, result))


# combine
num_parties = 1
xs_train_val, y_train_val, xs_test, y_test, _, _ = load_data_train_test("mnist", "mnist",
                                                                        num_parties=num_parties, file_type='torch')
acc_summary = []
for party_id in range(num_parties):
    acc_list = []
    for i, (train_idx, val_idx) in enumerate(KFold(n_splits=5, shuffle=True).split(y_train_val)):
        print("Cross Validation Fold {}".format(i))
        start = datetime.now()
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        name = "combine_mnist_fold_{}".format(i)
        writer = SummaryWriter("runs/{}".format(name))
        single_model = SingleParty(
            party_id=party_id,
            num_epochs=200,
            lr=3e-4,
            hidden_layers=[300, 200, 100],
            batch_size=128,
            weight_decay=1e-5,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            task="multi_classification",
            n_classes=10,
            test_batch_size=1000,
            test_freq=1,
            n_channels=1,
            model_type='cnn',
            optimizer='adam',
            cuda_parallel=False,
            num_workers=4
        )
        x_train = xs_train[party_id]
        x_val = xs_val[party_id]
        x_test = xs_test[party_id]
        acc, _, _, _ = single_model.train(x_train, y_train, x_val, y_val)
        test_acc = single_model.eval_image(x_test, y_test, ['accuracy'], has_transform=True)
        acc_list.append(test_acc)
        print(single_model.params)
    print("Party {} finished training".format(party_id))
    acc_summary.append(acc_list)
    print("-------------------------------------------------")
print("Accuracy summary" + repr(acc_summary))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, result))



# Post FedOnce
num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
    load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
acc_summary = []
f1_summary = []
acc_list = []
f1_list = []
start = datetime.now()
print("Starts training")
for i, (train_idx, val_idx) in enumerate(KFold(n_splits=5, shuffle=True).split(y_train_val)):
    print("Cross Validation Fold {}".format(i))
    start = datetime.now()
    xs_train = [data[train_idx] for data in xs_train_val]
    y_train = y_train_val[train_idx]
    xs_val = [data[val_idx] for data in xs_train_val]
    y_val = y_train_val[val_idx]
    model_name = "post_fedonce_mnist_party_{}_fold_{}".format(num_parties, i)
    name = "{}/".format(model_name)
    active_party = 0
    cache_local_name = "fedonce_mnist_party_{}_fold_{}".format(num_parties, i)
    cache_agg_name = "{}_active_{}".format(cache_local_name, active_party)
    writer = SummaryWriter("runs/{}".format(name))
    aggregate_model = SplitNNModel(
        num_parties=num_parties,
        name=model_name,
        num_epochs=100,
        local_hidden_layers=[400, 200, 200],
        local_output_size=16,
        lr=3e-4,
        agg_hidden_layers=[128],
        batch_size=128,
        weight_decay=3e-5,
        writer=writer,
        device='cuda:{}'.format(args.gpu),
        update_target_freq=1,
        task='multi_classification',
        n_classes=10,
        test_batch_size=1000,
        test_freq=1,
        cuda_parallel=False,
        n_channels=1,
        model_type='cnn',
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
    acc, f1, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, xs_test, y_test, use_cache=True)
    print("Finished training.")
    test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
    acc_list.append(test_acc)
    print(aggregate_model.params)
acc_summary.append(acc_list)
f1_summary.append(f1_list)
print("Finished training.")
print("-------------------------------------------------")
print("Accuracy summary: " + repr(acc_summary))
print("F1 score summary: " + repr(f1_summary))
for i, result in enumerate(acc_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: Accuracy mean={}, std={}".format(i, mean, std))



# FedOnce: Tuning d
num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
    load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
acc_summary = []
for active_party in range(num_parties):
    start = datetime.now()
    print("Active Party is {}".format(active_party))
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        start = datetime.now()
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        model_name = "fedonce_mnist_party_{}_fold_{}".format(num_parties, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            full_name=name,
            num_epochs=200,
            num_local_rounds=100,
            local_lr=1e-4,
            local_hidden_layers=[50, 50, 30],
            local_batch_size=128,
            local_weight_decay=1e-5,
            local_output_size=args.local_output_dim,
            num_agg_rounds=1,
            agg_lr=1e-3,
            agg_hidden_layers=[128],
            agg_batch_size=128,
            agg_weight_decay=1e-5,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            update_target_freq=3,
            task='multi_classification',
            n_classes=10,
            test_batch_size=1000,
            test_freq=1,
            cuda_parallel=False,
            n_channels=1,
            model_type='cnn',
            optimizer='adam',
            privacy=None
        )
        acc, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
        acc_list.append(test_acc)
        print(aggregate_model.params)
        time_min = (datetime.now() - start).seconds / 60
        print("Time(min) {}: ".format(time_min))

    print("Active party {} finished training.".format(active_party))
    acc_summary.append(acc_list)
    print("----------------------------------------------------")
print("Accuracy summary" + repr(acc_summary))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, result))




# representation with noise
num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
    load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
acc_summary = []
for active_party in range(num_parties):
    start = datetime.now()
    print("Active Party is {}".format(active_party))
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        start = datetime.now()
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        model_name = "fedonce_mnist_party_{}_fold_{}".format(num_parties, i)
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
            local_hidden_layers=[50, 50, 30],
            local_batch_size=128,
            local_weight_decay=1e-5,
            local_output_size=16,
            num_agg_rounds=1,
            agg_lr=1e-3,
            agg_hidden_layers=[128],
            agg_batch_size=128,
            agg_weight_decay=1e-5,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            update_target_freq=3,
            task='multi_classification',
            n_classes=10,
            test_batch_size=1000,
            test_freq=1,
            cuda_parallel=False,
            n_channels=1,
            model_type='cnn',
            optimizer='adam',
            privacy=None,
            repr_noise=args.repr_noise
        )
        acc, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
        acc_list.append(test_acc)
        print(aggregate_model.params)
        time_min = (datetime.now() - start).seconds / 60
        print("Time(min) {}: ".format(time_min))

    print("Active party {} finished training.".format(active_party))
    acc_summary.append(acc_list)
    print("----------------------------------------------------")
print("Accuracy summary" + repr(acc_summary))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, result))


# FedOnce + PCA
num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
    load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
acc_summary = []
for active_party in range(num_parties):
    start = datetime.now()
    print("Active Party is {}".format(active_party))
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        start = datetime.now()
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        model_name = "fedonce_mnist_party_{}_pca".format(num_parties, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = PCAVerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            full_name=name,
            num_epochs=200,
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
            task='multi_classification',
            n_classes=10,
            test_batch_size=1000,
            test_freq=1,
            cuda_parallel=False,
            n_channels=1,
            model_type='cnn',
            optimizer='adam',
        )
        acc, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, use_cache=False)
        test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
        acc_list.append(test_acc)
        print(aggregate_model.params)
        time_min = (datetime.now() - start).seconds / 60
        print("Time(min) {}: ".format(time_min))
        break
    print("Active party {} finished training.".format(active_party))
    acc_summary.append(acc_list)
    print("----------------------------------------------------")
print("Accuracy summary" + repr(acc_summary))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, result))
