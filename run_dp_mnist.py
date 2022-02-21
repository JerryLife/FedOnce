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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0, help='Index of GPU')
parser.add_argument('--local-output-dim', '-d', type=int, default=3)
args = parser.parse_args()


############################### FedOnce-L1 ##################################

# vertical FL with DP
num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
    load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
acc_summary = []
eps = 2
for active_party in range(num_parties):
    print("Active Party is {}".format(active_party))
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Active Party is {}".format(active_party))
        model_name = "dp_fedonce_mnist_party_{}_eps_{}_fold_{}".format(num_parties, eps, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=15,
            num_local_rounds=6,
            local_lr=0.5,
            local_hidden_layers=[50, 50, 30],
            local_batch_size=256,
            local_weight_decay=0,
            local_output_size=8,
            num_agg_rounds=1,
            agg_lr=0.8,
            agg_hidden_layers=[10],
            agg_batch_size=256,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            delta=1e-5,
            inter_party_comp_method="MA",
            grad_norm_C=1.5
        )
        acc, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
        acc_list.append(test_acc)
        print(aggregate_model.params)

    print("Active party {} finished training.".format(active_party))
    acc_summary.append(acc_list)
    print("----------------------------------------------------")
print("Accuracy summary" + repr(acc_summary))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, np.average(result)))


num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
    load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
acc_summary = []
eps = 4
for active_party in range(num_parties):
    print("Active Party is {}".format(active_party))
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Active Party is {}".format(active_party))
        model_name = "dp_fedonce_mnist_party_{}_eps_{}_fold_{}".format(num_parties, eps, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=15,
            num_local_rounds=6,
            local_lr=0.5,
            local_hidden_layers=[50, 50, 30],
            local_batch_size=256,
            local_weight_decay=0,
            local_output_size=8,
            num_agg_rounds=1,
            agg_lr=0.9,
            agg_hidden_layers=[10],
            agg_batch_size=256,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            delta=1e-5,
            inter_party_comp_method="MA",
            grad_norm_C=1.5
        )
        acc, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
        acc_list.append(test_acc)
        print(aggregate_model.params)

    print("Active party {} finished training.".format(active_party))
    acc_summary.append(acc_list)
    print("----------------------------------------------------")
print("Accuracy summary" + repr(acc_summary))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, np.average(result)))



num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
    load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
acc_summary = []
eps = 6
for active_party in range(num_parties):
    print("Active Party is {}".format(active_party))
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Active Party is {}".format(active_party))
        model_name = "dp_fedonce_mnist_party_{}_eps_{}_fold_{}".format(num_parties, eps, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=15,
            num_local_rounds=7,
            local_lr=0.5,
            local_hidden_layers=[50, 50, 30],
            local_batch_size=256,
            local_weight_decay=0,
            local_output_size=8,
            num_agg_rounds=1,
            agg_lr=0.9,
            agg_hidden_layers=[10],
            agg_batch_size=256,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            delta=1e-5,
            inter_party_comp_method="MA",
            grad_norm_C=1.5
        )
        acc, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
        acc_list.append(test_acc)
        print(aggregate_model.params)

    print("Active party {} finished training.".format(active_party))
    acc_summary.append(acc_list)
    print("----------------------------------------------------")
print("Accuracy summary" + repr(acc_summary))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, np.average(result)))


num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
    load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
acc_summary = []
eps = 8
for active_party in range(num_parties):
    print("Active Party is {}".format(active_party))
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Active Party is {}".format(active_party))
        model_name = "dp_fedonce_mnist_party_{}_eps_{}_fold_{}".format(num_parties, eps, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=30,
            num_local_rounds=10,
            local_lr=0.5,
            local_hidden_layers=[50, 50, 30],
            local_batch_size=512,
            local_weight_decay=0,
            local_output_size=8,
            num_agg_rounds=1,
            agg_lr=0.9,
            agg_hidden_layers=[10],
            agg_batch_size=512,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            delta=1e-5,
            inter_party_comp_method="MA",
            grad_norm_C=1.5
        )
        acc, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
        acc_list.append(test_acc)
        print(aggregate_model.params)

    print("Active party {} finished training.".format(active_party))
    acc_summary.append(acc_list)
    print("----------------------------------------------------")
print("Accuracy summary" + repr(acc_summary))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, np.average(result)))


################################ Priv-Baseline ###################################

num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
    load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
acc_summary = []
eps = 2
for active_party in range(num_parties):
    print("Active Party is {}".format(active_party))
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Active Party is {}".format(active_party))
        model_name = "dp_simple_fedonce_mnist_party_{}_eps_{}_fold_{}".format(num_parties, eps, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=15,
            num_local_rounds=6,
            local_lr=0.5,
            local_hidden_layers=[50, 50, 30],
            local_batch_size=256,
            local_weight_decay=0,
            local_output_size=8,
            num_agg_rounds=1,
            agg_lr=0.8,
            agg_hidden_layers=[10],
            agg_batch_size=256,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            delta=1e-5,
            inter_party_comp_method="simple",
            grad_norm_C=1.5
        )
        acc, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
        acc_list.append(test_acc)
        print(aggregate_model.params)

    print("Active party {} finished training.".format(active_party))
    acc_summary.append(acc_list)
    print("----------------------------------------------------")
print("Accuracy summary" + repr(acc_summary))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, np.average(result)))


num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
    load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
acc_summary = []
eps = 4
for active_party in range(num_parties):
    print("Active Party is {}".format(active_party))
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Active Party is {}".format(active_party))
        model_name = "dp_simple_fedonce_mnist_party_{}_eps_{}_fold_{}".format(num_parties, eps, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=15,
            num_local_rounds=6,
            local_lr=0.5,
            local_hidden_layers=[50, 50, 30],
            local_batch_size=256,
            local_weight_decay=0,
            local_output_size=8,
            num_agg_rounds=1,
            agg_lr=0.9,
            agg_hidden_layers=[10],
            agg_batch_size=256,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            delta=1e-5,
            inter_party_comp_method="simple",
            grad_norm_C=1.5
        )
        acc, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
        acc_list.append(test_acc)
        print(aggregate_model.params)

    print("Active party {} finished training.".format(active_party))
    acc_summary.append(acc_list)
    print("----------------------------------------------------")
print("Accuracy summary" + repr(acc_summary))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, np.average(result)))


num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
    load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
acc_summary = []
eps = 6
for active_party in range(num_parties):
    print("Active Party is {}".format(active_party))
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Active Party is {}".format(active_party))
        model_name = "dp_simple_fedonce_mnist_party_{}_eps_{}_fold_{}".format(num_parties, eps, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=15,
            num_local_rounds=6,
            local_lr=0.5,
            local_hidden_layers=[50, 50, 30],
            local_batch_size=256,
            local_weight_decay=0,
            local_output_size=8,
            num_agg_rounds=1,
            agg_lr=0.9,
            agg_hidden_layers=[10],
            agg_batch_size=256,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            delta=1e-5,
            inter_party_comp_method="simple",
            grad_norm_C=1.5
        )
        acc, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
        acc_list.append(test_acc)
        print(aggregate_model.params)

    print("Active party {} finished training.".format(active_party))
    acc_summary.append(acc_list)
    print("----------------------------------------------------")
print("Accuracy summary" + repr(acc_summary))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, np.average(result)))


num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
    load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
acc_summary = []
eps = 8
for active_party in range(num_parties):
    print("Active Party is {}".format(active_party))
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Active Party is {}".format(active_party))
        model_name = "dp_simple_fedonce_mnist_party_{}_eps_{}_fold_{}".format(num_parties, eps, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=15,
            num_local_rounds=6,
            local_lr=0.5,
            local_hidden_layers=[50, 50, 30],
            local_batch_size=256,
            local_weight_decay=0,
            local_output_size=8,
            num_agg_rounds=1,
            agg_lr=0.9,
            agg_hidden_layers=[10],
            agg_batch_size=256,
            agg_weight_decay=0,
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
            optimizer='sgd',
            momentum=0,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            delta=1e-5,
            inter_party_comp_method="simple",
            grad_norm_C=1.5
        )
        acc, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
        acc_list.append(test_acc)
        print(aggregate_model.params)

    print("Active party {} finished training.".format(active_party))
    acc_summary.append(acc_list)
    print("----------------------------------------------------")
print("Accuracy summary" + repr(acc_summary))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, np.average(result)))



######## Same model l0

num_parties = 4
xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
    load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
acc_summary = []
eps = 1e10
for active_party in range(num_parties):
    print("Active Party is {}".format(active_party))
    acc_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    for i, (train_idx, val_idx) in enumerate(kfold):
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Active Party is {}".format(active_party))
        model_name = "dp_simple_fedonce_mnist_party_{}_eps_{}_fold_{}".format(num_parties, eps, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=100,
            num_local_rounds=100,
            local_lr=1e-3,
            local_hidden_layers=[50, 50, 30],
            local_batch_size=256,
            local_weight_decay=1e-5,
            local_output_size=8,
            num_agg_rounds=1,
            agg_lr=1e-3,
            agg_hidden_layers=[10],
            agg_batch_size=256,
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
            momentum=0.9,
            privacy="MA",
            batches_per_lot=1,
            epsilon=eps,
            delta=1e-5,
            inter_party_comp_method="MA",
            grad_norm_C=2
        )
        acc, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, use_cache=False)
        test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
        acc_list.append(test_acc)
        print(aggregate_model.params)

    print("Active party {} finished training.".format(active_party))
    acc_summary.append(acc_list)
    print("----------------------------------------------------")
print("Accuracy summary" + repr(acc_summary))
for i, result in enumerate(acc_summary):
    print("Party {}: accuracy={}".format(i, np.average(result)))

#
# num_parties = 4
# xs_train_val, y_train_val, xs_test, y_test, x_train, x_test = \
#     load_data_train_test("mnist", "mnist", num_parties=num_parties, file_type='torch')
# acc_summary = []
# eps = np.log(1e100)
# for active_party in range(num_parties):
#     print("Active Party is {}".format(active_party))
#     acc_list = []
#     kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
#     for i, (train_idx, val_idx) in enumerate(kfold):
#         xs_train = [data[train_idx] for data in xs_train_val]
#         y_train = y_train_val[train_idx]
#         xs_val = [data[val_idx] for data in xs_train_val]
#         y_val = y_train_val[val_idx]
#         print("Active Party is {}".format(active_party))
#         model_name = "dp_simple_fedonce_mnist_party_{}_eps_{}_fold_{}".format(num_parties, eps, i)
#         name = "{}_active_{}".format(model_name, active_party)
#         writer = SummaryWriter("runs/{}".format(name))
#         aggregate_model = VerticalFLModel(
#             num_parties=num_parties,
#             active_party_id=active_party,
#             name=model_name,
#             num_epochs=200,
#             num_local_rounds=100,
#             local_lr=1e-4,
#             local_hidden_layers=[50, 50, 30],
#             local_batch_size=256,
#             local_weight_decay=1e-5,
#             local_output_size=8,
#             num_agg_rounds=1,
#             agg_lr=1e-3,
#             agg_hidden_layers=[10],
#             agg_batch_size=256,
#             agg_weight_decay=1e-5,
#             writer=writer,
#             device='cuda:{}'.format(args.gpu),
#             update_target_freq=1,
#             task='multi_classification',
#             n_classes=10,
#             test_batch_size=1000,
#             test_freq=1,
#             cuda_parallel=False,
#             n_channels=1,
#             model_type='cnn',
#             optimizer='adam',
#             momentum=0,
#             privacy='MA',
#             batches_per_lot=1,
#             epsilon=eps,
#             delta=1e-5,
#             inter_party_comp_method="simple",
#             grad_norm_C=1.5
#         )
#         acc, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, use_cache=False)
#         test_acc = aggregate_model.eval_image(xs_test, y_test, ['accuracy'], has_transform=True)
#         acc_list.append(test_acc)
#         print(aggregate_model.params)
#
#     print("Active party {} finished training.".format(active_party))
#     acc_summary.append(acc_list)
#     print("----------------------------------------------------")
# print("Accuracy summary" + repr(acc_summary))
# for i, result in enumerate(acc_summary):
#     print("Party {}: accuracy={}".format(i, np.average(result)))
