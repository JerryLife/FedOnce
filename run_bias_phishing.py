from utils.data_utils import load_data_cross_validation, load_data_train_test
from model.fl_model import VerticalFLModel
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel
from model.models import FC

from torch.utils.tensorboard import SummaryWriter
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
import torch

import os.path
import wget
import bz2
import shutil
import zipfile
import numpy as np

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt

if not os.path.isdir("data"):
    os.mkdir("data")
if not os.path.isfile("data/phishing"):
    print("Downloading phishing data")
    wget.download("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing",
                  "data/phishing")


# load data
xs_train, y_train, xs_test, y_test, x_train, x_test = load_data_train_test("phishing", num_parties=1,
                                                                           test_size=0.2, file_type='libsvm')

# calculate XGBoost feature importance
print("Starts training XGBoost on phishing")
xg_cls = xgb.XGBClassifier(objective='binary:logistic',
                           learning_rate=0.1,
                           max_depth=6,
                           n_estimators=100,
                           reg_alpha=10,
                           verbosity=2)
xg_cls.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric='error')
y_pred = xg_cls.predict(x_test)
acc = accuracy_score(y_test, y_pred)
importance = xg_cls.feature_importances_
print("Finished training. Overall accuracy {}".format(acc))

# save feature importance
np.savetxt("cache/feature_importance_phishing.txt", importance)

# load importance from file
importance = np.loadtxt("cache/feature_importance_phishing.txt")


# FedOnce
def run_vertical_fl(beta):
    num_parties = 2
    cross_valid_data = load_data_cross_validation("phishing", num_parties=num_parties,
                                                  file_type='libsvm', n_fold=5, feature_order=np.argsort(importance),
                                                  num_good_features=30, good_feature_ratio_alpha=beta)
    active_party = 0
    print("Active party {} starts training".format(active_party))
    score_list = []
    for i, (xs_train, y_train, xs_test, y_test) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "vertical_fl_phishing_party_{}_fold_{}_beta_{:.1f}".format(num_parties, i, beta)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=100,
            num_local_rounds=100,
            local_lr=3e-4,
            local_hidden_layers=[50, 30],
            local_batch_size=100,
            local_weight_decay=1e-5,
            local_output_size=3,
            num_agg_rounds=1,
            agg_lr=1e-4,
            agg_hidden_layers=[10],
            agg_batch_size=100,
            agg_weight_decay=1e-4,
            writer=writer,
            device='cuda:0',
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
            delta=1.0/xs_train[0].shape[0]
        )
        acc, _, _, _ = aggregate_model.train(xs_train, y_train, xs_test, y_test, use_cache=False)
        print("Active party {} finished training.".format(active_party))
        score_list.append(acc)
        print(aggregate_model.params)
    print("Accuracy for active party {}".format(active_party) + str(score_list))
    mean = np.mean(score_list)
    std = np.std(score_list)
    out = "Party {}, beta {:.1f}: Accuracy mean={}, std={}".format(active_party, beta, mean, std)
    print(out)
    return mean, std

betas = np.arange(0.0, 1.1, 0.1)
results = Parallel(n_jobs=6)(delayed(run_vertical_fl)(beta) for beta in betas)
print("-------------------------------------------------")
for beta, (mean, std) in zip(betas, results):
    print("Party {}, beta {:.1f}: Accuracy mean={}, std={}".format(0, beta, mean, std))


# Solo
def run_single(beta):
    num_parties = 2
    cross_valid_data = load_data_cross_validation("phishing", num_parties=num_parties,
                                                  file_type='libsvm', n_fold=5, feature_order=np.argsort(importance),
                                                  num_good_features=30, good_feature_ratio_alpha=beta)
    party_id = 0
    print("Party {} starts training".format(party_id))
    score_list = []
    for i, (xs_train, y_train, xs_test, y_test) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        name = "single_phishing_party_{}_single_{}_fold_{}_beta_{:.1f}".format(num_parties, party_id, i, beta)
        writer = SummaryWriter("runs/{}".format(name))
        single_model = SingleParty(
            party_id=party_id,
            num_epochs=100,
            lr=1e-4,
            hidden_layers=[50, 30],
            batch_size=100,
            weight_decay=1e-4,
            writer=writer,
            device='cuda:1',
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
        acc, _, _, _ = single_model.train(x_train, y_train, x_test, y_test)
        score_list.append(acc)
        print(single_model.params)
    print("Accuracy for party {}".format(party_id) + str(score_list))
    mean = np.mean(score_list)
    std = np.std(score_list)
    _out = "Party {}, beta {}: Acc mean={}, std={}".format(party_id, beta, mean, std)
    print(_out)
    return mean, std

betas = np.arange(0.0, 1.1, 0.1)
results = Parallel(n_jobs=11)(delayed(run_single)(beta) for beta in betas)
print("-------------------------------------------------")
for beta, (mean, std) in zip(betas, results):
    print("Party {}, beta {:.1f}: Accuracy mean={}, std={}".format(0, beta, mean, std))
#

# combine
num_parties = 1
cross_valid_data = load_data_cross_validation("phishing", num_parties=num_parties,
                                              file_type='libsvm', n_fold=5)
f1_summary = []
acc_summary = []
for party_id in range(num_parties):
    print("Party {} starts training".format(party_id))
    acc_list = []
    f1_list = []
    for i, (xs_train, y_train, xs_test, y_test) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        name = "combine_phishing_fold_{}".format(i)
        writer = SummaryWriter("runs/{}".format(name))
        single_model = SingleParty(
            party_id=party_id,
            num_epochs=100,
            lr=1e-4,
            hidden_layers=[100, 50],
            batch_size=100,
            weight_decay=1e-4,
            writer=writer,
            device='cuda:0',
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
        acc, f1, _ = single_model.train(x_train, y_train, x_test, y_test)
        acc_list.append(acc)
        f1_list.append(f1)
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
for i, result in enumerate(acc_summary):
    mean = np.mean(result)
    std = np.std(result)
    print("Party {}: F1-score mean={}, std={}".format(i, mean, std))
