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
if not (os.path.isdir("data/UJIndoorLoc") and os.path.isfile("data/UJIndoorLoc/trainingData.csv")):
    print("Downloading UJ data")
    wget.download("https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIndoorLoc.zip",
                  "data/UJIndoorLoc.zip")
    with zipfile.ZipFile("data/UJIndoorLoc.zip", 'r') as zip_ref:
        zip_ref.extractall("data/")

# load data
xs_train, y_train, xs_test, y_test, x_train, x_test = load_data_train_test("UJIndoorLoc/trainingData.csv",
                                                                           num_parties=1,
                                                                           test_size=0.2, file_type='csv')

# calculate XGBoost feature importance
print("Starts training XGBoost on UJ")
xg_cls = xgb.XGBRegressor(objective='reg:squarederror',
                          learning_rate=0.1,
                          max_depth=6,
                          n_estimators=200,
                          reg_alpha=10,
                          verbosity=2)
xg_cls.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric='rmse')
y_pred = xg_cls.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
importance = xg_cls.feature_importances_
print("Finished training. Overall RMSE {}".format(rmse))

# save feature importance
np.savetxt("cache/feature_importance_uj.txt", importance)

# load importance from file
importance = np.loadtxt("cache/feature_importance_uj.txt")

# plot feature importance
print("Plot feature importance")
importance = np.asarray(importance)
plt.bar(list(range(len(importance))), np.sort(importance, axis=0), width=1.0)
plt.xlabel("Sorted Feature ID")
plt.ylabel("Feature Importance")
plt.title("Feature Importance Distribution (UJ)")
plt.gcf().savefig("_figure/feature_dist_uj")
plt.show()
plt.close()

def run_vertical_fl(beta):
    num_parties = 2
    cross_valid_data = load_data_cross_validation("UJIndoorLoc/trainingData.csv", num_parties=num_parties,
                                                  file_type='csv', n_fold=5, feature_order=np.argsort(importance),
                                                  num_good_features=100, good_feature_ratio_alpha=beta)
    active_party = 0
    print("Active party {} starts training".format(active_party))
    score_list = []
    for i, (xs_train, y_train, xs_test, y_test) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "vertical_fl_uj_party_{}_fold_{}_beta_{:.1f}".format(num_parties, i, beta)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=600,
            num_local_rounds=500,
            local_lr=3e-3,
            local_hidden_layers=[50, 50, 30],
            local_batch_size=100,
            local_weight_decay=1e-5,
            local_output_size=15,
            num_agg_rounds=1,
            agg_lr=3e-4,
            agg_hidden_layers=[30],
            agg_batch_size=100,
            agg_weight_decay=1e-5,
            writer=writer,
            device='cuda:0',
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
            delta=1.0/xs_train[0].shape[0]
        )
        _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_test, y_test, use_cache=False)
        print("Active party {} finished training.".format(active_party))
        score_list.append(rmse)
        print(aggregate_model.params)
    print("RMSE for active party {}".format(active_party) + str(score_list))
    mean = np.mean(score_list)
    std = np.std(score_list)
    _out = "Party {}, beta {:.1f}: RMSE mean={}, std={}".format(active_party, beta, mean, std)
    print(_out)
    return mean, std

betas = np.arange(0.0, 1.1, 0.1)
results = Parallel(n_jobs=-1)(delayed(run_vertical_fl)(beta) for beta in betas)
print("-------------------------------------------------")
for beta, (mean, std) in zip(betas, results):
    print("Party {}, beta {}: RMSE mean={}, std={}".format(0, beta, mean, std))


# single party
def run_single(beta):
    num_parties = 2
    cross_valid_data = load_data_cross_validation("UJIndoorLoc/trainingData.csv", num_parties=num_parties,
                                                  file_type='csv', n_fold=5, feature_order=np.argsort(importance),
                                                  num_good_features=100, good_feature_ratio_alpha=beta)
    party_id = 0
    print("Party {} starts training".format(party_id))
    score_list = []
    for i, (xs_train, y_train, xs_test, y_test) in enumerate(cross_valid_data):
        print("Cross Validation Fold {}".format(i))
        name = "single_uj_party_{}_fold_{}_beta_{:.1f}".format(num_parties, i, beta)
        writer = SummaryWriter("runs/{}".format(name))
        single_model = SingleParty(
            party_id=party_id,
            num_epochs=100,
            lr=3e-4,
            hidden_layers=[50, 50, 20],
            batch_size=100,
            weight_decay=1e-5,
            writer=writer,
            device='cuda:1',
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
        x_test = xs_test[party_id]
        _, _, rmse, _ = single_model.train(x_train, y_train, x_test, y_test)
        score_list.append(rmse)
        print(single_model.params)
    print("Accuracy for party {}".format(party_id) + str(score_list))
    mean = np.mean(score_list)
    std = np.std(score_list)
    out = "Party {}, beta {:.1f}: RMSE mean={}, std={}".format(party_id, beta, mean, std)
    print(out)
    return mean, std

betas = np.arange(0.0, 1.1, 0.1)
results = Parallel(n_jobs=-1)(delayed(run_single)(beta) for beta in betas)
print("-------------------------------------------------")
for beta, (mean, std) in zip(betas, results):
    print("Party {}, beta {:.1f}: RMSE mean={}, std={}".format(0, beta, mean, std))
