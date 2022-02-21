import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms
import torch_optimizer as adv_optim
from scipy.sparse import csr_matrix

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score

from torchdp.privacy_engine import PrivacyEngine

import os.path
from datetime import datetime

from model.models import FC, AggModel, CNN, ResNet18, NCF
from utils.utils import generate_random_targets, calc_optimal_target_permutation, \
    is_perturbation
from utils.data_utils import LocalDataset, AggDataset, ImageDataset
from utils.utils import convert_name_to_path, Log
from utils.exceptions import *
from privacy.eps_calculator import GradientDPCalculator


class SplitNN(nn.Module):
    def __init__(self, local_models, agg_model):
        super().__init__()
        self.agg_model = agg_model
        self.local_models = nn.ModuleList(local_models)

    def forward(self, Xs):
        local_outputs = [model(X) for X, model in zip(Xs, self.local_models)]
        local_out = torch.cat(local_outputs, dim=1)
        out = self.agg_model(local_out)
        return out


class SplitNNModel:
    def __init__(self, num_parties, name="", num_epochs=100, lr=1e-4, weight_decay=1e-4,
                 local_hidden_layers=None, batch_size=100, local_output_size=3, agg_hidden_layers=None,
                 writer: SummaryWriter = None, device='cpu', update_target_freq=1,
                 task='binary_classification', n_classes=1, test_batch_size=1000, test_freq=1, cuda_parallel=False,
                 n_channels=1, model_type='fc', optimizer='sgd', num_workers=0,
                 privacy=None, batches_per_lot=1, epsilon=1, delta=1e-4, grad_norm_C=1.0,
                 ncf_counts=None, ncf_embed_dims=None, cache_local_name="", cache_agg_name="", active_party=None):


        self.active_party = active_party# only used for loading cache
        self.cache_agg_name = cache_agg_name
        self.cache_local_name = cache_local_name
        self.ncf_embed_dims = ncf_embed_dims
        self.ncf_counts = ncf_counts
        self.num_workers = num_workers
        self.grad_norm_C = grad_norm_C
        self.delta = delta
        self.epsilon = epsilon
        self.privacy = privacy
        self.optimizer = optimizer
        self.model_type = model_type
        self.n_channels = n_channels
        self.cuda_parallel = cuda_parallel
        self.test_freq = test_freq
        self.test_batch_size = test_batch_size
        self.n_classes = n_classes
        self.task = task
        self.name = name
        self.update_target_freq = update_target_freq
        self.local_output_dim = local_output_size
        self.weight_decay = weight_decay
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_parties = num_parties
        self.writer = writer

        if agg_hidden_layers is None:
            self.agg_hidden_layers = [10, 10]
        else:
            self.agg_hidden_layers = agg_hidden_layers

        if local_hidden_layers is None:
            self.local_hidden_layers = [100, 100, 50]
        else:
            self.local_hidden_layers = local_hidden_layers

        self.local_models = []
        self.agg_model = None
        self.local_labels = None

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = None

        self.split_nn = None

        self.comm_size = 0
        if not os.path.isdir("log"):
            os.mkdir("log")
        self.log = Log("log/LOG_comm_round_{}".format(name))
        self.log.write("Comm_size\tAccuracy\tF1-score\tRMSE")

    @staticmethod
    def load_fedonce_agg_model(fedonce_agg_model_path):
        state_dict = torch.load(fedonce_agg_model_path, map_location=lambda storage, location: storage)
        local_state_dict = {k.replace("active_model.", ""): v for k, v in state_dict.items() if k.startswith("active_model")}
        agg_state_dict = {k.replace("agg_fc_layers", "fc_layers"): v for k, v in state_dict.items()
                          if k.startswith("agg_fc_layers")}
        return local_state_dict, agg_state_dict

    @staticmethod
    def mse_loss(x, y):
        return torch.mean(torch.sum((x - y) ** 2, dim=1))

    def train(self, Xs, y, Xs_val=None, y_val=None, Xs_test=None, y_test=None, use_cache=False):
        if use_cache and not os.path.isdir('cache'):
            os.mkdir('cache')

        num_instances = Xs[0].shape[0]

        if use_cache:
            agg_model_path = "cache/{}_agg_model_dim_{}.pth".format(self.cache_agg_name, self.local_output_dim)
            active_state_dict, agg_state_dict = SplitNNModel.load_fedonce_agg_model(agg_model_path)

        # initialize every local model
        for party_id in range(self.num_parties):
            # initialize local model
            num_features = Xs[party_id].shape[1]
            if self.task in ["binary_classification", "regression"]:
                if self.model_type == 'fc':
                    local_model = FC(num_features, self.local_hidden_layers, output_size=self.local_output_dim)
                elif self.model_type == 'ncf':
                    local_model = NCF(self.ncf_counts[party_id], self.ncf_embed_dims[party_id],
                                      self.local_hidden_layers, output_size=self.local_output_dim)
                else:
                    assert False
            elif self.task in ["multi_classification"]:
                if self.model_type != 'fc':
                    assert Xs[0].shape[1] == Xs[0].shape[2], "Only support square figures"
                    self.image_size = Xs[0].shape[1]
                if self.model_type == 'resnet18':
                    local_model = ResNet18(image_size=self.image_size, num_classes=self.local_output_dim)
                elif self.model_type == 'cnn':
                    local_model = CNN(n_channels=self.n_channels, image_size=Xs[party_id].shape[1],
                                      output_dim=self.local_output_dim)
                elif self.model_type == 'fc':
                    local_model = FC(num_features, self.local_hidden_layers, output_size=self.local_output_dim)
                else:
                    raise UnsupportedModelError
            else:
                raise UnsupportedTaskError

            if use_cache:
                if party_id == self.active_party:
                    local_model.load_state_dict(active_state_dict)
                else:
                    fmt_name = convert_name_to_path(self.cache_local_name)
                    local_model_path = "cache/{}_model_{}_dim_{}.pth".format(fmt_name, party_id, self.local_output_dim)
                    local_model.load_state_dict(torch.load(local_model_path,
                                                           map_location=lambda storage, location: storage))

            self.local_models.append(local_model.to(self.device))

        # initialize agg model
        if self.task in ["binary_classification", "regression"]:
            if self.model_type == 'fc':
                self.agg_model = FC(self.local_output_dim * self.num_parties,
                                    hidden_sizes=self.agg_hidden_layers,
                                    output_size=1,
                                    activation='sigmoid')
            elif self.model_type == 'ncf':
                self.agg_model = FC(self.local_output_dim * self.num_parties,
                                    hidden_sizes=self.agg_hidden_layers,
                                    output_size=1,
                                    activation=None)
            else:
                assert False
        elif self.task in ["multi_classification"]:
            self.agg_model = FC(self.local_output_dim * self.num_parties,
                                hidden_sizes=self.agg_hidden_layers,
                                output_size=self.n_classes,
                                activation=None)
        else:
            raise UnsupportedTaskError

        if use_cache:
            self.agg_model.load_state_dict(agg_state_dict)

        # initialize splitNN
        self.split_nn = SplitNN(self.local_models, self.agg_model.to(self.device))

        # train splitNN
        print("Start training SplitNN")
        # optimizer should be defined here for once due to torchdp
        model = self.split_nn
        model = model.to(self.device)
        if self.cuda_parallel:
            model = nn.DataParallel(model)

        if self.task in ["binary_classification", "regression"]:
            Xs_tensor = [torch.from_numpy(X).float() for X in Xs]
            y_tensor = torch.from_numpy(y).float()
            dataset = TensorDataset(*Xs_tensor, y_tensor)
        elif self.task in ["multi_classification"]:
            if self.model_type == 'fc':
                Xs_tensor = [torch.from_numpy(X).float() for X in Xs]
                y_tensor = torch.from_numpy(y).float()
                dataset = TensorDataset(*Xs_tensor, y_tensor)
            else:
                # image classification, X is an ndarray of PIL images
                self.image_size = Xs[0].shape[1]
                transform_train = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomCrop(self.image_size, padding=self.image_size // 8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, ], std=[0.5, ])
                ])
                dataset = ImageDataset(Xs, y, transform=transform_train)
        else:
            raise UnsupportedTaskError
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        # define loss function
        if self.task == "binary_classification":
            loss_fn = nn.BCELoss()
        elif self.task == "regression":
            loss_fn = nn.MSELoss()
        elif self.task == "multi_classification":
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise UnsupportedTaskError

        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.split_nn.parameters(), lr=self.lr,
                                   weight_decay=self.weight_decay)
            scheduler = None
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.split_nn.parameters(), lr=self.lr,
                                  momentum=0.9, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        elif self.optimizer == 'lamb':
            optimizer = adv_optim.Lamb(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise UnsupportedOptimizerError

        # training
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_val_auc = 0.0
        best_val_rmse = np.inf
        self.comm_size = 0
        best_test_acc = 0.0
        best_test_rmse = np.inf
        for ep in range(self.num_epochs):
            start_epoch_time = datetime.now()
            model.train()
            total_loss = 0.0
            num_batches = 0
            for _, data in enumerate(data_loader):
                Xs_i = [X.to(self.device) for X in data[:-1]]
                y_i = data[-1].to(self.device)

                optimizer.zero_grad()
                y_pred = model(Xs_i)
                # forward propagation communication size in bytes
                self.comm_size += (model.agg_model.fc_layers[0].in_features
                                  * np.dtype(np.float32).itemsize)

                if self.task in ["binary_classification", "regression"]:
                    loss = loss_fn(y_pred.view(-1), y_i)
                elif self.task == "multi_classification":
                    # image classification by default
                    loss = loss_fn(y_pred, y_i.long())
                else:
                    raise UnsupportedTaskError

                total_loss += loss.item()
                num_batches += 1
                loss.backward()
                optimizer.step()

                # backward propagation size in bytes
                self.comm_size += (np.prod(model.agg_model.fc_layers[0].weight.grad.shape) +
                                   np.prod(model.agg_model.fc_layers[0].bias.grad.shape)) \
                                   * np.dtype(np.float32).itemsize

            if self.optimizer == 'sgd':
                scheduler.step(1)

            if self.privacy is None:
                print("[SplitNN] Epoch {}: training loss {}"
                      .format(ep + 1, total_loss / num_batches))
            elif self.privacy == 'MA':
                epsilon, alpha = optimizer.privacy_engine.get_privacy_spent(self.delta)
                print("[SplitNN] Epoch {}: training loss {}, eps {}, delta {}, alpha {}"
                      .format(ep + 1, total_loss / num_batches, epsilon, self.delta, alpha))
                self.writer.add_scalar('Aggregation privacy accumulation', epsilon, ep + 1)
            else:
                raise UnsupportedPrivacyMechanismError

            if self.writer:
                self.writer.add_scalar('Aggregation training loss', total_loss / num_batches, ep + 1)

            # test model
            if Xs_val is not None and y_val is not None and (ep + 1) % self.test_freq == 0:
                model.eval()
                with torch.no_grad():
                    if self.task == 'binary_classification':
                        y_pred_train, y_score_train = self.predict(Xs)
                        y_pred_val, y_score_val = self.predict(Xs_val)
                        y_pred_test, y_score_test = self.predict(Xs_test)
                        train_acc = accuracy_score(y, y_pred_train)
                        val_acc = accuracy_score(y_val, y_pred_val)
                        test_acc = accuracy_score(y_test, y_pred_test)
                        train_f1 = f1_score(y, y_pred_train) 
                        val_f1 = f1_score(y_val, y_pred_val) 
                        train_auc = roc_auc_score(y, y_score_train)
                        val_auc = roc_auc_score(y_val, y_score_val)
                        if val_f1 > best_val_f1:
                            best_val_f1 = val_f1
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_test_acc = test_acc
                        if val_auc > best_val_auc:
                            best_val_auc = val_auc
                        print(
                            "[Final] Epoch {}: train accuracy {}, val accuracy {}".format(ep + 1, train_acc, val_acc))
                        print("[Final] Epoch {}: train f1 {}, val f1 {}".format(ep + 1, train_f1, val_f1))
                        print("[Final] Epoch {}: best val acc {}, best val f1 {}".format(ep + 1, best_val_acc,
                                                                                           best_val_f1))
                        print("[Final] Epoch {}: train auc {}, test f1 {}".format(ep + 1, train_auc, val_auc))
                        self.writer.add_scalars("Aggregation model/train & test accuracy",
                                                {'train': train_acc,
                                                 'test': val_acc}, ep + 1)
                        self.writer.add_scalars("Aggregation model/train & test F1-score",
                                                {'train': train_f1,
                                                 'test': val_f1}, ep + 1)
                        self.writer.add_scalars("Aggregation model/train & test AUC score",
                                                {'train': train_auc,
                                                 'test': val_auc}, ep + 1)
                    elif self.task == 'regression':
                        y_pred_train, y_score_train = self.predict(Xs)
                        y_pred_val, y_score_val = self.predict(Xs_val)
                        y_pred_test, y_score_test = self.predict(Xs_test)
                        train_rmse = np.sqrt(mean_squared_error(y, y_score_train))
                        val_rmse = np.sqrt(mean_squared_error(y_val, y_score_val))
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_score_test))
                        if val_rmse < best_val_rmse:
                            best_val_rmse = val_rmse
                            best_test_rmse = test_rmse
                        print("[Final] Epoch {}: train rmse {}, val rmse {}".format(ep + 1, train_rmse, val_rmse))
                        print("[Final] Epoch {}: best val rmse {}".format(ep + 1, best_val_rmse))
                        self.writer.add_scalars("Aggregation model/train & test rmse",
                                                {'train': train_rmse,
                                                 'test': val_rmse}, ep + 1)
                    elif self.task == 'multi_classification':
                        if self.model_type == 'fc':
                            train_acc = self.eval_image(Xs, y, ['accuracy'], has_transform=False)
                            val_acc = self.eval_image(Xs_val, y_val, ['accuracy'], has_transform=False)
                            test_acc = self.eval_image(Xs_test, y_test, ['accuracy'], has_transform=False)
                        else:
                            train_acc = self.eval_image(Xs, y, ['accuracy'], has_transform=True)
                            val_acc = self.eval_image(Xs_val, y_val, ['accuracy'], has_transform=True)
                            test_acc = self.eval_image(Xs_test, y_test, ['accuracy'], has_transform=True)
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_test_acc = test_acc
                        print(
                            "[Final] Epoch {}: train accuracy {}, test accuracy {}".format(ep + 1, train_acc, val_acc))
                        print("[Final] Epoch {}: best test acc {}".format(ep + 1, best_val_acc))
                        self.writer.add_scalars("Aggregation model/train & test accuracy",
                                                {'train': train_acc,
                                                 'test': val_acc}, ep + 1)
                    else:
                        raise UnsupportedTaskError
                    self.log.write("{}\t{}\t{}\t{}".format(self.comm_size, best_test_acc,
                                                           0, best_test_rmse))
            epoch_duration_sec = (datetime.now() - start_epoch_time).seconds
            print("Epoch {} duration {} sec".format(ep + 1, epoch_duration_sec), flush=True)
        return best_val_acc, best_val_f1, best_val_rmse, best_val_auc

    def predict(self, Xs):
        Xs_tensor = [torch.from_numpy(X).float().to(self.device) for X in Xs]
        y_score = self.split_nn.to(self.device)(Xs_tensor).detach().cpu().numpy()
        y_pred = np.where(y_score > 0.5, 1, 0)
        return y_pred, y_score

    def eval_image(self, Xs: list, y: np.ndarray, metrics: list, has_transform=True):
        # deal with metrics
        metric_func_list = []
        acc_metric_id = -1
        for i, metric in enumerate(metrics):
            if metric == 'f1_score':
                if self.task == 'binary_classification':
                    metric_func = lambda y, y_score: f1_score(y, np.where(y_score > 0.5, 1, 0))
                elif self.task == 'multi_classification':
                    metric_func = lambda y, y_score: f1_score(y, np.argmax(y_score, axis=1), average='micro')
                else:
                    raise UnsupportedTaskError
                metric_func_list.append(metric_func)
            elif metric == 'accuracy':
                # accuracy is calculated independently
                acc_metric_id = i
                metric_func = lambda y, y_score: 0
                metric_func_list.append(metric_func)
            elif metric == 'rmse':
                metric_func = lambda y, y_score: np.sqrt(mean_squared_error(y, y_score))
                metric_func_list.append(metric_func)
            else:
                raise UnsupportedMetricError

        num_instances = Xs[0].shape[0]
        if has_transform:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, ], std=[0.5, ])
            ])
            dataset = ImageDataset(Xs, y, transform=transform_test)
        else:
            Xs = [torch.from_numpy(X).float() for X in Xs]
            y = torch.tensor(y)
            dataset = ImageDataset(Xs, y, transform=None)
        dataloader = DataLoader(dataset, self.test_batch_size, shuffle=False, num_workers=self.num_workers)

        results = np.zeros([len(metrics)])
        ep_cnt = 0

        correct = 0
        for data in dataloader:
            Xs_i = [X.to(self.device) for X in data[:-1]]
            y_i = data[-1].detach().cpu().numpy()

            y_score = self.split_nn.to(self.device)(Xs_i).detach().cpu().numpy()

            if self.task == 'binary_classification':
                y_pred = np.where(y_score > 0.5, 1, 0)
            elif self.task == 'multi_classification':
                y_pred = np.argmax(y_score, axis=1)
            else:
                raise UnsupportedTaskError
            correct += (y_pred == y_i).sum()

            for j in range(len(metrics)):
                results[j] += metric_func_list[j](y_i, y_score)
            ep_cnt += 1

        results /= ep_cnt
        if acc_metric_id >= 0:
            results[acc_metric_id] = correct / num_instances

        if len(results) == 1:
            return results[0]
        else:
            return results

    @property
    def params(self):
        attrs = vars(self)
        output = '\n'.join("%s=%s" % item for item in attrs.items())
        return output
