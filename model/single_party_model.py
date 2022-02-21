import time
from datetime import datetime

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


from model.models import FC, CNN, ResNet18, SmallCNN, NCF
from utils.data_utils import ImageDataset
from utils.exceptions import *
from privacy.eps_calculator import GradientDPCalculator

from torchdp.privacy_engine import PrivacyEngine

torch.manual_seed(0)

class SingleParty:
    def __init__(self, party_id, num_epochs=100, lr=1e-4, hidden_layers=None, batch_size=100, weight_decay=1e-4,
                 writer: SummaryWriter = None, device='cpu', task="binary_classification", model_type='fc',
                 test_batch_size=1000, n_classes=10, test_freq=1, n_channels=3, optimizer='sgd', cuda_parallel=False,
                 num_workers=0, momentum=0, privacy=None, batches_per_lot=1, epsilon=1, delta=0, grad_norm_C=1,
                 ncf_counts=None, ncf_embed_dims=None):

        self.ncf_counts = ncf_counts
        self.ncf_embed_dims = ncf_embed_dims
        self.grad_norm_C = grad_norm_C
        self.delta = delta
        self.epsilon = epsilon
        self.batches_per_lot = batches_per_lot
        self.privacy = privacy
        self.momentum = momentum
        self.num_workers = num_workers
        self.cuda_parallel = cuda_parallel
        self.optimizer = optimizer
        self.n_channels = n_channels
        self.model_type = model_type
        self.test_freq = test_freq
        self.n_classes = n_classes
        self.test_batch_size = test_batch_size
        self.task = task
        self.party_id = party_id
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.num_epochs = num_epochs
        self.writer = writer

        self.model = None

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = None

    def train(self, X, y, X_test, y_test):
        """
        Train one local party
        :param ep: index of epochs
        :param party_id: which party
        :param X: local data in party_id
        :param y: local labels
        :return: local prediction for X
        """
        if isinstance(X, csr_matrix):
            X = X.todense()
        if isinstance(y, csr_matrix):
            y = y.todense()
        num_instances = X.shape[0]

        if self.task in ["binary_classification", "regression"]:
            if self.model_type == 'fc':
                X_tensor = torch.from_numpy(X).float()
                y_tensor = torch.from_numpy(y).float()
            elif self.model_type == 'ncf':
                X_tensor = torch.from_numpy(X).long()
                y_tensor = torch.from_numpy(y).float()
            else:
                assert False
            dataset = TensorDataset(X_tensor, y_tensor)
        elif self.task in ["multi_classification"]:
            if self.model_type == 'fc':
                X_tensor = torch.from_numpy(X).float()
                y_tensor = torch.from_numpy(y).float()
                dataset = TensorDataset(X_tensor, y_tensor)
            else:
                # image classification, X is an ndarray of PIL images
                self.image_size = X.shape[1]
                if self.privacy is None:
                    transform_train = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.RandomCrop(self.image_size, padding=self.image_size // 8),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, ], std=[0.5, ])
                    ])
                else:
                    # Remove all data argumentation in DP
                    transform_train = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, ], std=[0.5, ])
                    ])
                dataset = ImageDataset([X], y, transform=transform_train)
        else:
            raise UnsupportedTaskError
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                 num_workers=self.num_workers, drop_last=True)

        if self.task in ["binary_classification", "regression"]:
            if self.model_type == 'fc':
                num_instances, num_features = X.shape
                if self.hidden_layers is None:
                    self.model = FC(num_features, [100, 100, 50], activation='sigmoid')
                else:
                    self.model = FC(num_features, self.hidden_layers, activation='sigmoid')
            elif self.model_type == 'ncf':
                self.model = NCF(self.ncf_counts, self.ncf_embed_dims, self.hidden_layers)
            else:
                assert False
        elif self.task in ["multi_classification"]:
            if self.model_type == 'resnet18':
                self.model = ResNet18(image_size=X.shape[1])
            elif self.model_type == 'cnn':
                if self.privacy is None:
                    dropout = True
                else:
                    dropout = False
                self.model = CNN(n_channels=self.n_channels, image_size=X.shape[1],
                                 kernel_size=3, stride=1, dropout=dropout)
                # self.model = SmallCNN(n_channels=self.n_channels, image_size=X.shape[1])
            elif self.model_type == 'fc':
                num_instances, num_features = X.shape
                if self.hidden_layers is None:
                    self.model = FC(num_features, [100, 100, 50], output_size=7, activation=None)
                else:
                    self.model = FC(num_features, self.hidden_layers, output_size=7, activation=None)
            else:
                raise False
        else:
            raise UnsupportedTaskError

        # define loss function
        model = self.model.to(self.device)
        if self.cuda_parallel:
            model = nn.DataParallel(model)

        if self.task == "binary_classification":
            loss_fn = nn.BCELoss()
        elif self.task == "regression":
            loss_fn = nn.MSELoss()
        elif self.task == "multi_classification":
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise UnsupportedTaskError

        # define optimizer
        if self.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        elif self.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'lamb':
            optimizer = adv_optim.Lamb(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise UnsupportedOptimizerError

        # if privacy is required, create dp calculator
        if self.privacy is None:
            dp_getter = None
        elif self.privacy == "MA":
            dp_getter = GradientDPCalculator(
                num_instances=num_instances,
                batch_size=self.batch_size * self.batches_per_lot,
                num_local_epochs=0,
                num_agg_epochs=self.num_epochs,
                epsilon=self.epsilon,
                delta=self.delta
            )
            privacy_engine = PrivacyEngine(
                module=model,
                batch_size=self.batch_size * self.batches_per_lot,
                sample_size=dp_getter.num_instances,
                alphas=[dp_getter.alpha],
                noise_multiplier=dp_getter.sigma,
                max_grad_norm=self.grad_norm_C
            )
            privacy_engine.attach(optimizer)
            print("Privacy engine attached, sigma={}".format(dp_getter.sigma))
        else:
            raise UnsupportedPrivacyMechanismError

        # test the accuracy
        best_test_acc = 0.0
        best_test_f1 = 0.0
        best_test_auc = 0.0
        best_test_rmse = np.inf
        total_loss = 0.0
        num_mini_batches = 0
        print("Start training")
        for i in range(self.num_epochs):
            start_epoch_time = datetime.now()
            self.model.train()
            for j, (X_i, y_i) in enumerate(data_loader, 0):
                X_i = X_i.to(self.device)
                y_i = y_i.to(self.device)
                optimizer.zero_grad()

                y_pred = model(X_i)

                if self.task in ["binary_classification", "regression"]:
                    loss = loss_fn(y_pred.view(-1), y_i)
                elif self.task == "multi_classification":
                    # image classification by default
                    loss = loss_fn(y_pred, y_i.long())
                else:
                    raise UnsupportedTaskError

                total_loss += loss.item()
                loss.backward()

                if self.privacy == 'MA' and ((j + 1) % self.batches_per_lot != 0) and (j + 1 < len(data_loader)):
                    optimizer.virtual_step()
                else:
                    optimizer.step()
                num_mini_batches += 1

            # if self.optimizer == "sgd":
            #     scheduler.step()

            if self.privacy is None:
                print("Party {}, Epoch {}: training loss {}"
                      .format(self.party_id, i + 1, total_loss / num_mini_batches))
                if self.writer:
                    self.writer.add_scalar('Party {}/training loss'.format(self.party_id),
                                           total_loss / num_mini_batches, i + 1)
            elif self.privacy == 'MA':
                epsilon, alpha = optimizer.privacy_engine.get_privacy_spent(self.delta)
                print("Party {}, Epoch {}: training loss {}, eps {}, delta {}, alpha {}"
                      .format(self.party_id, i + 1, total_loss / num_mini_batches,
                              epsilon, self.delta, alpha))
                if self.writer:
                    self.writer.add_scalar('Party {}/training loss'.format(self.party_id),
                                           total_loss / num_mini_batches, i + 1)
                    self.writer.add_scalar('Party {}/privacy accumulation'.format(self.party_id),
                                           epsilon, i + 1)
            else:
                raise UnsupportedPrivacyMechanismError

            total_loss = 0.0
            num_mini_batches = 0

            if X_test is not None and y_test is not None and (i + 1) % self.test_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    if self.task == "binary_classification":
                        y_pred_train, y_score_train = self.predict(X)
                        y_pred_test, y_score_test = self.predict(X_test)
                        train_acc = accuracy_score(y, y_pred_train)
                        test_acc = accuracy_score(y_test, y_pred_test)
                        train_f1 = f1_score(y, y_pred_train)
                        test_f1 = f1_score(y_test, y_pred_test)
                        train_auc = roc_auc_score(y, y_score_train)
                        test_auc = roc_auc_score(y_test, y_score_test)
                        if test_f1 > best_test_f1:
                            best_test_f1 = test_f1
                        if test_acc > best_test_acc:
                            best_test_acc = test_acc
                        if test_auc > best_test_auc:
                            best_test_auc = test_auc
                        print("[Final] Epoch {}: train accuracy {}, test accuracy {}".format(i + 1, train_acc, test_acc))
                        print("[Final] Epoch {}: train f1 {}, test f1 {}".format(i + 1, train_f1, test_f1))
                        print("[Final] Epoch {}: train auc {}, test auc {}".format(i + 1, train_auc, test_auc))
                        print("[Final] Epoch {}: best test acc {}, best test f1 {}, best test auc {}"
                              .format(i + 1, best_test_acc, best_test_f1, best_test_auc))
                        if self.writer is not None:
                            self.writer.add_scalars("Aggregation model/train & test accuracy",
                                                    {'train': train_acc,
                                                     'test': test_acc}, i + 1)
                            self.writer.add_scalars("Aggregation model/train & test F1-score",
                                                    {'train': train_f1,
                                                     'test': test_f1}, i + 1)
                            self.writer.add_scalars("Aggregation model/train & test AUC score",
                                                    {'train': train_auc,
                                                     'test': test_auc}, i + 1)
                    elif self.task == "regression":
                        y_pred_train, y_score_train = self.predict(X)
                        y_pred_test, y_score_test = self.predict(X_test)
                        train_rmse = np.sqrt(mean_squared_error(y, y_score_train))
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_score_test))
                        if test_rmse < best_test_rmse:
                            best_test_rmse = test_rmse
                        print("[Final] Epoch {}: train rmse {}, test rmse {}".format(i + 1, train_rmse, test_rmse))
                        print("[Final] Epoch {}: best test rmse {}".format(i + 1, best_test_rmse))
                        if self.writer is not None:
                            self.writer.add_scalars("Aggregation model/train & test rmse",
                                                    {'train': train_rmse,
                                                     'test': test_rmse}, i + 1)
                    elif self.task == "multi_classification":
                        if self.model_type == 'fc':
                            train_acc = self.eval_image(X, y, ['accuracy'], has_transform=False)
                            test_acc = self.eval_image(X_test, y_test, ['accuracy'], has_transform=False)
                        else:
                            train_acc = self.eval_image(X, y, ['accuracy'], has_transform=True)
                            test_acc = self.eval_image(X_test, y_test, ['accuracy'], has_transform=True)
                        if test_acc > best_test_acc:
                            best_test_acc = test_acc
                        print("[Final] Epoch {}: train accuracy {}, test accuracy {}".format(i + 1, train_acc, test_acc))
                        print("[Final] Epoch {}: best test accuracy {}".format(i + 1, best_test_acc))
                        if self.writer is not None:
                            self.writer.add_scalars("Aggregation model/train & test accuracy",
                                                    {'train': train_acc,
                                                     'test': test_acc}, i + 1)
                    else:
                        raise UnsupportedTaskError
            epoch_duration_sec = (datetime.now() - start_epoch_time).seconds
            print("Epoch {} duration {} sec".format(i + 1, epoch_duration_sec), flush=True)
        return best_test_acc, best_test_f1, best_test_rmse, best_test_auc

    def predict(self, X):
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_score = self.model.to(self.device)(X_tensor).detach().cpu().numpy()
        y_pred = np.where(y_score > 0.5, 1, 0)
        return y_pred, y_score

    def eval_image(self, X: np.ndarray, y: np.ndarray, metrics: list, has_transform=True):
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

        num_instances = X.shape[0]

        if has_transform:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, ], std=[0.5, ])
            ])
            dataset = ImageDataset([X], y, transform=transform_test)
        else:
            dataset = TensorDataset(torch.from_numpy(X).float(), torch.tensor(y))
        dataloader = DataLoader(dataset, self.test_batch_size, shuffle=False, num_workers=self.num_workers)

        results = np.zeros([len(metrics)])
        ep_cnt = 0

        correct = 0
        for X_i, y_i in dataloader:
            X_i = X_i.to(self.device)
            y_i = y_i.detach().cpu().numpy()

            y_score = self.model.to(self.device)(X_i).detach().cpu().numpy()

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