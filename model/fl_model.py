import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms

from datetime import datetime

torch.manual_seed(0)

from scipy.sparse import csr_matrix

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score

from torchdp.privacy_engine import PrivacyEngine
import torch_optimizer as adv_optim

import os.path

from model.models import FC, AggModel, CNN, ResNet18, SmallCNN, BareSmallCNN, NCF
from utils.utils import generate_random_targets, calc_optimal_target_permutation, \
    is_perturbation
from utils.data_utils import LocalDataset, AggDataset, ImageDataset
from utils.utils import convert_name_to_path
from utils.exceptions import *
from privacy.eps_calculator import GradientDPCalculator, GaussianDPCalculator


class VerticalFLModel:
    def __init__(self, num_parties, active_party_id, name="", num_epochs=100, num_local_rounds=1, local_lr=1e-4,
                 local_hidden_layers=None, local_batch_size=100, local_weight_decay=1e-4, local_output_size=3,
                 num_agg_rounds=1, agg_lr=1e-4, agg_hidden_layers=None, agg_batch_size=100, agg_weight_decay=1e-4,
                 writer: SummaryWriter = None, device='cpu', update_target_freq=1,
                 task='binary_classification', n_classes=1, test_batch_size=1000, test_freq=1, cuda_parallel=False,
                 n_channels=1, model_type='fc', optimizer='sgd', momentum=0, num_workers=0,
                 privacy=None, batches_per_lot=1, epsilon=1, delta=1e-4, grad_norm_C=1.0, inter_party_comp_method=None,
                 ncf_counts=None, ncf_embed_dims=None, full_name="", out_privacy=None, out_norm=1., repr_noise=0.0,
                 train_order=None):

        self.repr_noise = repr_noise
        self.out_norm = out_norm
        self.out_privacy = out_privacy
        self.full_name = full_name
        self.ncf_embed_dims = ncf_embed_dims
        self.ncf_counts = ncf_counts
        self.inter_party_comp_method = inter_party_comp_method
        self.num_workers = num_workers
        self.momentum = momentum
        self.batches_per_lot = batches_per_lot
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
        self.active_party_id = active_party_id
        self.name = name
        self.update_target_freq = update_target_freq
        self.local_output_dim = local_output_size
        self.agg_batch_size = agg_batch_size
        self.agg_weight_decay = agg_weight_decay
        self.agg_lr = agg_lr
        self.num_agg_rounds = 1
        self.local_weight_decay = local_weight_decay
        self.local_batch_size = local_batch_size
        self.local_lr = local_lr
        self.num_local_rounds = num_local_rounds
        self.num_epochs = num_epochs
        self.num_parties = num_parties
        self.writer = writer
        self.train_order = train_order

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

        self.device_name = device if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)

        self.image_size = None

        self.local_dp_getter = None
        self.agg_dp_getter = None

    @staticmethod
    def mse_loss(x, y):
        return torch.mean(torch.sum((x - y) ** 2, dim=1))

    def train_local_party(self, ep, party_id, X, y, model):
        """
        Train one local party
        :param ep: index of epochs
        :param party_id: which party
        :param X: local data in party_id
        :param y: local labels: These labels will be updated!
        :return: local prediction for X
        """
        if isinstance(X, csr_matrix):
            X = X.todense()
        if isinstance(y, csr_matrix):
            y = y.todense()
        y_copy = y.copy()
        num_instances = X.shape[0]
        start_local_time = datetime.now()

        # define data and dataloader
        if self.task in ["binary_classification", "regression"]:
            X_tensor = torch.from_numpy(X).float()
            y_tensor = torch.from_numpy(y).float()  # y will be updated
            dataset = LocalDataset(X_tensor, y_tensor)
        elif self.task in ["multi_classification"]:
            # image classification, X is an ndarray of PIL images
            if self.privacy is None:
                if self.model_type == 'fc':
                    X_tensor = torch.from_numpy(X).float()
                    y_tensor = torch.from_numpy(y).float()  # y will be updated
                    dataset = LocalDataset(X_tensor, y_tensor)
                else:
                    transform_train = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.RandomCrop(self.image_size, padding=self.image_size // 8),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, ], std=[0.5, ])
                    ])
                    dataset = LocalDataset(X, y, transform=transform_train)
            else:
                transform_train = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, ], std=[0.5, ])
                ])
                dataset = LocalDataset(X, y, transform=transform_train)
        else:
            raise UnsupportedTaskError
        data_loader = DataLoader(dataset, batch_size=self.local_batch_size, shuffle=True,
                                 drop_last=True, num_workers=self.num_workers,
                                 multiprocessing_context='fork' if self.num_workers > 0 else None)

        # define model
        model = model.to(self.device)
        if self.cuda_parallel:
            model = nn.DataParallel(model)

        loss_fn = self.mse_loss

        # define optimizer
        if self.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.local_lr, weight_decay=self.local_weight_decay)
        elif self.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=self.local_lr, momentum=self.momentum,
                                  weight_decay=self.local_weight_decay)
        elif self.optimizer == 'lamb':
            optimizer = adv_optim.Lamb(model.parameters(), lr=self.local_lr, weight_decay=self.local_weight_decay)
        else:
            raise UnsupportedTaskError

        # attach privacy engine to optimizer if required
        if self.privacy == 'MA':
            privacy_engine = PrivacyEngine(
                module=model,
                batch_size=self.local_dp_getter.batch_size,
                sample_size=self.local_dp_getter.num_instances,
                alphas=[self.local_dp_getter.alpha],
                noise_multiplier=self.local_dp_getter.sigma,
                max_grad_norm=self.grad_norm_C
            )
            privacy_engine.attach(optimizer)

        total_loss = 0.0
        num_mini_batches = 0
        for i in range(self.num_local_rounds):
            start_epoch = datetime.now()
            update_targets = ((i + 1) % self.update_target_freq == 0)
            for j, (idx, X_i, y_i) in enumerate(data_loader, 0):
                X_i = X_i.to(self.device)
                y_i = y_i.to(self.device)

                optimizer.zero_grad()

                y_pred = model(X_i)

                if update_targets:
                    output = y_pred.cpu().detach().numpy()
                    new_targets = calc_optimal_target_permutation(output, y_i.cpu().detach().numpy())
                    new_targets_tensor = torch.from_numpy(new_targets)
                    dataset.update_targets(idx, new_targets_tensor)
                    y_i = torch.from_numpy(new_targets).to(self.device)

                loss = loss_fn(y_pred, y_i)

                total_loss += loss.item()
                loss.backward()

                if self.privacy == 'MA' and ((j + 1) % self.batches_per_lot != 0) and (j + 1 < len(data_loader)):
                    optimizer.virtual_step()
                else:
                    optimizer.step()
                num_mini_batches += 1

            if self.privacy is None:
                print("[Local] Party {}, Epoch {}: training loss {}"
                      .format(party_id, ep * self.num_local_rounds + i + 1, total_loss / num_mini_batches))
            elif self.privacy == 'MA':
                epsilon, alpha = optimizer.privacy_engine.get_privacy_spent(self.delta)
                print("[Local] Party {}, Epoch {}: training loss {}, eps {}, delta {}, alpha {}"
                      .format(party_id, ep * self.num_local_rounds + i + 1, total_loss / num_mini_batches,
                              epsilon, self.delta, alpha))
                self.writer.add_scalar('Party {}/privacy accumulation'.format(party_id),
                                       epsilon, ep * self.num_agg_rounds + i + 1)
            else:
                raise UnsupportedPrivacyMechanismError

            if self.writer:
                self.writer.add_scalar('Party {}/training loss'.format(party_id),
                                       total_loss / num_mini_batches, ep * self.num_local_rounds + i + 1)
            total_loss = 0.0
            num_mini_batches = 0
            epoch_duration_sec = (datetime.now() - start_epoch).seconds
            print("Epoch {} duration {} sec".format(i + 1, epoch_duration_sec), flush=True)

        duration_local = (datetime.now() - start_local_time).seconds
        print("Local training finished, time = {} sec".format(duration_local))

        assert is_perturbation(y, y_copy)
        return model

    def train_aggregation(self, ep, Z, X_active, y, optimizer, scheduler=None):
        """
        :param ep: index of epochs
        :param Z: local labels except active party: will be changed in function!
        :param y: real labels
        :return: updated input (updated local labels)
        """
        if isinstance(Z, csr_matrix):
            Z = Z.todense()
        if isinstance(y, csr_matrix):
            y = y.todense()
        Z_copy = Z.copy()
        num_instances = X_active.shape[0]

        if self.task in ["binary_classification", "regression"]:
            # party 0 is active party
            X_tensor = torch.from_numpy(X_active).float()
            Z_tensor = torch.from_numpy(Z).float()
            y_tensor = torch.from_numpy(y).float()
            dataset = AggDataset(X_tensor, Z_tensor, y_tensor)
        elif self.task in ["multi_classification"]:
            if self.model_type == 'fc':
                X_tensor = torch.from_numpy(X_active).float()
                Z_tensor = torch.from_numpy(Z).float()
                y_tensor = torch.from_numpy(y).float()
                dataset = AggDataset(X_tensor, Z_tensor, y_tensor)
            else:
                # image classification, X is an ndarray of PIL images
                # if self.privacy is None:
                #     transform_train = transforms.Compose([
                #         transforms.ToPILImage(),
                #         transforms.RandomCrop(self.image_size, padding=self.image_size // 8),
                #         transforms.RandomHorizontalFlip(),
                #         transforms.ToTensor(),
                #         transforms.Normalize(mean=[0.5, ], std=[0.5, ])
                #     ])
                # else:
                #     transform_train = transforms.Compose([
                #         transforms.ToPILImage(),
                #         transforms.ToTensor(),
                #         transforms.Normalize(mean=[0.5, ], std=[0.5, ])
                #     ])
                transform_train = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, ], std=[0.5, ])
                ])
                Z_tensor = torch.from_numpy(Z).float()
                dataset = AggDataset(X_active, Z_tensor, y, transform=transform_train)
        else:
            raise UnsupportedTaskError
        data_loader = DataLoader(dataset, batch_size=self.agg_batch_size, shuffle=True,
                                 drop_last=True, num_workers=self.num_workers,
                                 multiprocessing_context='fork' if self.num_workers > 0 else None)

        model = self.agg_model
        if self.cuda_parallel:
            model = nn.DataParallel(model)

        if self.task == "binary_classification":
            loss_fn = nn.BCELoss()
        elif self.task == "regression":
            loss_fn = nn.MSELoss()
        elif self.task == "multi_classification":
            # image classification by default
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise UnsupportedTaskError

        total_loss = 0.0
        num_mini_batches = 0
        for i in range(self.num_agg_rounds):
            for j, (idx, X_i, Z_i, y_i) in enumerate(data_loader, 0):
                X_i = X_i.to(self.device)
                Z_i = Z_i.to(self.device)
                y_i = y_i.to(self.device)

                optimizer.zero_grad()
                model.Z = None

                model.Z = Z_i
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
                # optimizer.step()
                num_mini_batches += 1

            # if self.optimizer == 'sgd':
            #     scheduler.step(1)

            if self.privacy is None:
                print("[Aggregating] Epoch {}: training loss {}"
                      .format(ep * self.num_agg_rounds + i + 1, total_loss / num_mini_batches))
            elif self.privacy == 'MA':
                epsilon, alpha = optimizer.privacy_engine.get_privacy_spent(self.delta)
                print("[Aggregating] Epoch {}: training loss {}, eps {}, delta {}, alpha {}"
                      .format(ep * self.num_agg_rounds + i + 1, total_loss / num_mini_batches, epsilon, self.delta,
                              alpha))
                self.writer.add_scalar('Aggregation privacy accumulation',
                                       epsilon, ep * self.num_agg_rounds + i + 1)
            else:
                raise UnsupportedPrivacyMechanismError
            # print("[Aggregating] Epoch {}: training loss {}"
            #       .format(ep * self.num_agg_rounds + i + 1, total_loss / num_mini_batches))

            if self.writer:
                self.writer.add_scalar('Aggregation training loss',
                                       total_loss / num_mini_batches, ep * self.num_agg_rounds + i + 1)
            total_loss = 0.0
            num_mini_batches = 0

            assert is_perturbation(Z_copy, Z)

    def train(self, Xs, y, Xs_test=None, y_test=None, use_cache=True):
        if use_cache and not os.path.isdir('cache'):
            os.mkdir('cache')
        num_instances = Xs[0].shape[0]
        if self.privacy is not None:
            assert self.agg_batch_size == self.local_batch_size
            if self.inter_party_comp_method == 'MA':
                self.local_dp_getter = GradientDPCalculator(
                    num_instances=num_instances,
                    batch_size=self.agg_batch_size * self.batches_per_lot,
                    num_local_epochs=self.num_local_rounds,
                    num_agg_epochs=self.num_epochs,
                    epsilon=self.epsilon,
                    delta=self.delta,
                    num_parties=self.num_parties
                )
                self.agg_dp_getter = self.local_dp_getter
            elif self.inter_party_comp_method == 'simple':
                self.local_dp_getter = GradientDPCalculator(
                    num_instances=num_instances,
                    batch_size=self.local_batch_size * self.batches_per_lot,
                    num_local_epochs=0,
                    num_agg_epochs=self.num_local_rounds,
                    epsilon=self.epsilon / self.num_parties,
                    delta=self.delta / self.num_parties,
                    num_parties=1
                )
                self.agg_dp_getter = GradientDPCalculator(
                    num_instances=num_instances,
                    batch_size=self.agg_batch_size * self.batches_per_lot,
                    num_local_epochs=0,
                    num_agg_epochs=self.num_epochs,
                    epsilon=self.epsilon / self.num_parties,
                    delta=self.delta / self.num_parties,
                    num_parties=1
                )
            else:
                raise UnsupportedPrivacyMechanismError
            print("DP calculation finished, local_sigma={}, agg_sigma={}".format(self.local_dp_getter.sigma,
                                                                                 self.agg_dp_getter.sigma))

        fmt_name = convert_name_to_path(self.name)
        label_path = "cache/{}_labels_dim_{}.npy".format(fmt_name, self.local_output_dim)
        perturb_label_path = "cache/{}_perturb_labels_dim_{}.npy".format(fmt_name, self.local_output_dim)
        pred_label_path = "cache/{}_pred_labels_dim_{}.npy".format(fmt_name, self.local_output_dim)
        if use_cache and os.path.isfile(label_path) \
                and os.path.isfile(perturb_label_path) and os.path.isfile(pred_label_path):
            print("Loading random labels from cache")
            self.local_labels = np.load(label_path)
            perturb_labels = np.load(perturb_label_path)
            pred_labels = np.load(pred_label_path)
            print("Finished loading")
        else:
            print("Initializing and local labels")
            # initialize local models and local labels
            local_labels = []

            for party_id in range(self.num_parties):
                local_labels.append(generate_random_targets(num_instances, self.local_output_dim)[None, :, :])
            self.local_labels = np.concatenate(local_labels, axis=0)
            perturb_labels = self.local_labels.copy()
            pred_labels = np.zeros(self.local_labels.shape)
            if use_cache:
                np.save(label_path, self.local_labels)
            print("Finished initializing")

        # training local models except for the active party
        print("Start training local models")
        if self.train_order is None:
            self.train_order = list(range(self.num_parties))
        for party_id in self.train_order:
            if party_id != self.active_party_id:
                # initialize local model
                num_features = Xs[party_id].shape[1]
                if self.task in ["binary_classification", "regression"]:
                    if self.model_type == 'fc':
                        local_model = FC(num_features, self.local_hidden_layers, output_size=self.local_output_dim,
                                         activation='tanh')
                    elif self.model_type == 'ncf':
                        local_model = NCF(self.ncf_counts[party_id], self.ncf_embed_dims[party_id],
                                          self.local_hidden_layers, output_size=self.local_output_dim)
                    else:
                        assert False
                elif self.task in ["multi_classification"]:
                    if self.model_type != 'fc':
                        assert Xs[0].shape[1] == Xs[0].shape[2]
                        self.image_size = Xs[0].shape[1]
                    if self.model_type == 'resnet18':
                        local_model = ResNet18(image_size=self.image_size, num_classes=self.local_output_dim)
                    elif self.model_type == 'cnn':
                        if self.privacy is None:
                            local_model = CNN(n_channels=self.n_channels, image_size=Xs[party_id].shape[1],
                                              output_dim=self.local_output_dim)
                        else:
                            local_model = BareSmallCNN(n_channels=self.n_channels,
                                                       image_size=Xs[self.active_party_id].shape[1],
                                                       output_dim=self.local_output_dim)
                    elif self.model_type == 'fc':
                        local_model = FC(num_features, self.local_hidden_layers, output_size=self.local_output_dim)
                    else:
                        raise UnsupportedModelError
                else:
                    raise UnsupportedTaskError

                # load or train local model
                local_model_path = "cache/{}_model_{}_dim_{}.pth".format(fmt_name, party_id, self.local_output_dim)
                if use_cache and os.path.isfile(local_model_path):
                    print("Loading local model for party {} from {}".format(party_id, local_model_path))
                    if self.cuda_parallel:
                        local_model = nn.DataParallel(local_model)
                    local_model.load_state_dict(torch.load(local_model_path,
                                                           map_location=lambda storage, location: storage))
                    self.local_models.append(local_model)
                else:
                    local_model = self.train_local_party(0, party_id, Xs[party_id],
                                                         perturb_labels[party_id, :, :], local_model)
                    pred_labels[party_id, :, :] = self.predict_local(Xs[party_id], local_model)

                    if use_cache:
                        # save perturbed labels for each model
                        np.save(pred_label_path, pred_labels)
                        np.save(perturb_label_path, perturb_labels)
                        torch.save(local_model.state_dict(), local_model_path)

                    self.local_models.append(local_model)

        # if self.privacy:
        #     # add noise to pred_labels
        #     print("Clip the l2-norm of pred_labels")
        #     l2_norms = np.linalg.norm(pred_labels, ord=2, axis=(1, 2))
        #     max_norm = self.out_norm
        #     clip_ratio = np.minimum(max_norm / (l2_norms + 1e-7), 1.0)
        #     for i, party_labels in enumerate(pred_labels):
        #         party_labels *= clip_ratio[i]
        #
        #     print("Adding noise to predicted labels")
        #     out_dp_calc = GaussianDPCalculator(steps=self.num_parties - 1,
        #                                        epsilon=self.epsilon,
        #                                        delta=self.delta,
        #                                        clip_norm=max_norm)
        #     if self.out_privacy == 'rdp':
        #         sigma = out_dp_calc.calc_rdp_sigma()
        #         noise = np.random.normal(scale=sigma, size=pred_labels.shape)
        #         pred_labels += noise
        #     elif self.out_privacy == 'simple':
        #         sigma = out_dp_calc.calc_simple_sigma()
        #         noise = np.random.normal(scale=sigma, size=pred_labels.shape)
        #         pred_labels += noise
        #     else:
        #         pass

        # add noise to representations
        print("Adding noise {} to predicted labels".format(self.repr_noise))
        noise = np.random.normal(scale=self.repr_noise, size=pred_labels.shape)
        pred_labels += noise

        # initialize agg model
        if self.task in ["binary_classification", "regression"]:
            num_features = Xs[self.active_party_id].shape[1]
            if self.model_type == 'fc':
                active_model = FC(num_features, self.local_hidden_layers, output_size=self.local_output_dim)
                self.agg_model = AggModel(mid_output_dim=self.local_output_dim,
                                          num_parties=self.num_parties,
                                          agg_hidden_sizes=self.agg_hidden_layers,
                                          active_model=active_model,
                                          output_dim=1,
                                          activation='sigmoid')
            elif self.model_type == 'ncf':
                active_model = NCF(self.ncf_counts[self.active_party_id], self.ncf_embed_dims[self.active_party_id],
                                   self.local_hidden_layers, output_size=self.local_output_dim)
                self.agg_model = AggModel(mid_output_dim=self.local_output_dim,
                                          num_parties=self.num_parties,
                                          agg_hidden_sizes=self.agg_hidden_layers,
                                          active_model=active_model,
                                          output_dim=1,
                                          activation=None)
            else:
                assert False
        elif self.task in ["multi_classification"]:
            if self.model_type == 'resnet18':
                active_model = ResNet18(image_size=self.image_size, num_classes=self.local_output_dim)
            elif self.model_type == 'cnn':
                if self.privacy is None:
                    active_model = CNN(n_channels=self.n_channels, image_size=Xs[self.active_party_id].shape[1],
                                       output_dim=self.local_output_dim)
                else:
                    active_model = BareSmallCNN(n_channels=self.n_channels,
                                                image_size=Xs[self.active_party_id].shape[1],
                                                output_dim=self.local_output_dim)
            elif self.model_type == 'fc':
                num_features = Xs[self.active_party_id].shape[1]
                active_model = FC(num_features, self.local_hidden_layers, output_size=self.local_output_dim)
                self.agg_model = AggModel(mid_output_dim=self.local_output_dim,
                                          num_parties=self.num_parties,
                                          agg_hidden_sizes=self.agg_hidden_layers,
                                          active_model=active_model,
                                          output_dim=self.n_classes,
                                          activation=None)
            else:
                raise UnsupportedModelError
            self.agg_model = AggModel(mid_output_dim=self.local_output_dim,
                                      num_parties=self.num_parties,
                                      agg_hidden_sizes=self.agg_hidden_layers,
                                      active_model=active_model,
                                      output_dim=self.n_classes,
                                      activation=None)
        else:
            raise UnsupportedTaskError

        # train agg model
        print("Start training aggregation model")
        # optimizer should be defined here for once due to torchdp
        self.agg_model = self.agg_model.to(self.device)
        if self.optimizer == 'adam':
            model_optimizer = optim.Adam(self.agg_model.parameters(), lr=self.agg_lr,
                                         weight_decay=self.agg_weight_decay)
            # scheduler = None
        elif self.optimizer == 'sgd':
            model_optimizer = optim.SGD(self.agg_model.parameters(), lr=self.agg_lr,
                                        momentum=self.momentum, weight_decay=self.agg_weight_decay)
            # scheduler = optim.lr_scheduler.StepLR(model_optimizer, step_size=100, gamma=0.1)
        elif self.optimizer == 'lamb':
            model_optimizer = adv_optim.Lamb(self.agg_model.parameters(), lr=self.agg_lr,
                                             weight_decay=self.agg_weight_decay)
        else:
            raise UnsupportedOptimizerError

        # if privacy is required, create dp calculator
        if self.privacy is None:
            pass
        elif self.privacy == "MA":
            privacy_engine = PrivacyEngine(
                module=self.agg_model,
                batch_size=self.agg_dp_getter.batch_size,
                sample_size=self.agg_dp_getter.num_instances,
                alphas=[self.agg_dp_getter.alpha],
                noise_multiplier=self.agg_dp_getter.sigma,
                max_grad_norm=self.grad_norm_C
            )
            privacy_engine.attach(model_optimizer)
        else:
            raise UnsupportedPrivacyMechanismError

        best_test_acc = 0.0
        best_test_f1 = 0.0
        best_test_auc = 0.0
        best_test_rmse = np.inf
        for ep in range(self.num_epochs):
            self.chmod('train')
            start_epoch = datetime.now()
            passive_party_range = list(range(self.num_parties))
            passive_party_range.remove(self.active_party_id)
            Z = pred_labels[passive_party_range, :, :].transpose((1, 0, 2)).reshape(num_instances, -1)

            self.train_aggregation(ep, Z, Xs[self.active_party_id], y, model_optimizer)

            if Xs_test is not None and y_test is not None and (ep + 1) % self.test_freq == 0:
                self.chmod('eval')
                with torch.no_grad():
                    if self.task == 'binary_classification':
                        y_score_train = self.predict_agg(Xs)
                        y_score_test = self.predict_agg(Xs_test)
                        y_pred_train = np.where(y_score_train > 0.5, 1, 0)
                        y_pred_test = np.where(y_score_test > 0.5, 1, 0)
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
                        print(
                            "[Final] Epoch {}: train accuracy {}, test accuracy {}".format(ep + 1, train_acc, test_acc))
                        print("[Final] Epoch {}: train f1 {}, test f1 {}".format(ep + 1, train_f1, test_f1))
                        print("[Final] Epoch {}: train auc {}, test f1 {}".format(ep + 1, train_auc, test_auc))
                        print("[Final] Epoch {}: best test acc {}, best test f1 {}, best test auc {}".format(ep + 1,
                                                                                                             best_test_acc,
                                                                                                             best_test_f1,
                                                                                                             best_test_auc))
                        self.writer.add_scalars("Aggregation model/train & test accuracy",
                                                {'train': train_acc,
                                                 'test': test_acc}, ep + 1)
                        self.writer.add_scalars("Aggregation model/train & test F1-score",
                                                {'train': train_f1,
                                                 'test': test_f1}, ep + 1)
                        self.writer.add_scalars("Aggregation model/train & test AUC score",
                                                {'train': train_auc,
                                                 'test': test_auc}, ep + 1)
                    elif self.task == 'regression':
                        y_score_train = self.predict_agg(Xs)
                        y_score_test = self.predict_agg(Xs_test)
                        train_rmse = np.sqrt(mean_squared_error(y, y_score_train))
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_score_test))
                        if test_rmse < best_test_rmse:
                            best_test_rmse = test_rmse
                        print("[Final] Epoch {}: train rmse {}, test rmse {}".format(ep + 1, train_rmse, test_rmse))
                        print("[Final] Epoch {}: best test rmse {}".format(ep + 1, best_test_rmse))
                        self.writer.add_scalars("Aggregation model/train & test rmse",
                                                {'train': train_rmse,
                                                 'test': test_rmse}, ep + 1)
                    elif self.task == 'multi_classification':
                        if self.model_type == 'fc':
                            train_acc = self.eval_image(Xs, y, ['accuracy'], has_transform=False)
                            test_acc = self.eval_image(Xs_test, y_test, ['accuracy'], has_transform=False)
                        else:
                            train_acc = self.eval_image(Xs, y, ['accuracy'], has_transform=True)
                            test_acc = self.eval_image(Xs_test, y_test, ['accuracy'], has_transform=True)
                        if test_acc > best_test_acc:
                            best_test_acc = test_acc
                        print(
                            "[Final] Epoch {}: train accuracy {}, test accuracy {}".format(ep + 1, train_acc, test_acc))
                        print("[Final] Epoch {}: best test acc {}".format(ep + 1, best_test_acc))
                        self.writer.add_scalars("Aggregation model/train & test accuracy",
                                                {'train': train_acc,
                                                 'test': test_acc}, ep + 1)
                    else:
                        raise UnsupportedTaskError
            epoch_duration_sec = (datetime.now() - start_epoch).seconds
            print("Epoch {} duration {} sec".format(ep + 1, epoch_duration_sec), flush=True)
        # save aggregate model
        agg_model_path = "cache/{}_agg_model_dim_{}.pth".format(self.full_name, self.local_output_dim)
        torch.save(self.agg_model.state_dict(), agg_model_path)
        return best_test_acc, best_test_f1, best_test_rmse, best_test_auc

    def predict_local(self, X, model, batch_size=256):
        if self.task in ["binary_classification", "regression"]:
            X_tensor = torch.from_numpy(X).float()
            dataset = TensorDataset(X_tensor)
        elif self.task == "multi_classification":
            if self.model_type == 'fc':
                X_tensor = torch.from_numpy(X).float()
                dataset = TensorDataset(X_tensor)
            else:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, ], std=[0.5, ])
                ])
                dataset = ImageDataset([X], transform=transform_test)
        else:
            raise UnsupportedTaskError
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)

        Z_list = []
        with torch.no_grad():
            model = model.to(self.device)
            model.eval()
            for (X_i,) in dataloader:
                X_i = X_i.to(self.device)
                Z_i = model(X_i)
                Z_list.append(Z_i)
        Z = torch.cat(Z_list).detach().cpu().numpy()

        assert Z.shape[0] == X.shape[0]
        return np.float32(Z)

    def predict_agg(self, Xs):
        local_labels_pred = []
        for party_id in range(self.num_parties):
            if party_id != self.active_party_id:
                X_tensor = torch.from_numpy(Xs[party_id]).float().to(self.device)
                local_party_id = party_id if party_id < self.active_party_id else party_id - 1
                Z_pred_i = self.local_models[local_party_id].to(self.device)(X_tensor)
                local_labels_pred.append(Z_pred_i.detach().cpu().numpy()[None, :, :])
        local_labels_pred = np.concatenate(local_labels_pred, axis=0)
        num_instances = local_labels_pred.shape[1]

        Z = local_labels_pred.transpose((1, 0, 2)).reshape(num_instances, -1)
        Z_tensor = torch.from_numpy(Z).float().to(self.device)
        X_tensor = torch.from_numpy(Xs[self.active_party_id]).float().to(self.device)
        model = self.agg_model.to(self.device)
        model.Z = Z_tensor
        y_score = model(X_tensor)
        y_score = y_score.detach().cpu().numpy()
        return y_score

    def eval_image(self, Xs, y, metrics: list, has_transform=True):
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
        total = 0
        for data in dataloader:
            Xs_i = data[:-1]
            y_i = data[-1].detach().cpu().numpy()

            local_labels_pred = []
            for party_id in range(self.num_parties):
                if party_id != self.active_party_id:
                    X_tensor = Xs_i[party_id].to(self.device)
                    local_party_id = party_id if party_id < self.active_party_id else party_id - 1
                    Z_pred_i = self.local_models[local_party_id].to(self.device)(X_tensor)
                    local_labels_pred.append(Z_pred_i.detach().cpu().numpy()[None, :, :])
            local_labels_pred = np.concatenate(local_labels_pred, axis=0)
            num_instances = local_labels_pred.shape[1]

            Z = local_labels_pred.transpose((1, 0, 2)).reshape(num_instances, -1)
            Z_tensor = torch.from_numpy(Z).float().to(self.device)
            X_tensor = Xs_i[self.active_party_id].to(self.device)
            model = self.agg_model.to(self.device)
            model.Z = Z_tensor
            y_score = model(X_tensor)
            y_score = y_score.detach().cpu().numpy()

            if self.task == 'binary_classification':
                y_pred = np.where(y_score > 0.5, 1, 0)
            elif self.task == 'multi_classification':
                y_pred = np.argmax(y_score, axis=1)
            else:
                raise UnsupportedTaskError
            correct += (y_pred == y_i).sum()
            total += num_instances

            for j in range(len(metrics)):
                results[j] += metric_func_list[j](y_i, y_score)
            ep_cnt += 1

        results /= ep_cnt
        if acc_metric_id >= 0:
            results[acc_metric_id] = correct / total

        if len(results) == 1:
            return results[0]
        else:
            return results

    @property
    def params(self):
        attrs = vars(self)
        output = '\n'.join("%s=%s" % item for item in attrs.items())
        return output

    def save_local_labels(self, path):
        np.save(path, self.local_labels)

    @staticmethod
    def load_local_labels(path):
        return np.load(path)

    def chmod(self, mode):
        if mode == 'eval':
            for model in self.local_models:
                model.eval()
            self.agg_model.eval()
        elif mode == 'train':
            for model in self.local_models:
                model.train()
            self.agg_model.train()
        else:
            raise UnsupportedModeError
