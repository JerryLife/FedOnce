import numpy as np
import scipy.optimize
import torch
import torch.nn as nn

import math

from model.models import AggModel

def generate_random_targets(n: int, z: int, method='sphere'):
    """
    Generate a matrix of random target assignment.
    Each target assignment vector has unit length (hence can be view as random point on hypersphere)
    :param n: the number of samples to generate.
    :param z: the latent space dimensionality
    :return: the sampled representations
    """
    if method == 'sphere':
        # Generate random targets using gaussian distrib.
        samples = np.random.normal(0, 1, (n, z)).astype(np.float32)
        # rescale such that fit on unit sphere.
        radiuses = np.expand_dims(np.sqrt(np.sum(np.square(samples), axis=1)), 1)
        # return rescaled targets
        return samples / radiuses
    elif method == 'uniform':
        return np.random.uniform(0, 1, (n, z)).astype(np.float32)
    else:
        print("Unsupported random method")
        return None


def calc_optimal_target_permutation(feats: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute the new target assignment that minimises the SSE between the mini-batch feature space and the targets.
    :param feats: the learnt features (given some input images)
    :param targets: the currently assigned targets.
    :return: the targets reassigned such that the SSE between features and targets is minimised for the batch.
    """
    # Compute cost matrix
    cost_matrix = np.zeros([feats.shape[0], targets.shape[0]])
    # calc SSE between all features and targets
    for i in range(feats.shape[0]):
         cost_matrix[:, i] = np.sum(np.square(feats-targets[i, :]), axis=1)

    _, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    # Permute the targets based on hungarian algorithm optimisation
    targets[range(feats.shape[0])] = targets[col_ind]
    return targets


def calc_optimal_input_permutation(Z: torch.Tensor, X: torch.Tensor, y: torch.Tensor,
                                   model: AggModel) -> torch.Tensor:
    num_instances = list(Z.size())[0]
    assert num_instances == list(y.size())[0]
    start_loss = nn.BCELoss()(model(Z, X).view(-1), y).item()

    cost_matrix = np.zeros([list(Z.size())[0], list(y.size())[0]])

    for i in range(num_instances):
        X_i = X[None, i].expand(num_instances, -1)
        y_pred = model(Z, X_i)
        y_i = y[i].expand_as(y_pred)
        loss = nn.BCELoss(reduction='none')(y_pred, y_i).view(-1)
        cost_matrix[:, i] = loss.cpu().detach().numpy()

    row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost_matrix)

    Z[col_idx] = Z[row_idx]
    end_loss = nn.BCELoss()(model(Z, X).view(-1), y).item()
    assert start_loss >= end_loss
    return Z


def is_perturbation(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False

    # create hash table for a
    hash_table_a = {}
    for i, a_i in enumerate(a):
        hash_i = hash(a_i.tostring())
        hash_table_a[hash_i] = i

    # check each element in b
    for i, b_i in enumerate(b):
        hash_i = hash(b_i.tostring())
        if hash_i not in hash_table_a:
            return False
    return True


def get_closest_factor(n: int):
    """
    find the two closest integers a & b which, when multiplied, equal to n
    :param n: integer
    :return: a, b
    """
    a = math.ceil(math.sqrt(n))
    while True:
        if n % a == 0:
            b = n // a
            return a, b
        a -= 1


def convert_name_to_path(name: str):
    return name.replace("/", "_")


class Log:
    def __init__(self, file_name):
        self.file_name = file_name
        self.handle = open(file_name, 'w')

    def write(self, str):
        self.handle.write(str)
        if str[-1] != '\n':
            self.handle.write("\n")

