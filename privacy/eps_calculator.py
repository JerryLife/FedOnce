import math

import torch
import torch.nn as nn

from torchdp import privacy_analysis as tf_privacy

from scipy.optimize import bisect
import numpy as np


def gaussian_noise(shape, sigma, clip_C):
    return torch.normal(0, sigma * clip_C, shape)


class GradientDPCalculator:
    def __init__(self, num_instances, batch_size, num_local_epochs, num_agg_epochs, delta, epsilon=None, num_parties=1):
        super().__init__()
        self.num_agg_epochs = num_agg_epochs
        self.num_local_epochs = num_local_epochs
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.delta = delta
        self.num_instances = num_instances
        self.alpha = None

        if self.epsilon is not None and not np.isinf(epsilon):
            xtor = 1e-7  # absolute tolerance
            calc_eps_by_sigma = lambda sigma: self.compute_privacy_loss(sigma, num_parties)[0] - self.epsilon
            self.sigma = bisect(calc_eps_by_sigma, 1e-3, 1e4, xtol=xtor) + xtor  # ensure sigma is large enough
            eps, self.alpha = self.compute_privacy_loss(self.sigma, num_parties)
            assert eps <= self.epsilon, "eps={} self.epsilon={}".format(eps, self.epsilon)
        elif np.isinf(epsilon):
            self.sigma = 0

    def compute_privacy_loss(self, sigma, num_parties=2, delta=None):
        """
        Based on pytorch-dp and tensorflow-privacy
        :param sigma: noise multiplier
        :return: [epsilon, alpha]
        """
        q = self.batch_size / self.num_instances
        local_steps = int(math.ceil(self.num_local_epochs * self.num_instances / self.batch_size))
        agg_steps = int(math.ceil(self.num_agg_epochs * self.num_instances / self.batch_size))
        orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                  list(range(5, 64)) + [128, 256, 512])
        steps = local_steps * (num_parties - 1) + agg_steps

        # calculate renyi divergence
        rdp = tf_privacy.compute_rdp(q, sigma, steps, orders)

        if delta is None:
            delta = self.delta

        return tf_privacy.get_privacy_spent(orders, rdp, delta)


class GaussianDPCalculator:
    def __init__(self, steps, delta, epsilon, clip_norm):
        super().__init__()
        self.clip_norm = clip_norm
        self.steps = steps
        self.epsilon = epsilon
        self.delta = delta

    def calc_rdp_sigma(self):
        xtor = 2e-12  # absolute tolerance
        calc_eps_by_sigma = lambda sigma: self.rdp_comp(sigma,
                                                        self.steps, self.delta)[0] - self.epsilon
        sigma = bisect(calc_eps_by_sigma, 1e-3, 1e4, xtol=xtor) + xtor  # ensure sigma is large enough
        eps, self.alpha = self.rdp_comp(sigma * self.clip_norm, self.steps, self.delta)
        return sigma * self.clip_norm

    def calc_simple_sigma(self):
        xtor = 2e-12  # absolute tolerance
        eps = self.epsilon / self.steps
        delta = self.delta / self.steps
        calc_eps_by_sigma = lambda sigma: self.rdp_comp(sigma,
                                                        1, delta)[0] - eps
        sigma = bisect(calc_eps_by_sigma, 1e-3, 1e4, xtol=xtor) + xtor  # ensure sigma is large enough
        eps, self.alpha = self.rdp_comp(sigma * self.clip_norm, 1, delta)
        return sigma * self.clip_norm

    def rdp_comp(self, sigma, steps, delta):
        """
        Based on pytorch-dp and tensorflow-privacy
        :param sigma: noise multiplier
        :return: [epsilon, alpha]
        """
        q = 0.01
        orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                  list(range(5, 64)) + [128, 256, 512])

        # calculate renyi divergence
        rdp = tf_privacy.compute_rdp(q, sigma, steps, orders)

        return tf_privacy.get_privacy_spent(orders, rdp, delta)



class AdvanceCompositionCalculator:
    def __init__(self, total_delta):
        self.total_delta = total_delta

    def compute_privacy_loss(self, eps, delta, n):
        residual_delta = self.total_delta - delta * n
        assert residual_delta > 0, "Total delta is too small"
        total_eps = np.sqrt(2 * n * np.log(1 / residual_delta)) * eps + \
                    n * eps * (np.exp(eps) - 1)
        return total_eps
