#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchdp import PrivacyEngine
from torchvision import transforms
from torchvision.datasets import FakeData


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 3)
        self.conv2 = nn.Conv1d(16, 32, 3, 1)
        self.convf = nn.Conv1d(32, 32, 1, 1)
        for p in self.convf.parameters():
            p.requires_grad = False
        self.fc1 = nn.Linear(23, 17)
        self.fc2 = nn.Linear(32 * 17, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 10, 10]
        x = F.max_pool2d(x, 2, 2)  # -> [B, 16, 5, 5]
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])  # -> [B, 16, 25]
        x = F.relu(self.conv2(x))  # -> [B, 32, 23]
        x = self.convf(x)  # -> [B, 32, 23]
        x = self.fc1(x)  # -> [B, 32, 17]
        x = x.view(-1, x.shape[-2] * x.shape[-1])  # -> [B, 32 * 17]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"


class GradientAccumulation_test(unittest.TestCase):
    def setUp(self):
        self.DATA_SIZE = 64
        self.BATCH_SIZE = 16
        self.LR = 0  # we want to call optimizer.step() without modifying the model
        self.ALPHAS = [1 + x / 10.0 for x in range(1, 100, 10)]
        self.criterion = nn.CrossEntropyLoss()

        self.setUp_data()
        self.setUp_model_and_optimizer()

    def setUp_data(self):
        self.ds = FakeData(
            size=self.DATA_SIZE,
            image_size=(1, 35, 35),
            num_classes=10,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        self.dl = DataLoader(self.ds, batch_size=self.BATCH_SIZE)

    def setUp_model_and_optimizer(self):
        self.model = SampleConvNet()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.LR, momentum=0
        )

        self.optimizer.zero_grad()

        # accumulate .grad over the entire data
        for x, y in self.dl:
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()

        self.effective_batch_grad = torch.cat(
            [p.grad.reshape(-1) for p in self.model.parameters() if p.requires_grad]
        ) * (self.BATCH_SIZE / self.DATA_SIZE)

        self.optimizer.zero_grad()

    def setUp_privacy_engine(self, batch_size):
        self.privacy_engine = PrivacyEngine(
            self.model,
            batch_size=batch_size,
            sample_size=self.DATA_SIZE,
            alphas=self.ALPHAS,
            noise_multiplier=0,
            max_grad_norm=999,
        )
        self.privacy_engine.attach(self.optimizer)

    def test_grad_sample_accumulation(self):
        """
        Calling loss.backward() multiple times should sum up the gradients in .grad
        and accumulate all the individual gradients in .grad-sample
        """
        self.setUp_privacy_engine(self.DATA_SIZE)

        for x, y in self.dl:
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()  # should accumulate grads in .grad and .grad_sample

        # the accumulated per-sample gradients
        per_sample_grads = torch.cat(
            [
                p.grad_sample.reshape(self.DATA_SIZE, -1)
                for p in self.model.parameters()
                if p.requires_grad
            ],
            dim=-1,
        )
        # average up all the per-sample gradients
        accumulated_grad = torch.mean(per_sample_grads, dim=0)

        # the full data gradient accumulated in .grad
        grad = torch.cat(
            [p.grad.reshape(-1) for p in self.model.parameters() if p.requires_grad]
        ) * (self.BATCH_SIZE / self.DATA_SIZE)

        self.optimizer.step()

        # the accumulated gradients in .grad without any hooks
        orig_grad = self.effective_batch_grad

        self.assertTrue(
            torch.allclose(accumulated_grad, orig_grad, atol=10e-5, rtol=10e-3)
        )
        self.assertTrue(torch.allclose(grad, orig_grad, atol=10e-5, rtol=10e-3))

    def test_clipper_accumulation(self):
        """
        Calling optimizer.virtual_step() should accumulate clipped gradients to form
        one large batch.
        """
        self.setUp_privacy_engine(self.DATA_SIZE)

        for x, y in self.dl:
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.virtual_step()

        # the full data gradient accumulated in .grad
        grad = torch.cat(
            [p.grad.reshape(-1) for p in self.model.parameters() if p.requires_grad]
        ) * (self.BATCH_SIZE / self.DATA_SIZE)

        self.optimizer.step()

        # .grad should contain the average gradient over the entire data
        accumulated_grad = torch.cat(
            [p.grad.reshape(-1) for p in self.model.parameters() if p.requires_grad]
        )

        # the accumulated gradients in .grad without any hooks
        orig_grad = self.effective_batch_grad

        self.assertTrue(
            torch.allclose(accumulated_grad, orig_grad, atol=10e-5, rtol=10e-3)
        )
        self.assertTrue(torch.allclose(grad, orig_grad, atol=10e-5, rtol=10e-3))

    def test_mixed_accumulation(self):
        """
        Calling loss.backward() multiple times aggregates all per-sample gradients in
        .grad-sample. Then, calling optimizer.virtual_step() should clip all gradients
        and aggregate them into one large batch.
        """
        self.setUp_privacy_engine(self.DATA_SIZE)

        for idx, (x, y) in enumerate(self.dl):
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()

            # accumulate per-sample grads for two mini batches
            # and then aggregate them in the clipper
            if (idx + 1) % 2 == 0:
                self.optimizer.virtual_step()

        # the full data gradient accumulated in .grad
        grad = torch.cat(
            [p.grad.reshape(-1) for p in self.model.parameters() if p.requires_grad]
        ) * (self.BATCH_SIZE / self.DATA_SIZE)

        self.optimizer.step()

        # .grad should contain the average gradient over the entire data
        accumulated_grad = torch.cat(
            [p.grad.reshape(-1) for p in self.model.parameters() if p.requires_grad]
        )

        # the accumulated gradients in .grad without any hooks
        orig_grad = self.effective_batch_grad

        self.assertTrue(
            torch.allclose(accumulated_grad, orig_grad, atol=10e-5, rtol=10e-3)
        )
        self.assertTrue(torch.allclose(grad, orig_grad, atol=10e-5, rtol=10e-3))

    def test_grad_sample_erased(self):
        """
        Calling optimizer.step() should erase any accumulated per-sample gradients.
        """

        self.setUp_privacy_engine(2 * self.BATCH_SIZE)

        for idx, (x, y) in enumerate(self.dl):
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()

            # accumulate per-sample gradients for two mini-batches to form an
            # effective batch of size `2*BATCH_SIZE`. Once an effective batch
            # has been accumulated, we call `optimizer.step()` to clip and
            # average the per-sample gradients. This should erase the
            # `grad_sample` fields for each parameter
            if (idx + 1) % 2 == 0:

                # take a step. The clipper will compute the mean gradient
                # for the entire effective batch and populate each parameter's
                # `.grad` field.
                self.optimizer.step()

                for param_name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.assertFalse(
                            hasattr(param, "grad_sample"),
                            f"Per-sample gradients haven't been erased "
                            f"for {param_name}",
                        )
            else:
                for param_name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.assertTrue(
                            hasattr(param, "grad_sample"),
                            f"Per-sample gradients aren't accumulated "
                            f"for {param_name}",
                        )

    def test_summed_grad_erased(self):
        """
        Calling optimizer.step() should erase any accumulated clipped gradients.
        """

        self.setUp_privacy_engine(2 * self.BATCH_SIZE)

        for idx, (x, y) in enumerate(self.dl):
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()

            # perform a virtual step for each mini-batch
            # this will accumulate clipped gradients in each parameter's
            # `summed_grads` field.
            self.optimizer.virtual_step()

            # accumulate gradients for two mini-batches to form an
            # effective batch of size `2*BATCH_SIZE`. Once an effective batch
            # has been accumulated, we call `optimizer.step()` to compute the
            # average gradient for the entire batch. This should erase the
            # `summed_grads` fields for each parameter.
            if (idx + 1) % 2 == 0:

                # take a step. The clipper will compute the mean gradient
                # for the entire effective batch and populate each parameter's
                # `.grad` field.
                self.optimizer.step()

                for param_name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.assertFalse(
                            hasattr(param, "summed_grad"),
                            f"Accumulated clipped gradients haven't been erased "
                            f"¨for {param_name}",
                        )
            else:
                for param_name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.assertTrue(
                            hasattr(param, "summed_grad"),
                            f"Clipped gradients aren't accumulated "
                            f"for {param_name}",
                        )

    def test_throws_wrong_batch_size(self):
        """
        If we accumulate the wrong number of gradients and feed this batch to
        the privacy engine, we expect a failure.
        """
        self.setUp_privacy_engine(2 * self.BATCH_SIZE)

        x, y = next(iter(self.dl))
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.virtual_step()

        # consuming a batch that is smaller than expected should work
        with self.assertWarns(Warning):
            self.optimizer.step()
            self.optimizer.zero_grad()

        for _ in range(3):
            x, y = next(iter(self.dl))
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.virtual_step()

        # consuming a larger batch than expected should fail
        with self.assertRaises(ValueError):
            self.optimizer.step()
