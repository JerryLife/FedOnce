import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.exceptions import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_size=32):
        super(ResNet, self).__init__()
        self.image_size = image_size
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if out.size(2) >= self.image_size // 8:
            out = F.avg_pool2d(out, self.image_size // 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class FC(nn.Module):
    def __init__(self, input_size, hidden_sizes: list, output_size=1, activation=None):
        super(FC, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.fc_layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        if len(hidden_sizes) != 0:
            for i in range(len(hidden_sizes) - 1):
                self.fc_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, X):
        if len(list(self.fc_layers)) == 0:
            return X

        out = F.relu(self.fc_layers[0](X))
        for fc in self.fc_layers[1:-1]:
            out = F.relu(fc(out))
        if self.activation == 'sigmoid':
            out = torch.sigmoid(self.fc_layers[-1](out))
        elif self.activation == 'tanh':
            out = torch.tanh(self.fc_layers[-1](out))
        elif self.activation is None:
            out = self.fc_layers[-1](out)
        else:
            assert False
        return out


class CNN(nn.Module):
    def __init__(self, n_channels, image_size=28, kernel_size=3, stride=1, output_dim=10, dropout=True):
        super().__init__()

        self.has_dropout = dropout
        self.n_channels = n_channels
        self.image_size = image_size
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        maxpool_kernel_size = 2
        self.max_pool = nn.MaxPool2d(maxpool_kernel_size)

        if self.has_dropout:
            self.conv_layers = nn.Sequential(
                self.conv1,
                nn.ReLU(),
                self.conv2,
                nn.ReLU(),
                self.max_pool,
                self.dropout1,
                nn.Flatten(1)
            )
        else:
            self.conv_layers = nn.Sequential(
                self.conv1,
                nn.ReLU(),
                self.conv2,
                nn.ReLU(),
                self.max_pool,
                nn.Flatten(1)
            )

        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc1(x)
        x = F.relu(x)
        if self.has_dropout:
            x = self.dropout2(x)
        output = self.fc2(x)
        return output

    @property
    def fc_input_size(self):
        x = torch.randn((1, self.n_channels, self.image_size, self.image_size))
        out = self.conv_layers(x)
        return out.shape[1]


class SmallCNN(nn.Module):
    def __init__(self, image_size, n_channels=1, output_dim=10):
        super().__init__()
        self.image_size = image_size
        self.n_channels = n_channels

        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        x = torch.randn((1, self.n_channels, self.image_size, self.image_size))
        out = F.max_pool2d(self.conv2(F.max_pool2d(self.conv1(x), 2, 1)), 2, 1)
        self.fc_input_size = out.shape[1] * out.shape[2] * out.shape[3]
        self.fc1 = nn.Linear(self.fc_input_size, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))   # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)   # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))   # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)   # -> [B, 32, 4, 4]
        x = x.view(-1, self.fc_input_size)  # -> [B, 512]
        x = F.relu(self.fc1(x))     # -> [B, 32]
        x = self.fc2(x)             # -> [B, 10]
        return x

class BareSmallCNN(nn.Module):
    def __init__(self, image_size, n_channels=1, output_dim=10):
        super().__init__()
        self.image_size = image_size
        self.n_channels = n_channels

        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        x = torch.randn((1, self.n_channels, self.image_size, self.image_size))
        out = F.max_pool2d(self.conv2(F.max_pool2d(self.conv1(x), 2, 1)), 2, 1)
        self.fc_input_size = out.shape[1] * out.shape[2] * out.shape[3]
        self.fc = nn.Linear(self.fc_input_size, output_dim)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))   # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)   # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))   # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)   # -> [B, 32, 4, 4]
        x = x.view(-1, self.fc_input_size)  # -> [B, 512]
        x = self.fc(x)
        return x



class BareCNN(nn.Module):
    def __init__(self, n_channels, image_size=28, kernel_size=3, stride=1):
        super().__init__()

        self.n_channels = n_channels
        self.image_size = image_size
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        maxpool_kernel_size = 2
        self.max_pool = nn.MaxPool2d(maxpool_kernel_size)

        self.conv_layers = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.max_pool,
            self.dropout1,
            nn.Flatten(1)
        )

    def forward(self, x):
        return self.conv_layers(x)

    @property
    def fc_input_size(self):
        x = torch.randn((1, self.n_channels, self.image_size, self.image_size))
        out = self.conv_layers(x)
        return out.shape[1]


class AggModel(nn.Module):
    def __init__(self, mid_output_dim, num_parties: int,
                 agg_hidden_sizes, active_model: nn.Module, output_dim=1, activation=None):
        super().__init__()
        self.activation = activation
        self.mid_output_size = mid_output_dim
        self.num_parties = num_parties

        self.active_model = active_model

        # FC layers for aggregation
        self.agg_fc_layers = nn.ModuleList([nn.Linear(mid_output_dim * num_parties, agg_hidden_sizes[0])])
        if len(agg_hidden_sizes) != 0:
            for i in range(len(agg_hidden_sizes) - 1):
                self.agg_fc_layers.append(nn.Linear(agg_hidden_sizes[i], agg_hidden_sizes[i + 1]))
        self.agg_fc_layers.append(nn.Linear(agg_hidden_sizes[-1], output_dim))
        self.Z = None

    def forward(self, X):
        """
        forward function for Z, X. Z has to be assigned before forward() is called.
        Not using two inputs is because pytorch_dp only supports one-input module.
        :param X:
        :return:
        """
        # assert list(Z.size())[0] == list(X.size())[0]
        Z = self.Z
        assert Z is not None, "Z has not been defined"
        assert list(Z.size())[1] == self.mid_output_size * (self.num_parties - 1), f"{list(Z.size())[1]=},{self.mid_output_size * (self.num_parties - 1)=}"

        # FC layers for active party before aggregation
        active_out = self.active_model(X)

        # FC layers for aggregation
        agg_input = torch.cat([active_out, Z], dim=1)
        out = F.relu(self.agg_fc_layers[0](agg_input))
        for fc in self.agg_fc_layers[1:-1]:
            out = F.relu(fc(out))
        out = self.agg_fc_layers[-1](out)
        if self.activation is None:
            final_out = out
        elif self.activation == 'sigmoid':
            final_out = torch.sigmoid(out)
        elif self.activation == 'softmax':
            final_out = torch.softmax(out, dim=1)
        else:
            raise UnsupportedActivationFuncError
        return final_out


class NCF(nn.Module):
    def __init__(self, counts: list, emb_dims: list, hidden_sizes: list, output_size=1, activation=None):
        super(NCF, self).__init__()
        self.activation = activation

        self.embedding_layers = nn.ModuleList([])
        for i in range(len(counts)):
            self.embedding_layers.append(nn.Embedding(counts[i], emb_dims[i]))

        assert len(hidden_sizes) > 0
        self.fc_layers = nn.ModuleList([nn.Linear(sum(emb_dims), hidden_sizes[0])])
        if len(hidden_sizes) != 0:
            for i in range(len(hidden_sizes) - 1):
                self.fc_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        x = x.long()
        embedding_list = []
        for i, embedding_layer in enumerate(self.embedding_layers):
            embedding = embedding_layer(x[:, i])
            embedding_list.append(embedding)
        embed = torch.cat(embedding_list, dim=1)

        out = F.relu(self.fc_layers[0](embed))
        for fc in self.fc_layers[1:-1]:
            out = F.relu(fc(out))
        if self.activation == 'sigmoid':
            out = torch.sigmoid(self.fc_layers[-1](out))
        elif self.activation == 'tanh':
            out = torch.tanh(self.fc_layers[-1](out))
        elif self.activation is None:
            out = self.fc_layers[-1](out)
        else:
            assert False

        return out