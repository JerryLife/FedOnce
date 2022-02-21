import os
import os.path

from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, KMNIST

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import numpy as np

from utils.exceptions import UnsupportedFormatError


def split_train_test(file_name, file_type, root="data/", test_rate=0.1, csv_skiprows=1):
    print("Loading data from file")
    if file_type == 'libsvm':
        x, y = load_svmlight_file(root + file_name)
    elif file_type == 'csv':
        try:
            dataset = np.loadtxt(root + file_name, delimiter=',', skiprows=csv_skiprows)
        except ValueError:
            dataset = np.genfromtxt(root + file_name, delimiter=',', skip_header=csv_skiprows, filling_values=0)
        x = dataset[:, :-1]
        y = dataset[:, -1].reshape(-1)
    elif file_type == 'torch':
        raise UnsupportedFormatError
        # if file_name == 'cifar10':
        #     # trainset will be normalized later in vertical_fl.py after data argumentation
        #     dataset = CIFAR10(root=root, train=True, transform=None, download=True)
        #     x, y = dataset.data, np.array(dataset.targets)
        # elif file_name == 'mnist':
        #     dataset = MNIST(root=root, train=True, transform=None, download=True)
        #     x, y = dataset.data.detach().numpy()[:, :, :, None], np.array(dataset.targets)
        # else:
        #     raise UnsupportedFormatError
    else:
        raise UnsupportedFormatError

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_rate)

    train_save_path = os.path.join(root, file_name + ".train")
    test_save_path = os.path.join(root, file_name + ".test")
    if file_type == 'libsvm':
        dump_svmlight_file(x_train, y_train, train_save_path)
        dump_svmlight_file(x_test, y_test, test_save_path)
    elif file_type == 'csv':
        data_train = np.concatenate([x_train, y_train.reshape(-1, 1)], axis=1)
        np.savetxt(train_save_path, data_train, delimiter=',', header='train')
        data_test = np.concatenate([x_test, y_test.reshape(-1, 1)], axis=1)
        np.savetxt(test_save_path, data_test, delimiter=',', header='test')
    else:
        raise UnsupportedFormatError
