from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np


class_sep = 1.0
size = '20m'
X, y = make_classification(20000000, 20, n_informative=2, n_redundant=16, class_sep=class_sep, random_state=0)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=0)
np.save("data/syn/syn_{}_{}.train".format(size, class_sep), np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1))
np.save("data/syn/syn_{}_{}.val".format(size, class_sep), np.concatenate([X_val, y_val.reshape(-1, 1)], axis=1))
np.save("data/syn/syn_{}_{}.test".format(size, class_sep), np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1))



# class_sep = 0.5
# X, y = make_classification(10000000, 20, class_sep=class_sep, random_state=0)
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=0)
# np.save("data/syn/syn_10m_{}.train".format(class_sep), np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1))
# np.save("data/syn/syn_10m_{}.val".format(class_sep), np.concatenate([X_val, y_val.reshape(-1, 1)], axis=1))
# np.save("data/syn/syn_10m_{}.test".format(class_sep), np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1))


# class_sep = 0.5
# X, y = make_classification(40000000, 20, class_sep=class_sep, random_state=0)
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=0)
# np.save("data/syn/syn_40m_{}.train".format(class_sep), np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1))
# np.save("data/syn/syn_40m_{}.val".format(class_sep), np.concatenate([X_val, y_val.reshape(-1, 1)], axis=1))
# np.save("data/syn/syn_40m_{}.test".format(class_sep), np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1))

