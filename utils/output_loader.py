import re
import numpy as np

class XGBoostOutputLoader:
    def __init__(self, path, metric):
        self.path = path
        self.metric = metric
        self.num_rounds = 0

    def extract_data(self):
        """
        Extract data from output file
        :return: [(train_list_fold_i, ...), (valid_list_fold_i, ...)]
        """
        train_score_list = []
        valid_score_list = []
        self.num_rounds = 0

        with open(self.path, 'r') as file:
            state = 0
            init = True
            train_score_fold_i = []
            valid_score_fold_i = []
            for line in file:
                if "Cross Validation Fold" in line:
                    if not init:
                        train_score_list.append(train_score_fold_i)
                        valid_score_list.append(valid_score_fold_i)
                    self.num_rounds = 0
                    train_score_fold_i = []
                    valid_score_fold_i = []
                    init = False
                    continue
                elif "INFO" in line:
                    state = 1
                    continue
                elif state == 1:
                    self.num_rounds += 1
                    scores = []
                    for s in re.split(":|\t|\n", line):
                        try:
                            scores.append(float(s))
                        except ValueError:
                            pass
                    if self.metric == 'rmse':
                        train_score_fold_i.append(scores[0])
                        valid_score_fold_i.append(scores[1])
                    elif self.metric == 'error':
                        train_score_fold_i.append(1 - scores[0])
                        valid_score_fold_i.append(1 - scores[1])
                    else:
                        assert False
                    state = 0
                else:
                    pass
            train_score_list.append(train_score_fold_i)
            valid_score_list.append(valid_score_fold_i)

        return train_score_list, valid_score_list

    def comm_size_list(self, num_instances, num_features, num_parties, num_bins=255):
        comm_size_per_round = 2 * num_instances * np.dtype(np.float32).itemsize * num_parties + \
            num_features * num_bins * np.dtype(np.float32).itemsize
        return np.arange(1, self.num_rounds + 1) * 1e-6 * comm_size_per_round