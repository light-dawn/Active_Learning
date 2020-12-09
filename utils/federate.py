import numpy as np
from copy import deepcopy
import torch

# The average aggregation of the federated global model
def average_weights(local_weights, diff_privacy=0.001):
    """
    Federated averaging
    :param w: list of client model parameters
    :param diff_privacy: magnitude of randomization, differential privacy
    :return: updated server model parameters
    """
    w_avg = deepcopy(local_weights[0])
    #print(w_avg)
    for k in w_avg.keys():
        for i in range(1, len(local_weights)):
            w_avg[k] = w_avg[k] + local_weights[i][k]
        w_avg[k] = torch.div(w_avg[k], len(local_weights)) + torch.mul(torch.randn(w_avg[k].shape), diff_privacy)
    return w_avg


def partition(all_indices, num_users, random_seed=2020):
    num_items = int(len(all_indices)/num_users)
    print(f"Items kept by each clients: {num_items}")
    dict_users = {}
    for i in range(num_users):
        np.random.seed(random_seed)
        dict_users[i] = set(np.random.choice(all_indices, num_items, replace=False))
        all_indices = list(set(all_indices) - dict_users[i])
        dict_users[i] = list(dict_users[i])
        # print("Length of dict users: ", len(dict_users[i]))
    return dict_users