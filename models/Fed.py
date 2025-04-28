import copy
import torch
from torch import nn

import copy
import torch


def FedAvg_0(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        weighted_sum = torch.zeros_like(w_avg[k])
        for i in range(len(w)):
            weighted_sum += w[i][k].to(weighted_sum.dtype)
        w_avg[k] = weighted_sum / len(w)
    return w_avg