import torch
from torch import nn, autograd
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

from models import Nets
from models.Nets import MLP, CNNCifar


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[int(self.idxs[item])]
        return image, label


class LocalUpdate(object):
    loss_func: CrossEntropyLoss

    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.local_model = CNNCifar(args=args).to(args.device)


    def train(self, net):
        net.train()
        # CUDA
        torch.backends.cudnn.benchmark = True

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images = images.to(self.args.device, non_blocking=True)
                labels = labels.to(self.args.device, non_blocking=True)

                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        gradients = torch.cat([torch.reshape(param.grad, (-1,)) for param in net.parameters()]).clone().detach()

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), gradients

