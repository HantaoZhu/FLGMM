import numpy as np
import torch
from torchvision import datasets, transforms
from utils.language_utils import ALL_LETTERS

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def fmnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def add_gaussian_noise(dataset, dict_users, client):
    """
    Add Gaussian noise to the dataset of a specific client.

    :param dataset: The entire dataset.
    :param dict_users: A dictionary where keys are client IDs and values are lists of data indices.
    :param client: The client ID to which noise should be added.
    :return: The modified dataset with noise added to the specified client's data.
    """
    new_train_dataset = []
    indices = list(dict_users[client])

    for idx in range(len(dataset)):
        feature, label = dataset[idx]

        # Check if the current index belongs to the specified client
        if idx in indices:
            noise = torch.tensor(np.random.normal(0, 5, feature.shape))
            noise = noise.to(torch.float32)
            new_data = feature + noise
            clip_data = torch.clamp(new_data, -1, 1)
            new_train_dataset.append((clip_data, label))
        else:
            # If the index does not belong to the specified client, keep the data unchanged
            new_train_dataset.append((feature, label))

    return new_train_dataset

def add_noise_to_client_2(dataset, dict_users, client):
    """
    Label flip to a selected client's data
    :param dict_users: dict of image index for each client
    :param client: the client ID to add noise
    """
    indices = list(dict_users[client])
    for idx in indices:
        # Change the label to 9 - k
        current_label = dataset.targets[idx]
        new_label = 9 - current_label
        dataset.targets[idx] = new_label

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    # d = mnist_noniid(dataset_train, num)
