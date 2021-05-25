"""
https://zenodo.org/record/2535967#.YKn_SS0RrOQ
"""

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from utils import DATA_DIR
import torch
import os
import numpy as np


def get_cifar10c(severity, **kwargs):
    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2023, 0.1994, 0.2010]

    transforms = [
            ToTensor(),
            Normalize(cifar_mean, cifar_std)
    ]

    test = CIFAR10C(
            path=os.path.join(DATA_DIR, 'CIFAR-10-C'),
            severity=severity,
            transform=Compose(transforms)
    )

    return test


class CIFAR10C(Dataset):

    def __init__(self, path, severity=5, transform=None):
        self.transform = transform

        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('npy') and not f.startswith('labels')]
        files.sort()

        labels_file = os.path.join(path, 'labels.npy')

        data = np.array([np.load(f)[:10000*severity] for f in files])
        labels = np.load(labels_file)[:10000*severity]

        x = np.concatenate(data)
        y = np.tile(labels, len(files))

        self.data_x = x
        self.data_y = y

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_x[idx]
        sample_y = self.data_y[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, sample_y
