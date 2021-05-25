from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
from utils import DATA_DIR, RANDOM_SEED
import numpy as np


def get_cifar10(augment=False, test=False, **kwargs):
    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2023, 0.1994, 0.2010]

    train_val_idxs = list(range(50000))

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(train_val_idxs)

    train_idxs = train_val_idxs[:45000]
    val_idxs = train_val_idxs[45000:]

    transforms = [
            ToTensor(),
            Normalize(cifar_mean, cifar_std)
    ]

    # if augmentation is used during training,
    # we need to apply different transformation to test and validation sets
    # unfortunately we have to instantiate the same dataset object twice
    if augment:
        train_transforms = [
                ToTensor(),
                Normalize(cifar_mean, cifar_std),
                RandomCrop(32, 4),
                RandomHorizontalFlip()
        ]

    else:
        train_transforms = transforms

    train = Subset(
            CIFAR10(
                    train=True,
                    root=DATA_DIR,
                    download=True,
                    transform=Compose(train_transforms)
            ),
            train_idxs
    )

    validation = Subset(
            CIFAR10(
                    train=True,
                    root=DATA_DIR,
                    download=True,
                    transform=Compose(transforms)
            ),
            val_idxs
    )

    if test:
        test = CIFAR10(
                train=False,
                root=DATA_DIR,
                download=True,
                transform=Compose(transforms)
        )

    else:
        test = []

    return train, validation, test
