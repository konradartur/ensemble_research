from .cifar10 import get_cifar10
from .cifar10c import get_cifar10c


def get_dataset(name="cifar10", **kwargs):
    if name == "cifar10":
        return get_cifar10(**kwargs)
    elif name == "cifar10c":
        return get_cifar10c(**kwargs)
    else:
        raise NotImplementedError(f"dataset `{name}` not defined")
