from .noisy_labels import *
from .delayed_ensemble import *
from .regularizers import *


def reproduce_experiment(name, **kwargs):

    if name == "noisy_labels":
        experiment = NoisyLabels(**kwargs)
    elif name == "delayed_ensemble":
        experiment = DelayedEnsemble(**kwargs)
    elif name == "regularizers":
        experiment = Regularizers(**kwargs)
    else:
        raise NotImplementedError(f"experiment {name} not defined")

    experiment.reproduce()
