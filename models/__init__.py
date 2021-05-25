from .simple_cnn import SimpleCNN, MixupSimpleCNN


def get_model(name, **kwargs):
    if name == "SimpleCNN":
        return SimpleCNN(**kwargs)
    elif name == "MixupSimpleCNN":
        return MixupSimpleCNN(**kwargs)
    else:
        raise KeyError(f"model `{name}` not defined")
