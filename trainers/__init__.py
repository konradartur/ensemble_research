from pytorch_lightning import Trainer
from torch.cuda import is_available


def get_default_trainer(path, epochs, callbacks=None, gpus=1):

    if not callbacks:
        callbacks = []

    return Trainer(
            default_root_dir=path,
            logger=False,
            checkpoint_callback=False,
            callbacks=callbacks,
            max_epochs=epochs,
            gpus=gpus if is_available() else 0,
    )

