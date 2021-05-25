from itertools import product
import os
from datasets import get_dataset
from models import get_model
from trainers import get_default_trainer
import torch.utils.data
from callbacks import SavePredsCallback
import numpy as np
from pytorch_lightning.metrics.functional import accuracy
from utils import RESULTS_DIR


class Regularizers:

    def __init__(self, **kwargs):
        self.dataset_name = kwargs["dataset"]
        self.n_ensembles = kwargs['n_ensembles']  # one will serve as control
        self.n_epochs = kwargs['n_epochs']
        self.batch_size = kwargs['batch_size']
        self.dropouts = kwargs["dropouts"]
        self.mixup_alphas = kwargs["mixup_alphas"]

    def reproduce(self):

        for do, alpha in product(self.dropouts, self.mixup_alphas):
            self.run(do, alpha)

        self.report()

    def run(self, dropout, mixup_alpha):
        for i in range(self.n_ensembles):
            train, val, test = get_dataset()

            if mixup_alpha:
                model = get_model(name="MixupSimpleCNN", mixup_alpha=mixup_alpha, dropout=dropout)
            else:
                model = get_model(name="SimpleCNN", dropout=dropout)

            train_loader = torch.utils.data.DataLoader(
                    train,
                    num_workers=4,
                    batch_size=self.batch_size,
                    shuffle=True,
                    pin_memory=True
            )

            val_loader = torch.utils.data.DataLoader(
                    val,
                    num_workers=4,
                    batch_size=self.batch_size,
                    shuffle=False,
                    pin_memory=True
            )

            val_preds_callback = SavePredsCallback(val)

            path = os.path.join(RESULTS_DIR, f"do={dropout}_mixup={mixup_alpha}_model_{i}")
            trainer = get_default_trainer(path, self.n_epochs, [val_preds_callback])
            trainer.fit(model, train_loader, val_loader)

            os.makedirs(path, exist_ok=True)
            np.save(os.path.join(path, "val_preds.npy"), val_preds_callback.preds)

    def report(self,):
        _, val, _ = get_dataset(self.dataset_name)
        labels = torch.tensor([point[1] for point in val])

        for alpha, do in product(self.mixup_alphas, self.dropouts):
            ensemble_paths = [
                    os.path.join(RESULTS_DIR, f"do={do}_mixup={alpha}_model_{i}", "val_preds.npy")
                    for i in range(self.n_ensembles)
            ]

            control_path = ensemble_paths.pop(0)

            preds = {
                    "control": torch.tensor(np.load(control_path)),
                    "ensemble": self.ensemble_predictions(ensemble_paths)
            }

            print(f"accuracies for dropout={do}, mixup_alpha={alpha}: ")
            for k, v in preds.items():
                accs = self.epoch_accuracies(v, labels)
                print(
                        f"{k}: ",
                        f"{round(max(accs) * 100, 2)}%",
                        f"(epoch {torch.argmax(torch.tensor(accs))})"
                )

    @staticmethod
    def ensemble_predictions(paths):
        return torch.tensor(np.mean([np.load(path) for path in paths], axis=0), dtype=torch.float16)

    @staticmethod
    def single_predictions(path):
        return torch.tensor(np.load(path), dtype=torch.float16)

    def epoch_accuracies(self, preds, ref_labels, epochs=None):
        if epochs is None:
            epochs = self.n_epochs
        accuracies = []
        for epoch in range(epochs):
            accuracies.append(float(accuracy(preds[epoch], ref_labels)))
        return accuracies
