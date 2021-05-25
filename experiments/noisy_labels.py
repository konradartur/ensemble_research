from itertools import product
from models import get_model
from datasets import get_dataset
from trainers import get_default_trainer
from callbacks import SavePredsCallback
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
from scipy.stats import entropy
from torch.nn.functional import softmax
from pytorch_lightning.metrics.functional.classification import accuracy
import torch
import matplotlib.pyplot as plt
import pickle
from utils import RESULTS_DIR, RANDOM_SEED, DISRUPTIONS


class NoisyLabels:

    def __init__(self, **kwargs):
        self.model_name = kwargs["model"]
        self.dataset_name = kwargs["dataset"]
        self.n_ensembles = kwargs['n_ensembles']
        self.n_epochs = kwargs['n_epochs']
        self.batch_size = kwargs['batch_size']
        self.severity = kwargs["severity"]
        self.num_rand_labels = kwargs["num_rand_labels"]

    def reproduce(self):

        for i in self.num_rand_labels:
            self.run(i, self.severity)
        self.pickle_preds()  # save logits

        self.repickle_preds(self.severity)  # saves accuracies and entropies
        self.plot_id_vs_ood_accs()
        self.plot_id_vs_ood_ents()

        self.repickle_preds_disruptions()  # saves accuracies and entropies per each disruption
        self.plot_per_disruption_grid()

    def run(self, num_rand_labels, severity):
        train, val, test = get_dataset(self.dataset_name, augument=True, test=True)
        new_train = self.some_randomization(train, RANDOM_SEED, num_rand_labels) if num_rand_labels > 0 else train
        test_corrupted = get_dataset("cifar10c", severity=severity)

        print(len(new_train), len(val), len(test), len(test_corrupted))

        for i in range(self.n_ensembles):
            path = os.path.join(RESULTS_DIR, f'rand_labels={num_rand_labels}_nth_model={i}')

            model = get_model(self.model_name)

            train_loader = DataLoader(
                    new_train,
                    num_workers=4,
                    batch_size=self.batch_size,
                    shuffle=True,
                    pin_memory=True
            )

            val_loader = DataLoader(
                    val,
                    num_workers=4,
                    batch_size=self.batch_size,
                    shuffle=False,
                    pin_memory=True
            )

            train_save = SavePredsCallback(new_train)
            val_save = SavePredsCallback(val)
            test_save = SavePredsCallback(test)
            test_corrupted_save = SavePredsCallback(test_corrupted)

            trainer = get_default_trainer(path, self.n_epochs, [train_save, val_save, test_save, test_corrupted_save])

            trainer.fit(model, train_loader, val_loader)

            os.makedirs(path, exist_ok=True)
            np.save(os.path.join(path, "train_preds.npy"), train_save.preds)
            np.save(os.path.join(path, "val_preds.npy"), val_save.preds)
            np.save(os.path.join(path, "test_clean_preds.npy"), test_save.preds)
            np.save(os.path.join(path, "test_corrupted_preds.npy"), test_corrupted_save.preds)

    def report(self, **kwargs):
        pass

    def pickle_preds(self):
        for num_rand_labels in self.num_rand_labels:
            print(f"pickling dictionary for {num_rand_labels} noised labels")

            preds = {}
            for set_ in ["train", "val", "test_clean", "test_corrupted"]:
                ensemble_paths = []

                for n in range(self.n_ensembles):
                    cur_path = os.path.join(
                            RESULTS_DIR,
                            f"rand_labels={num_rand_labels}_nth_model={n}",
                            f"{set_}_preds.npy"
                    )

                    preds[f"single{n}_{set_}"] = self.single_predictions(cur_path)
                    ensemble_paths.append(cur_path)

                preds[f"ensemble_{set_}"] = self.ensemble_predictions(ensemble_paths)

            save_path = os.path.join(RESULTS_DIR, f"model_set_{num_rand_labels}.pkl")
            file = open(save_path, "wb")
            pickle.dump(preds, file)
            file.close()

    def repickle_preds(self, severity):
        train, val, test_clean = get_dataset(self.dataset_name, test=True)
        test_corrupted = get_dataset("cifar10c", severity=severity)

        models = [f"single{i}" for i in range(self.n_ensembles)] + ["ensemble"]
        sets = ["train", "val", "test_clean", "test_corrupted"]
        labels = {"train": torch.tensor([point[1] for point in train]),
                  "val": torch.tensor([point[1] for point in val]),
                  "test_clean": torch.tensor([point[1] for point in test_clean]),
                  "test_corrupted": torch.tensor([point[1] for point in test_corrupted])}

        for num_rand_labels in self.num_rand_labels:
            print(f"pickling dictionary for {num_rand_labels} noised labels")

            load_path = os.path.join(RESULTS_DIR, f"model_set_{num_rand_labels}.pkl")

            file = open(load_path, "rb")
            preds = pickle.load(file)
            file.close()

            accs = {}
            ents = {}
            for model, set_ in product(models, sets):
                accs[f"{model}_{set_}"] = self.accuracies_per_epoch(preds[f"{model}_{set_}"].type(torch.float), labels[f"{set_}"])
                ents[f"{model}_{set_}"] = self.entropies_per_epoch(preds[f"{model}_{set_}"].type(torch.float))

            save_path = os.path.join(RESULTS_DIR, f"model_set_{num_rand_labels}_accs.pkl")
            file = open(save_path, "wb")
            pickle.dump(accs, file)
            file.close()

            save_path = os.path.join(RESULTS_DIR, f"model_set_{num_rand_labels}_ents.pkl")
            file = open(save_path, "wb")
            pickle.dump(ents, file)
            file.close()

    def plot_id_vs_ood_accs(self):
        models = [f"single{i}" for i in range(self.n_ensembles)] + ["ensemble"]
        sets = ["test_clean", "test_corrupted"]

        fig, axs = plt.subplots(1, len(self.num_rand_labels), figsize=(12, 4), sharex=True, sharey=True)
        for ax, num_rand_labels in zip(axs.flat, self.num_rand_labels):

            colors = ["#FFB266", "#FF8000", "#66B2FF", "#004C99"]
            load_path = os.path.join(RESULTS_DIR, f"model_set_{num_rand_labels}_accs.pkl")
            file = open(load_path, "rb")
            points = pickle.load(file)
            file.close()

            for (model, set_) in product(models, sets):
                idx = 0
                if model == "ensemble":
                    idx += 2
                if set_ == "test_corrupted":
                    idx += 1
                ax.plot(range(self.n_epochs),
                        points[f"{model}_{set_}"],
                        color=colors[idx])

            ax.set_title(f"training with {str(num_rand_labels // 500) + '%' if num_rand_labels > 0 else 'no'} noisy labels")
            ax.grid()
            ax.set_xticklabels([])
            if num_rand_labels == 0:
                lines = [plt.Line2D([0], [0], color=color) for color in colors]
                labels = ["singles on clean",
                          "singles on corrupted",
                          "ensemble on clean",
                          "ensemble on corrupted"]
                ax.legend(lines,
                          labels,
                          loc="lower right")

        fig.tight_layout()
        save_path = os.path.join(RESULTS_DIR, "plots")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "id_vs_ood_accs.png"))

    def plot_id_vs_ood_ents(self):
        models = [f"single{i}" for i in range(self.n_ensembles)] + ["ensemble"]
        sets = ["test_clean", "test_corrupted"]
        colors = ["#FF8000", "#FFB266", "#FFE5CC", "#66B2FF"]

        fig, axs = plt.subplots(len(sets), len(self.num_rand_labels), figsize=(15, 7), sharex=True, sharey=True)

        for num_rand_labels in self.num_rand_labels:
            load_path = os.path.join(RESULTS_DIR, f"model_set_{num_rand_labels}_ents.pkl")
            file = open(load_path, "rb")
            points = pickle.load(file)
            file.close()

            for set_ in sets:
                row = sets.index(set_)
                column = self.num_rand_labels.index(num_rand_labels)
                ax = axs[row, column]
                ax.set_ylim(2.1, 2.31)

                c_idx = 0
                for model, color in zip(models, colors):
                    ax.plot(range(self.n_epochs),
                            points[f"{model}_{set_}"],
                            label=f"{model}",
                            color=colors[c_idx])
                    c_idx += 1

                if row is 0:
                    ax.set_title(f"training with"
                                 f" {str(num_rand_labels // 500) + '%' if num_rand_labels > 0 else 'no'} "
                                 f"noisy labels")
                ax.grid()
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                if row is 0 and column is 0:
                    ax.legend(loc="lower right")

        fig.tight_layout()
        save_path = os.path.join(RESULTS_DIR, "plots")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "id_vs_ood_ents.png"))

    def repickle_preds_disruptions(self):
        _, _, test_clean = get_dataset(self.dataset_name, test=True)

        models = [f"single{i}" for i in range(self.n_ensembles)] + ["ensemble"]
        sets = ["test_clean", "test_corrupted"]
        labels = torch.tensor([point[1] for point in test_clean])

        for num_rand_labels in self.num_rand_labels:

            load_path = os.path.join(RESULTS_DIR, f"model_set_{num_rand_labels}.pkl")

            file = open(load_path, "rb")
            preds = pickle.load(file)
            file.close()

            accs = {}
            ents = {}
            for model, set_ in product(models, sets):
                tmp_preds = preds[f"{model}_{set_}"].type(torch.float)

                if set_ == "test_corrupted":
                    idx = 0
                    for disruption in DISRUPTIONS:
                        for s in range(self.severity):
                            cur_preds = tmp_preds[:, idx:(idx + 10000)]
                            accs[f"{model}_{disruption}_{s + 1}"] = self.accuracies_per_epoch(cur_preds, labels)
                            ents[f"{model}_{disruption}_{s + 1}"] = self.entropies_per_epoch(cur_preds)

                            idx += 10000
                else:
                    accs[f"{model}_clean"] = self.accuracies_per_epoch(tmp_preds, labels)
                    ents[f"{model}_clean"] = self.entropies_per_epoch(tmp_preds)

            save_path = os.path.join(RESULTS_DIR, f"model_test_{num_rand_labels}_accs.pkl")
            file = open(save_path, "wb")
            pickle.dump(accs, file)
            file.close()

            save_path = os.path.join(RESULTS_DIR, f"model_test_{num_rand_labels}_ents.pkl")
            file = open(save_path, "wb")
            pickle.dump(ents, file)
            file.close()

    def plot_per_disruption_grid(self):
        models = [f"single{i}" for i in range(self.n_ensembles)] + ["ensemble"]
        metrics = ["accs", "ents"]
        points = {"accs": {}, "ents": {}}

        for num_rand_labels, metric in product(self.num_rand_labels, metrics):

            load_path = os.path.join(RESULTS_DIR, f"model_test_{num_rand_labels}_{metric}.pkl")

            file = open(load_path, "rb")
            points[metric][f"{num_rand_labels}"] = pickle.load(file)
            file.close()

        for metric in metrics:
            for disruption in DISRUPTIONS:

                fig, axs = plt.subplots(self.severity, len(self.num_rand_labels),
                                        figsize=(int(len(self.num_rand_labels) * 3.5), int(self.severity * 3.5)),
                                        sharex=True, sharey=True)

                for severity, num_rand_labels in product(range(self.severity), self.num_rand_labels):

                    tmp_points = points[metric][f"{num_rand_labels}"]
                    row = severity
                    column = self.num_rand_labels.index(num_rand_labels)
                    ax = axs[row, column]

                    for model in models:
                        name = disruption + '_' + str(severity + 1)
                        ax.plot(range(self.n_epochs), tmp_points[f"{model}_{name}"], label=f"{model}")

                    ax.grid()
                    ax.set_xticklabels([])
                    if row is 0:
                        ax.set_title((f"{num_rand_labels // 500}%" if num_rand_labels > 0 else "no") + " random labels")
                    if column is 0:
                        ax.set_ylabel(f"severity {severity + 1}")
                    if row is 0 and column is 0:
                        pos = "upper right" if disruption == "glass_blur" else "lower right"
                        ax.legend(loc=pos)

                fig.tight_layout()
                save_path = os.path.join(RESULTS_DIR, "plots")
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f"{disruption}_{metric}.png"))
                plt.close(fig)

    @staticmethod
    def some_randomization(data, seed, num_rand_labels):
        assert num_rand_labels > 0
        images = torch.stack([data[i][0] for i in range(len(data))])

        generator = torch.Generator().manual_seed(seed)
        random_labels = torch.randint(0, 10, size=(num_rand_labels,), generator=generator)
        clean_labels = torch.tensor([data[i][1] for i in range(num_rand_labels, len(data))])
        labels = torch.cat((random_labels, clean_labels)).type(torch.LongTensor)

        return TensorDataset(images, labels)

    @staticmethod
    def ensemble_predictions(paths):
        return torch.tensor(np.mean([np.load(path, allow_pickle=True) for path in paths], axis=0), dtype=torch.float16)

    @staticmethod
    def single_predictions(path):
        return torch.tensor(np.load(path, allow_pickle=True), dtype=torch.float16)

    @staticmethod
    def label_predictions(predictions):
        return torch.argmax(softmax(predictions, dim=-1), dim=-1)

    def accuracies_per_epoch(self, preds, ref_labels):
        accuracies = []
        for epoch in range(self.n_epochs):
            accuracies.append(float(accuracy(self.label_predictions(preds[epoch]), ref_labels)))
        return accuracies

    def entropies_per_epoch(self, preds):
        entropies = []
        for epoch in range(self.n_epochs):
            _, counts = np.unique(self.label_predictions(preds[epoch]).numpy(), return_counts=True)
            entropies.append(entropy(counts))
        return entropies
