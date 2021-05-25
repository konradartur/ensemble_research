import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pickle
from torch.utils.data import DataLoader
from callbacks import SavePredsCallback
from datasets import get_dataset
from models import get_model
from trainers import get_default_trainer
from sklearn.manifold import TSNE
from pytorch_lightning.metrics.functional import accuracy
from utils import RESULTS_DIR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DelayedEnsemble:

    def __init__(self, **kwargs):
        self.model_name = kwargs["model"]
        self.dataset_name = kwargs["dataset"]
        self.n_ensembles = kwargs["n_ensembles"]
        self.n_epochs = kwargs["n_epochs"]
        self.batch_size = kwargs["batch_size"]
        self.delayed_starts = kwargs["delayed_starts"]

    def reproduce(self):

        for start in self.delayed_starts:
            self.run(start)
            self.pickle_dict(start)

        self.plot_val_acc()
        self.pickle_points_for_tsne(50, 30)
        self.plot_tsne(30)

    def run(self, ensemble_after):
        train, val, _ = get_dataset(self.dataset_name, augument=True)

        train_loader = DataLoader(
                train,
                num_workers=4,
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=True
        )
        val_loader = DataLoader(
                val,
                num_workers=4,
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=False
        )

        init_model_path = os.path.join(RESULTS_DIR, f'ensemble_after={ensemble_after}', 'base_model')

        if ensemble_after > 0:  # train base model
            train_save = SavePredsCallback(train)
            val_save = SavePredsCallback(val)

            model = get_model(self.model_name)

            trainer = get_default_trainer(init_model_path, ensemble_after, [train_save, val_save])
            trainer.fit(model, train_loader, val_loader)

            os.makedirs(init_model_path, exist_ok=True)
            trainer.save_checkpoint(os.path.join(init_model_path, 'init_weights.ckpt'))
            np.save(os.path.join(init_model_path, 'train_preds.npy'), train_save.preds)
            np.save(os.path.join(init_model_path, 'val_preds.npy'), val_save.preds)

        rem_epochs = self.n_epochs - ensemble_after

        if rem_epochs > 0:  # fork base into ensemble
            for i in range(self.n_ensembles):
                model_path = os.path.join(RESULTS_DIR, f'ensemble_after={ensemble_after}', f'ensemble_{i}')

                train_save = SavePredsCallback(train)
                val_save = SavePredsCallback(val)

                if ensemble_after > 0:  # if trained, load
                    model = get_model(self.model_name).load_from_checkpoint(os.path.join(init_model_path, 'init_weights.ckpt'))
                else:
                    model = get_model(self.model_name)

                trainer = get_default_trainer(model_path, rem_epochs, [train_save, val_save])
                trainer.fit(model, train_loader, val_loader)

                os.makedirs(model_path, exist_ok=True)
                np.save(os.path.join(model_path, 'train_preds.npy'), train_save.preds)
                np.save(os.path.join(model_path, 'val_preds.npy'), val_save.preds)

    def report(self, **kwargs):
        pass

    def pickle_dict(self, ensemble_after):

        preds = {}
        for set_ in ["train", "val"]:
            if ensemble_after > 0:
                base_path = os.path.join(
                        RESULTS_DIR,
                        f"ensemble_after={ensemble_after}",
                        "base_model",
                        f"{set_}_preds.npy"
                )

                preds[f"base_{set_}"] = self.single_predictions(base_path)

            if ensemble_after < self.delayed_starts[-1]:
                all_paths = []
                for i in range(self.n_ensembles):
                    ens_path = os.path.join(
                            RESULTS_DIR,
                            f"ensemble_after={ensemble_after}",
                            f"ensemble_{i}",
                            f"{set_}_preds.npy"
                    )
                    all_paths.append(ens_path)
                    preds[f"ens{i}_{set_}"] = self.single_predictions(ens_path)
                preds[f"ensemble_{set_}"] = self.ensemble_predictions(all_paths)

        save_path = os.path.join(RESULTS_DIR, f"model_set_{ensemble_after}.pkl")
        file = open(save_path, "wb")
        pickle.dump(preds, file)
        file.close()

    def plot_val_acc(self, **kwargs):
        _, val, _ = get_dataset(self.dataset_name, **kwargs)
        labels = torch.tensor([point[1] for point in val])

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = ["#bad80a", "#009e49", "#00b294", "#00bcf2", "#00188f", "#68217a", "#ec008c", "#e81123", "#ff8c00", "#fff100"]

        for start, color in zip(self.delayed_starts, colors):

            load_path = os.path.join(RESULTS_DIR, f"model_set_{start}.pkl")

            file = open(load_path, "rb")
            preds = pickle.load(file)
            file.close()

            base_preds = preds["base_val"].type(torch.float) if start > 0 else None
            base_accs = self.epoch_accuracies(base_preds, labels, start)

            ensemble_preds = preds["ensemble_val"].type(torch.float) if start < self.n_epochs else None
            ensemble_accs = self.epoch_accuracies(ensemble_preds, labels, self.n_epochs - start)

            accs = base_accs + ensemble_accs
            label = f"ensemble since {start}" if start < self.n_epochs else "no ensemble"

            ax.plot(range(self.n_epochs), accs, color=color, label=label)

        ax.legend(loc="lower right")
        ax.grid()
        ax.set_title("validation accuracy during training")
        # ax.set_xlim(0, 210)
        # ax.set_ylim(0.73, 0.87)
        fig.tight_layout()

        save_path = os.path.join(RESULTS_DIR, "plots")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"delayed_ensemble_accuracies.png"))

    def pickle_points_for_tsne(self, sampling_density, perplexity):

        load_path = os.path.join(RESULTS_DIR, "model_set_0.pkl")

        file = open(load_path, "rb")
        zero_model = pickle.load(file)
        file.close()

        # load base model used in comparision with others
        zero_points = []
        for i in range(self.n_ensembles):
            data = (zero_model[f"ens{i}_train"][:, ::sampling_density]).type(torch.float)
            epochs, examples, logits = data.shape
            data = np.reshape(data, (epochs, examples * logits))
            zero_points.append(data)

        for start in self.delayed_starts[1:-1]:  # skip base and baseline
            print(start)

            load_path = os.path.join(RESULTS_DIR, f"model_set_{start}.pkl")

            file = open(load_path, "rb")
            delayed_model = pickle.load(file)
            file.close()

            delayed_points_base = []

            data = (delayed_model[f"base_train"][:, ::sampling_density]).type(torch.float)
            epochs, examples, logits = data.shape
            data = np.reshape(data, (epochs, examples * logits))
            delayed_points_base.append(data)

            delayed_points_ens = []
            for i in range(self.n_ensembles):
                data = (delayed_model[f"ens{i}_train"][:, ::sampling_density]).type(torch.float)
                epochs, examples, logits = data.shape
                data = np.reshape(data, (epochs, examples * logits))
                delayed_points_ens.append(data)

            tsne = TSNE(perplexity=perplexity)
            points = np.concatenate((
                    *zero_points,
                    *delayed_points_base,
                    *delayed_points_ens
            ))

            transformed = tsne.fit_transform(points)

            points = {}
            idx = 0

            for i in range(self.n_ensembles):
                points[f"e0_traj{i}"] = transformed[idx: (idx + self.n_epochs)]
                idx += self.n_epochs

            points["base_traj"] = transformed[idx: (idx + start)]
            idx += start

            for i in range(self.n_ensembles):
                points[f"eN_traj{i}"] = transformed[idx: (idx + self.n_epochs - start)]
                idx += self.n_epochs - start

            save_path = os.path.join(RESULTS_DIR, f"tsne_points_perplexity={perplexity}_start={start}.pkl")
            file = open(save_path, "wb")
            pickle.dump(points, file)
            file.close()

    def plot_tsne(self, perplexity):
        zero_colors = ["#005b96", "#6497b1", "#b3cde0"]
        delayed_base_color = "#ad6347"
        delayed_ens_colors = ["#f0e2a8", "#e8ca93", "#9a8262"]

        starts = self.delayed_starts[1:-1]
        fig, axes = plt.subplots(1, len(starts), figsize=(4*len(starts), 4))

        for ax, start in zip(axes.flatten(), starts):

            load_path = os.path.join(
                    RESULTS_DIR,
                    f"tsne_points_perplexity={perplexity}_start={start}.pkl"
            )

            file = open(load_path, "rb")
            points = pickle.load(file)
            file.close()

            ax.set_xticks([])
            ax.set_xlabel("")
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.set_title(f"delayed_start={start}")

            idx = 0

            for i in range(self.n_ensembles):  # plotting en0 trajectories
                ax.plot(
                        points[f"e0_traj{i}"][:, 0],
                        points[f"e0_traj{i}"][:, 1],
                        color=zero_colors[i],
                        marker='o',
                        markersize=5,
                        label=f"baseline {i}"
                )
                idx += self.n_epochs

            ax.plot(  # plotting bmj trajectory
                    points[f"base_traj"][:, 0],
                    points[f"base_traj"][:, 1],
                    color=delayed_base_color,
                    marker='o',
                    markersize=5,
                    label=f"base for delayed"
            )
            idx += start

            for i in range(self.n_ensembles):  # plotting enj trajectories
                ax.plot(
                        points[f"eN_traj{i}"][:, 0],
                        points[f"eN_traj{i}"][:, 1],
                        color=delayed_ens_colors[i],
                        marker='o',
                        markersize=5,
                        label=f"delayed ensemble {i}"
                )
                idx += self.n_epochs - start

            if start is starts[-1]:
                ax.legend(loc="lower left")

        fig.tight_layout()
        save_path = os.path.join(RESULTS_DIR, "plots")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"tsne_trajectories.png"))

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
