"""
SimpleCNN and its extension for Mixup training regime
"""
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning.metrics.functional as FM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleCNN(pl.LightningModule):

    def __init__(self, dropout=0):
        super().__init__()

        layers = [
                nn.Conv2d(3, 32, 3), nn.ReLU(),
                nn.Conv2d(32, 32, 3), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3), nn.ReLU(),
                nn.Conv2d(64, 64, 3), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
        ]

        if dropout:
            layers.append(nn.Dropout(float(dropout)))

        layers.append(nn.Linear(1600, 128))
        layers.append(nn.Linear(128, 10))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.network(x)
        loss = F.cross_entropy(pred, y)
        acc = FM.accuracy(pred, y)

        self.log_dict({
                'train_loss': loss,
                'train_acc': acc
        }, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.network(x)
        loss = F.cross_entropy(pred, y)
        acc = FM.accuracy(pred, y)

        self.log_dict({
                'train_loss': loss,
                'train_acc': acc
        }, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class MixupSimpleCNN(SimpleCNN):

    def __init__(self, mixup_alpha, **kwargs):
        super().__init__(**kwargs)
        self.mixup_alpha = float(mixup_alpha)

    @staticmethod
    def mixup_data(x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(DEVICE)

        mixed_x = lam * x + (1 - lam) * x[index, :]

        y = F.one_hot(y, num_classes=10)
        mixed_y = lam * y + (1 - lam) * y[index, :]

        return mixed_x, mixed_y, lam

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y, lam = self.mixup_data(x, y, self.mixup_alpha)

        pred = self.network(x)
        loss = self.cross_entropy(pred, y)

        return loss

    @staticmethod
    def cross_entropy(pred, y):
        log_prob = F.log_softmax(pred, dim=1)
        return -(y*log_prob).sum(dim=1).mean()
