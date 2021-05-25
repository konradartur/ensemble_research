from pytorch_lightning.callbacks import Callback
import torch
import torch.utils.data
import numpy as np


class SavePredsCallback(Callback):

    def __init__(self, data):
        super().__init__()
        self.loader = torch.utils.data.DataLoader(data, num_workers=4, batch_size=128, shuffle=False)
        self.preds = []

    def __pred_on_loader(self, model):
        tmp_preds = []

        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            for x, _ in self.loader:
                pred = model(x.to(device)).cpu().detach().numpy()
                tmp_preds.append(pred)
        model.train()
        return tmp_preds

    def on_train_epoch_end(self, trainer, model, outputs):
        self.preds.append(np.concatenate(self.__pred_on_loader(model)))
