import copy
from typing import Dict

import lightning as L
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import Dice

from early_hyperacute_stroke_dataset import conf
from early_hyperacute_stroke_dataset.libs.network import init_network
from early_hyperacute_stroke_dataset.libs.loss import init_loss


class SegmentationModule(L.LightningModule):
    def __init__(
            self,
            network_params: Dict,
            loss_params: Dict,
            optimizer_params: Dict,
            scheduler_params: Dict
    ):
        super().__init__()

        self.save_hyperparameters()

        # Fix unused.
        _ = optimizer_params
        _ = scheduler_params

        self._network = init_network(network_params)
        self._activation = nn.Softmax(dim=1)

        self._loss_fn = init_loss(loss_params)
        self._dice_fn = Dice(ignore_index=conf.LABELS["background"], average="macro", num_classes=3, zero_division=1.0)

    def forward(self, x):
        predict = self._network(x)

        return self._activation(predict)

    def training_step(self, batch, batch_num):
        loss, dice = self._processing_batch(batch)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_dice", dice, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_num):
        loss, dice = self._processing_batch(batch)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_dice", dice, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_num):
        loss, dice = self._processing_batch(batch)

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_dice", dice, sync_dist=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), **self.hparams.optimizer_params)
        scheduler = ReduceLROnPlateau(optimizer, **self.hparams.scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def get_network(self):
        return self._network

    def _processing_batch(self, batch):
        input_tensor, target_tensor = batch

        predicts = self._network(input_tensor)

        loss = self._loss_fn(predicts, target_tensor)

        predicts = self._activation(predicts)
        dice = self._dice_fn(torch.argmax(predicts, axis=1), torch.argmax(target_tensor, axis=1))

        return loss, dice
