from models.litmodels.baselitmodellinear import BaseLitModelLinear
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)
from models.utils.utils import torch2np
from models.utils.sam import SAM
import os
import math
import torch
import torch.nn as nn


class SAMLitModelLinear(BaseLitModelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        self._adjust_learning_rate()
        optimizer = self.optimizers()

        label = batch[1]

        # first forward-backward pass
        first_logit = self(batch[0])
        pred = first_logit.argmax(axis=1)

        first_loss = self.criterion(first_logit, label)
        if torch.isnan(first_loss):
            import pdb

            pdb.set_trace()

        self.manual_backward(first_loss)
        optimizer.first_step(zero_grad=True)

        # second forward-backward pass
        second_logit = self(batch[0])

        second_loss = self.criterion(second_logit, label)
        if torch.isnan(second_loss):
            import pdb

            pdb.set_trace()

        self.manual_backward(second_loss)
        optimizer.second_step(zero_grad=True)

        label = label.argmax(axis=1)
        label, pred = torch2np(label), torch2np(pred)
        acc = accuracy_score(label, pred)

        self.log(
            "train_loss",
            first_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": first_loss}

    def configure_optimizers(self):
        base_optimizer = torch.optim.AdamW
        self.optimizer = SAM(
            self.parameters(),
            base_optimizer,
            rho=self.args.rho,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
        )

        return [self.optimizer]
