from lightning import LightningModule
from easydict import EasyDict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)
from models.utils.utils import torch2np
import os
import math
import torch
import torch.nn as nn


class BaseLitModelLinear(LightningModule):
    def __init__(self, model: nn.Module, args: EasyDict):
        super().__init__()

        self.model = model

        self.criterion = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, x, debug=False):
        return self.model(x, debug=debug)

    def training_step(self, batch, batch_idx):
        self._adjust_learning_rate()
        label = batch[1]

        logit = self(batch[0])
        pred = logit.argmax(axis=1)

        loss = self.criterion(logit, label)
        if torch.isnan(loss):
            import pdb

            pdb.set_trace()

        label, pred = torch2np(label), torch2np(pred)
        label = label.argmax(axis=1)
        acc = accuracy_score(label, pred)

        self.log(
            "train_loss",
            loss,
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

        return {"loss": loss}

    def evaluate(self, batch, stage=None):
        label = batch[1]

        logit = self(batch[0])

        pred = logit.argmax(axis=1)

        loss = self.criterion(logit, label)
        label, pred = torch2np(label), torch2np(pred)
        label = label.argmax(axis=1)

        acc = accuracy_score(label, pred)
        kappa = cohen_kappa_score(label, pred)
        f1 = f1_score(label, pred, average="macro")

        if stage:
            self.log(
                f"{stage}_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_kappa",
                kappa,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_f1",
                f1,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage="eval")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logit = self(batch[0])
        pred = logit.argmax(axis=1)

        return pred

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
        )

        return {"optimizer": self.optimizer}

    def _adjust_learning_rate(self):
        """Decay the learning rate based on schedule"""
        warmup_epoch = self.args.EPOCHS // 10 if self.args.EPOCHS <= 100 else 40

        if self.current_epoch < warmup_epoch:
            cur_lr = self.args.lr * self.current_epoch / warmup_epoch + 1e-9
        else:
            cur_lr = (
                self.args.lr
                * 0.5
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * (self.current_epoch - warmup_epoch)
                        / (self.args.EPOCHS - warmup_epoch)
                    )
                )
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = cur_lr

    def on_validation_epoch_end(self):
        self.save_checkpoint()

    def save_checkpoint(self):
        """🔥 Custom Checkpoint 저장 메서드"""
        checkpoint_dir = f"{self.args.CKPT_PATH}/{self.args.experiment_name}/{self.args.current_time}_{self.args.msg}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, f"S{self.args.target_subject:02d}.ckpt"
        )

        checkpoint = {
            "epoch": self.current_epoch,
            "state_dict": self.state_dict(),
        }

        torch.save(checkpoint, checkpoint_path)
