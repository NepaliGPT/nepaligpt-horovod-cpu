#!/usr/bin/env python3
"""
PyTorch Lightning + HorovodStrategy (CPU-only) on MNIST.

Lightning will set up DistributedSampler automatically with HorovodStrategy,
so you can pass a normal DataLoader; Lightning will inject the sampler.
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import horovod.torch as hvd
hvd.init()

from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.strategies import HorovodStrategy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(LightningModule):
    def __init__(self, lr=1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def training_step(self, batch, _):
        x, y = batch
        logits = self.net(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Scale LR with world size, as Horovod expects
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr * hvd.size(), momentum=0.9)


def make_loader():
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    # Lightning+HorovodStrategy will inject DistributedSampler; keep shuffle=True for rank 0
    return DataLoader(ds, batch_size=128, shuffle=True, num_workers=1)


if __name__ == "__main__":
    seed_everything(42 + hvd.rank(), workers=True)
    trainer = Trainer(
        max_epochs=int(os.environ.get("EPOCHS", 1)),
        strategy=HorovodStrategy(),
        accelerator="cpu",
        devices=hvd.size(),
        log_every_n_steps=25,
    )
    trainer.fit(Net(), train_dataloaders=make_loader())
