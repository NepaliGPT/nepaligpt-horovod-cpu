#!/usr/bin/env python3
"""
Plain PyTorch + Horovod (CPU-only) on MNIST.

Usage:
  srun -n 8 python -u src/pytorch/pt_mnist_cpu.py
  # or with mpirun/horovodrun, provided MPI env is set.
"""
import os, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import horovod.torch as hvd

from torch.utils.data import DataLoader, distributed
from torchvision import datasets, transforms


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    hvd.init()

    # Avoid CPU oversubscription
    torch.set_num_threads(max(1, torch.get_num_threads() // max(1, hvd.local_size())))

    # Data (rank-sharded)
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    sampler = distributed.DistributedSampler(
        ds, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True, drop_last=False
    )
    dl = DataLoader(ds, batch_size=128, sampler=sampler, num_workers=1)

    # Model & optimizer
    model = MNISTNet()
    base_lr = 1e-2
    opt = optim.SGD(model.parameters(), lr=base_lr * hvd.size(), momentum=0.9)

    # Wrap optimizer for distributed allreduce
    opt = hvd.DistributedOptimizer(opt, named_parameters=model.named_parameters())

    # Broadcast initial states from rank 0
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(opt, root_rank=0)

    # Train
    epochs = int(os.environ.get("EPOCHS", 1))
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        t0 = time.time()
        running = 0.0
        for step, (x, y) in enumerate(dl, start=1):
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu())

            if step % 50 == 0 and hvd.rank() == 0:
                print(f"[epoch {epoch} step {step}] loss={running/50:.4f}")
                running = 0.0

        if hvd.rank() == 0:
            print(f"Epoch {epoch} finished in {time.time()-t0:.2f}s")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/mnist_epoch{epoch}.pt")


if __name__ == "__main__":
    main()
