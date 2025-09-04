#!/usr/bin/env python3
"""
Plain PyTorch + Horovod (CPU-only) with a tiny LM over synthetic token data.
Mimics next-token prediction to validate distributed throughput without dataset downloads.
"""
import os, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import horovod.torch as hvd
from torch.utils.data import Dataset, DataLoader, distributed


class ToySeqDataset(Dataset):
    def __init__(self, n=20000, seq=128, vocab=5000, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(1, vocab, (n, seq), generator=g)
        self.vocab = vocab

    def __len__(self): return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = torch.roll(x, shifts=-1, dims=0)  # next-token objective
        return x, y


class TinyLM(nn.Module):
    def __init__(self, vocab=5000, d=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.lm = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.emb(x)  # (B, T, d)
        h = self.ff(h)   # (B, T, d)
        return self.lm(h)  # (B, T, vocab)


def main():
    hvd.init()
    torch.set_num_threads(max(1, torch.get_num_threads() // max(1, hvd.local_size())))

    # Dataset & sharded loader
    ds = ToySeqDataset(n=int(os.environ.get("N", 20000)))
    sampler = distributed.DistributedSampler(
        ds, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True, drop_last=False
    )
    dl = DataLoader(ds, batch_size=int(os.environ.get("B", 32)), sampler=sampler, num_workers=1)

    model = TinyLM(vocab=ds.vocab, d=int(os.environ.get("D", 256)))
    base_lr = float(os.environ.get("LR", 3e-3))
    opt = optim.AdamW(model.parameters(), lr=base_lr * hvd.size())
    opt = hvd.DistributedOptimizer(opt, named_parameters=model.named_parameters())

    # Sync initial states
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(opt, root_rank=0)

    epochs = int(os.environ.get("EPOCHS", 2))
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        t0 = time.time()
        loss_acc = 0.0
        for step, (x, y) in enumerate(dl, start=1):
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            opt.step()
            loss_acc += float(loss.detach().cpu())

            if step % 50 == 0 and hvd.rank() == 0:
                print(f"[epoch {epoch} step {step}] loss={loss_acc/50:.4f}")
                loss_acc = 0.0

        if hvd.rank() == 0:
            print(f"Epoch {epoch} duration: {time.time()-t0:.2f}s")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/synth_epoch{epoch}.pt")


if __name__ == "__main__":
    main()
