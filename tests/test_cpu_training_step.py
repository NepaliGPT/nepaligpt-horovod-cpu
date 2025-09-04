import torch
import torch.nn.functional as F
from torch.optim import AdamW
from nepaligpt_hvd_cpu.model import TinyLM

def test_single_cpu_training_step():
    vocab, d, B, T = 120, 64, 2, 8
    model = TinyLM(vocab=vocab, d=d)
    opt = AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(1, vocab, (B, T))
    y = torch.roll(x, shifts=-1, dims=1)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    opt.zero_grad()
    loss.backward()
    opt.step()

    # loss should be finite and positive
    assert torch.isfinite(loss).item()
    assert loss.item() > 0.0
