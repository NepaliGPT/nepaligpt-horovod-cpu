import torch
from nepaligpt_hvd_cpu.model import TinyLM

def test_tinylm_forward_shape():
    vocab, d, B, T = 100, 32, 2, 7
    model = TinyLM(vocab=vocab, d=d)
    x = torch.randint(1, vocab, (B, T))
    logits = model(x)
    assert logits.shape == (B, T, vocab)
