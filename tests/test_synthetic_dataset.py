import torch
from nepaligpt_hvd_cpu.data import SyntheticLMDS

def test_synthetic_dataset_shapes():
    ds = SyntheticLMDS(n=10, seq=8, vocab=50)
    x, y = ds[0]
    assert x.shape == (8,)
    assert y.shape == (8,)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert len(ds) == 10

def test_synthetic_dataset_shift():
    ds = SyntheticLMDS(n=1, seq=5, vocab=10, seed=42)
    x, y = ds[0]
    # y is x shifted left by 1 (next-token objective)
    assert torch.equal(y[:-1], x[1:])
    # last label is the first token
    assert y[-1] == x[0]
