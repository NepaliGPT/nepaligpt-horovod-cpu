import os
import random
import numpy as np
import torch
import pytest

def pytest_configure():
    # keep tokenizers quiet in CPU tests
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

@pytest.fixture(autouse=True)
def set_seed():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
