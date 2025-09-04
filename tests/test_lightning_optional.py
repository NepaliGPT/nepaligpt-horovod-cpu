# tests/test_lightning_optional.py
import importlib
import pytest

lightning = importlib.util.find_spec("lightning")
torchvision = importlib.util.find_spec("torchvision")

pytestmark = pytest.mark.skipif(
    lightning is None or torchvision is None,
    reason="Lightning or torchvision not installed in this environment",
)

def test_lightning_imports():
    import lightning
    from lightning.pytorch.strategies import HorovodStrategy
    assert lightning is not None
    assert HorovodStrategy is not None
