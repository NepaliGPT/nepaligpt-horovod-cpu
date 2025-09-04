def test_imports():
    import nepaligpt_hvd_cpu
    from nepaligpt_hvd_cpu.data import SyntheticLMDS, make_dataloader
    from nepaligpt_hvd_cpu.model import TinyLM

    assert SyntheticLMDS is not None
    assert TinyLM is not None
    # make_dataloader requires Horovod; we only assert it's importable
    assert callable(make_dataloader)
