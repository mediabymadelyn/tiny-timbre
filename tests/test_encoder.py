import torch
from tiny_timbre.encoder import Encoder


def test_output_shape():
    x = torch.randn(4, 1, 128, 431)        # 4 fake 10-second spectrograms
    assert Encoder()(x).shape == (4, 256)


def test_any_length_works():
    x = torch.randn(2, 1, 128, 1292)       # full 30 s clip (used at test time)
    assert Encoder()(x).shape == (2, 256)


def test_parameter_count():
    n = sum(p.numel() for p in Encoder().parameters())
    assert n == 388_800                   # 387,840 conv + 960 BatchNorm
