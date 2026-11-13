import torch
from torch import nn
from tiny_timbre.model import Model


def test_output_shape():
    model = Model(head=nn.Linear(256, 97))
    x = torch.randn(4, 1, 128, 431)
    assert model(x).shape == (4, 97)
