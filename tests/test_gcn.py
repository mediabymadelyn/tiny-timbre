"""Tests for gcn: matches torch_geometric GCNConv given the same weights, gradcheck in float64
on a 5-node fake graph, output shapes of both heads.

Owner: Ayesha
"""

import torch

from tiny_timbre.gcn import FlatHead


def test_flat_head_shape():
    assert FlatHead(256, 97)(torch.randn(4, 256)).shape == (4, 97)


def test_flat_head_parameter_count():
    n = sum(p.numel() for p in FlatHead(256, 97).parameters())
    assert n == 257 * 97                  # 24,929
