"""Tests for gcn: matches torch_geometric GCNConv given the same weights, gradcheck in float64
on a 5-node fake graph, output shapes of both heads.

Owner: Ayesha
"""

import numpy as np
import torch
from torch_geometric.nn import GCNConv

from tiny_timbre.gcn import FlatHead, GCNHead, GCNLayer, graph_conv
from tiny_timbre.graph import build_adjacency, normalize_adjacency

# Fake 5-genre tree: 1 is the root, 2 and 3 are its children, 4 and 5 are children of 2.
IDS = [1, 2, 3, 4, 5]
PARENTS = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
A = build_adjacency(IDS, PARENTS)
A_HAT = torch.tensor(normalize_adjacency(A))


def test_flat_head_shape():
    assert FlatHead(256, 97)(torch.randn(4, 256)).shape == (4, 97)


def test_flat_head_parameter_count():
    n = sum(p.numel() for p in FlatHead(256, 97).parameters())
    assert n == 257 * 97                  # 24,929


def test_layer_matches_pyg_gcnconv():
    torch.manual_seed(0)
    layer = GCNLayer(4, 3)
    with torch.no_grad():
        layer.bias.normal_()              # nonzero, so the bias placement is checked too
    conv = GCNConv(4, 3)                  # defaults: adds self-loops, symmetric normalization
    with torch.no_grad():
        conv.lin.weight.copy_(layer.weight.T)
        conv.bias.copy_(layer.bias)
    edge_index = torch.tensor(np.array(np.nonzero(A)), dtype=torch.long)
    H = torch.randn(5, 4)
    torch.testing.assert_close(layer(H, A_HAT), conv(H, edge_index), rtol=1e-5, atol=1e-6)


def test_graph_conv_gradcheck_float64():
    torch.manual_seed(0)
    H = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    W = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda H, W, b: graph_conv(A_HAT.double(), H, W, b), (H, W, b))


def test_gcn_head_gradcheck_float64():
    torch.manual_seed(0)
    head = GCNHead(A_HAT, out_dim=8, hidden_dim=6).double()
    x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(head, (x,))


def test_gcn_head_shapes_for_each_depth():
    x = torch.randn(4, 256)
    for depth in (1, 2, 3):
        head = GCNHead(A_HAT, num_layers=depth)
        assert len(head.layers) == depth
        assert head.detectors().shape == (5, 256)
        assert head(x).shape == (4, 5)


def test_gcn_head_parameter_count_for_c_97():
    A_hat = normalize_adjacency(np.zeros((97, 97), dtype=np.float32))
    n = sum(p.numel() for p in GCNHead(A_hat).parameters())
    assert n == (97 * 512 + 512) + (512 * 256 + 256)   # 181,504


def test_child_loss_reaches_parent_only_through_gcn():
    """A loss on genre 4 alone updates its parent's input row in the GCN, but not in the flat head.

    With H^0 = I, row j of the first weight matrix is genre j's own input. Two layers reach
    two hops: from 4 that is 2 (parent), 1 (grandparent) and 5 (sibling), but not 3.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 256)
    child, parent, far = IDS.index(4), IDS.index(2), IDS.index(3)

    gcn = GCNHead(A_HAT)
    gcn(x)[:, child].sum().backward()
    grad = gcn.layers[0].weight.grad
    assert grad[parent].abs().sum() > 0
    assert grad[far].abs().sum() == 0

    flat = FlatHead(256, 5)
    flat(x)[:, child].sum().backward()
    assert flat.linear.weight.grad[parent].abs().sum() == 0
