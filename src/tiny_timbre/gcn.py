"""From-scratch GCN layer, GCN head over the genre graph, and the flat multi-label head.

GCN layer:  H^(l+1) = ReLU(A_hat H^(l) W^(l))   (no ReLU after the last layer)
GCN head:   H^0 = I (one-hot), two layers C -> 512 -> 256, scores z = H^(L) x
Flat head:  Linear 256 -> C

Every head maps the encoder output x of shape (batch, 256) to logits of shape (batch, C),
with no sigmoid: the loss (binary_cross_entropy_with_logits) applies it.

Owner: Ayesha
"""

import numpy as np
import torch
from torch import nn


def graph_conv(A_hat: torch.Tensor, H: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A_hat H W + b, before any activation. Same order as PyG GCNConv: transform, propagate, add bias."""
    return A_hat @ (H @ W) + b


class GCNLayer(nn.Module):
    """One graph convolution over C nodes: (C, in_dim) -> (C, out_dim)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)   # Glorot, as in Kipf & Welling and PyG

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        return graph_conv(A_hat, H, self.weight, self.bias)


class GCNHead(nn.Module):
    """Genre detectors from a GCN over the genre graph; score z = H^(L) x.

    A_hat is normalize_adjacency(build_adjacency(genre_ids, parents)) from graph.py. Its row and
    column order must be the same genre_ids order as the columns of the label matrix Y.

    Node features start as H^0 = I (one-hot), so nothing pretrained is used. Layer widths are
    C -> 512 -> ... -> 256: num_layers=1 gives C -> 256, num_layers=3 gives C -> 512 -> 512 -> 256.
    """

    def __init__(
        self,
        A_hat: np.ndarray | torch.Tensor,
        out_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()
        A_hat = torch.as_tensor(A_hat, dtype=torch.float32)
        C = A_hat.shape[0]
        self.register_buffer("A_hat", A_hat)
        self.register_buffer("H0", torch.eye(C))
        dims = [C] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(GCNLayer(d_in, d_out) for d_in, d_out in zip(dims[:-1], dims[1:]))

    def detectors(self) -> torch.Tensor:
        """H^(L): one out_dim-d detector per genre, shape (C, out_dim). ReLU between layers only."""
        H = self.H0
        for i, layer in enumerate(self.layers):
            H = layer(H, self.A_hat)
            if i < len(self.layers) - 1:
                H = torch.relu(H)
        return H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, out_dim) audio vectors -> (batch, C) logits."""
        return x @ self.detectors().T


class FlatHead(nn.Module):
    """Control head: one independent linear detector per genre. (in_dim + 1) * C parameters."""

    def __init__(self, in_dim: int = 256, num_genres: int = 97):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_genres)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
