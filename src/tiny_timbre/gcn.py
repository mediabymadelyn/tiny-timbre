"""From-scratch GCN layer, GCN head over the genre graph, and the flat multi-label head.

GCN layer:  H^(l+1) = ReLU(A_hat H^(l) W^(l))   (no ReLU after the last layer)
GCN head:   H^0 = I (one-hot), two layers C -> 512 -> 256, scores z = H^(L) x
Flat head:  Linear 256 -> C

Every head maps the encoder output x of shape (batch, 256) to logits of shape (batch, C),
with no sigmoid: the loss (binary_cross_entropy_with_logits) applies it.

Owner: Ayesha
"""

import torch
from torch import nn


class FlatHead(nn.Module):
    """Control head: one independent linear detector per genre. (in_dim + 1) * C parameters."""

    def __init__(self, in_dim: int = 256, num_genres: int = 97):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_genres)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
