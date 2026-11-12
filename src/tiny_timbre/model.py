"""Encoder + head (flat or GCN) wired together into one model.

Owner: Maddi
"""
import torch
from torch import nn

from tiny_timbre.encoder import Encoder


class Model(nn.Module):
    def __init__(self, head: nn.Module):
        super().__init__()
        self.encoder = Encoder()
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, 128, T) -> (batch, C) genre scores (logits, before sigmoid)."""
        x = self.encoder(x)
        x = self.head(x)
        return x