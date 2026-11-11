"""CNN encoder: log-mel spectrogram (1 x 128 x T) -> 256-d audio vector x.

Four blocks of Conv 3x3 -> BatchNorm -> ReLU -> MaxPool (32, 64, 128, 256 channels),
then global average pooling. 388,800 parameters.

Owner: Maddi
"""

import torch
from torch import nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size= 3, padding = 1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
        
    )
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
        conv_block(1, 32),
        conv_block(32, 64),
        conv_block(64, 128),
        conv_block(128, 256)
    )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            x=self.blocks(x)
            #avg over last two dimensions (height and time)
            x = torch.mean(x, dim=(-2,-1))
            return x

