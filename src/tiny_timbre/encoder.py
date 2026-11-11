"""CNN encoder: log-mel spectrogram (1 x 128 x T) -> 256-d audio vector x.

Four blocks of Conv 3x3 -> BatchNorm -> ReLU -> MaxPool (32, 64, 128, 256 channels),
then global average pooling. About 0.39M parameters.

Owner: film teammate
"""
