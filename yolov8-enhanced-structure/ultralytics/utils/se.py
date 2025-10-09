# se.py
import torch
import torch.nn as nn

class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block.
    Usage from YAML: SE, [channels]
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w
