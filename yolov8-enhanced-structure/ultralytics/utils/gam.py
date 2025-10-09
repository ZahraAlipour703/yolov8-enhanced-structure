# gam.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GAM(nn.Module):
    """Compact Global Attention Module.
    Simple, effective channel + spatial attention combination used in several YOLO forks.
    Usage from YAML: GAM, [channels]
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        # channel attention path
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # spatial attention path (single-channel map)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # channel attention
        ca = self.channel_mlp(x)            # B, C, 1, 1
        x = x * ca
        # spatial attention using channel-averaged map
        sa = torch.mean(x, dim=1, keepdim=True)  # B, 1, H, W
        sa = self.spatial_conv(sa)
        x = x * sa
        return x
