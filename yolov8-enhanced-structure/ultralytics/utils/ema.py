# ema.py
import math
import torch
from copy import deepcopy

class EMA:
    """Exponential Moving Average (EMA) for model weights.
    Typical usage:
        ema = EMA(model, decay=0.9999, device='cuda')
        # each training step after optimizer.step():
        ema.update(model)
        # to save ema weights: ema.ema.state_dict()
    """
    def __init__(self, model, decay=0.9999, device=None):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = float(decay)
        self.device = device
        if device:
            self.ema.to(device)
        self.num_updates = 0

    def update(self, model):
        """Update EMA parameters from the *source* model (model)"""
        self.num_updates += 1
        d = self.decay
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in esd.items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].to(v.device), alpha=1.0 - d)

    def state_dict(self):
        return self.ema.state_dict()

    def to(self, device):
        self.ema.to(device)
        self.device = device
