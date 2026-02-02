import torch
from torch import nn


class Wan22DualModel(nn.Module):
    def __init__(self, low_noise_model: nn.Module, high_noise_model: nn.Module, boundary: float = 0.9):
        super().__init__()
        self.low_noise_model = low_noise_model
        self.high_noise_model = high_noise_model
        self.boundary = boundary
        self.config = getattr(low_noise_model, "config", None)
        self._no_split_modules = getattr(low_noise_model, "_no_split_modules", None)

    def _normalize_timestep(self, t):
        if t is None:
            return None
        if torch.is_tensor(t):
            t_val = t.detach().flatten()[0].float().item()
        else:
            t_val = float(t)
        if t_val > 1.0:
            t_val = t_val / 1000.0
        return t_val

    def _select_model(self, t):
        t_val = self._normalize_timestep(t)
        if t_val is None:
            return self.low_noise_model
        if t_val >= self.boundary:
            return self.high_noise_model
        return self.low_noise_model

    def forward(self, *args, **kwargs):
        t = kwargs.get("t")
        if t is None and len(args) > 1:
            t = args[1]
        model = self._select_model(t)
        return model(*args, **kwargs)
