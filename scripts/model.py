"""Baseline CNN: (55, 223, 291) input maps -> 7 Tufts targets.

The final feature map is flattened (not globally pooled) so the dense head knows where each feature is
relative to Tufts, which sits at a fixed position in the grid.
"""
from __future__ import annotations

import torch
from torch import nn


def block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class BaselineCNN(nn.Module):
    def __init__(self, in_channels: int = 55, n_outputs: int = 7, grid: tuple[int, int] = (223, 291),
                 widths: tuple[int, ...] = (32, 64, 128, 128, 128), hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, widths[0], 3, padding=1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        chans = [widths[0], *widths]
        self.features = nn.Sequential(*[block(a, b) for a, b in zip(chans[:-1], chans[1:])])
        with torch.no_grad():
            n_flat = self.features(self.stem(torch.zeros(1, in_channels, *grid))).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(n_flat, hidden), nn.ReLU(inplace=True),
                                  nn.Linear(hidden, n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 7): 6 normalized continuous targets and the rain_binary logit."""
        return self.head(self.features(self.stem(x)))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
