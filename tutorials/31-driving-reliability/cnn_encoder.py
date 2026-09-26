"""Compact convolutional encoders shared by visual-driving agents."""

from __future__ import annotations

import torch
import torch.nn as nn


class NatureEncoder(nn.Module):
    def __init__(self, input_channels: int, image_size: int = 64, feature_size: int = 256) -> None:
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            flattened = self.convolutions(
                torch.zeros(1, input_channels, image_size, image_size)
            ).numel()
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(flattened, feature_size), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convolutions(observations))
