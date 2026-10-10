"""Small CNN baseline: raw waveform -> log-mel -> 4 conv blocks -> pooled -> classifier."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..features import LogMel


class EmotionCNN(nn.Module):
    def __init__(self, n_classes: int = 7, channels=(24, 48, 96, 192), dropout: float = 0.3,
                 sample_rate: int = 16000, **feature_kwargs):
        super().__init__()
        self.frontend = LogMel(sr=sample_rate, **feature_kwargs)
        blocks, cin = [], 1
        for c in channels:
            blocks += [nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            cin = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * cin, n_classes))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.features(self.frontend(wav))      # (B, C, F', T')
        x = x.mean(dim=2)                          # collapse frequency -> (B, C, T')
        x = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)   # average + max over time
        return self.head(x)
