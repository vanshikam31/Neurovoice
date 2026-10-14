"""WavLM backbone -> learnable weighted sum of all layers -> mean pooling over time -> linear head.

Takes raw 16 kHz waveforms (B, samples), like the CNN, so both models plug into the same training code.
`transformers` is imported lazily, so the CNN path works without it installed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EmotionWavLM(nn.Module):
    def __init__(self, n_classes: int = 7, pretrained: str | None = "microsoft/wavlm-base-plus",
                 dropout: float = 0.1, layer_weighting: bool = True, freeze_feature_encoder: bool = True,
                 layerdrop: float | None = 0.0, backbone_config: dict | None = None):
        super().__init__()
        from transformers import WavLMConfig, WavLMModel

        if pretrained:
            self.backbone = WavLMModel.from_pretrained(pretrained)
        else:   # random init: a "no pretraining" control experiment, or a tiny model for tests
            self.backbone = WavLMModel(WavLMConfig(**(backbone_config or {})))
        if layerdrop is not None:
            self.backbone.config.layerdrop = layerdrop
        if freeze_feature_encoder:       # the small conv front end is normally kept frozen
            self.backbone.freeze_feature_encoder()
        n_layers = self.backbone.config.num_hidden_layers + 1       # + the embedding output
        self.layer_logits = nn.Parameter(torch.zeros(n_layers)) if layer_weighting else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.config.hidden_size, n_classes))

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav = (wav - wav.mean(dim=1, keepdim=True)) / (wav.std(dim=1, keepdim=True) + 1e-5)
        out = self.backbone(wav, output_hidden_states=True)
        if self.layer_logits is not None:
            hs = torch.stack(out.hidden_states, dim=0)                          # (L, B, T, H)
            x = (hs * self.layer_weights().to(hs.dtype).view(-1, 1, 1, 1)).sum(dim=0)
        else:
            x = out.last_hidden_state
        return self.head(x.mean(dim=1))
