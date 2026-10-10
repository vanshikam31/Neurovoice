"""Log-mel front end as a torch module, so features are computed on the fly
(and on GPU if available) and the model takes raw waveforms as input."""
from __future__ import annotations

import librosa
import torch
import torch.nn as nn


class LogMel(nn.Module):
    def __init__(self, sr: int = 16000, n_fft: int = 1024, hop: int = 256, n_mels: int = 64,
                 fmin: float = 50, fmax: float = 8000, spec_augment: bool = True,
                 freq_mask: int = 8, time_mask: int = 24, n_masks: int = 2):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.spec_augment, self.freq_mask, self.time_mask, self.n_masks = spec_augment, freq_mask, time_mask, n_masks
        fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.register_buffer("fb", torch.from_numpy(fb).float())          # (n_mels, n_fft//2+1)
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, samples) -> (B, 1, n_mels, frames), instance-normalized."""
        spec = torch.stft(wav, self.n_fft, self.hop, window=self.window,
                          return_complex=True, center=True)
        mel = torch.matmul(self.fb, spec.abs().pow(2))                    # (B, n_mels, T)
        x = torch.log(mel + 1e-6)
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-5)
        if self.training and self.spec_augment:
            x = self._mask(x)
        return x.unsqueeze(1)

    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        _, n_mels, n_frames = x.shape
        for b in range(x.shape[0]):
            for _ in range(self.n_masks):
                w = int(torch.randint(0, self.freq_mask + 1, (1,)))
                f0 = int(torch.randint(0, max(1, n_mels - w), (1,)))
                x[b, f0:f0 + w, :] = 0
                w = int(torch.randint(0, self.time_mask + 1, (1,)))
                t0 = int(torch.randint(0, max(1, n_frames - w), (1,)))
                x[b, :, t0:t0 + w] = 0
        return x
