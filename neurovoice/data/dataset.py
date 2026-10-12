"""PyTorch dataset: fixed-length windows from preprocessed clips."""
from __future__ import annotations
 
import random
 
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
 
 
class EmotionDataset(Dataset):
    """Longer clips: random crop (train) or center crop (eval). Shorter clips: zero-padded at the end.
    Train mode also adds Gaussian noise at a random SNR with probability `noise_prob`."""
 
    def __init__(self, df: pd.DataFrame, sample_rate: int = 16000, window_s: float = 3.0,
                 train: bool = False, noise_prob: float = 0.5, snr_db=(15, 40),
                 dataset_weights: dict | None = None):
        # Manifests written on Windows contain backslashes; normalize so the same
        # manifest also works on Linux (e.g. Google Colab).
        self.paths = [str(p).replace("\\", "/") for p in df["proc_path"]]
        self.labels = df["label"].astype(int).tolist()
        dw = dataset_weights or {}
        self.weights = [float(dw.get(d, 1.0)) for d in df["dataset"]]
        self.sr, self.n = sample_rate, int(round(window_s * sample_rate))
        self.train, self.noise_prob, self.snr_db = train, noise_prob, snr_db
 
    def __len__(self) -> int:
        return len(self.paths)
 
    def _fit(self, y: torch.Tensor) -> torch.Tensor:
        if len(y) >= self.n:
            start = random.randint(0, len(y) - self.n) if self.train else (len(y) - self.n) // 2
            return y[start:start + self.n]
        out = torch.zeros(self.n)
        out[:len(y)] = y
        return out
 
    def __getitem__(self, i: int):
        y, sr = sf.read(self.paths[i], dtype="float32")
        if sr != self.sr:
            raise ValueError(f"{self.paths[i]}: expected {self.sr} Hz, got {sr}. Run preprocess first.")
        y = self._fit(torch.from_numpy(y))
        if self.train and random.random() < self.noise_prob:
            snr = random.uniform(*self.snr_db)
            power = y.pow(2).mean().clamp_min(1e-8)
            y = y + torch.randn_like(y) * (power / 10 ** (snr / 10)).sqrt()
        return y, self.labels[i], self.weights[i]
 
