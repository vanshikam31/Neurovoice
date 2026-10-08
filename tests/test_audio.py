import numpy as np

from neurovoice.data.preprocess import preprocess_waveform

SR = 16000


def test_trims_silence_and_normalizes():
    t = np.arange(SR) / SR
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    y = np.concatenate([np.zeros(SR // 2), tone, np.zeros(SR // 2)]).astype(np.float32)
    out = preprocess_waveform(y, top_db=30, peak=0.95)
    assert abs(len(out) / SR - 1.0) < 0.25  # trim keeps up to ~1 analysis frame of padding
    assert np.isclose(np.max(np.abs(out)), 0.95, atol=1e-4)


def test_silent_input_returns_empty():
    assert preprocess_waveform(np.zeros(SR, dtype=np.float32)).size == 0
