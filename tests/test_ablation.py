import sys
from pathlib import Path

import pandas as pd
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_train_smoke import make_manifest  # noqa: E402

import neurovoice.experiments as E  # noqa: E402
from neurovoice.data.dataset import EmotionDataset  # noqa: E402
from neurovoice.features import LogMel  # noqa: E402


def make_cfg(tmp_path, epochs=1):
    return {"name": "t", "seed": 0, "manifest": str(make_manifest(tmp_path)), "sample_rate": 16000,
            "window_s": 2.0,
            "features": {"n_fft": 512, "hop": 256, "n_mels": 32, "fmin": 50, "fmax": 8000, "spec_augment": True},
            "model": {"channels": [8, 16, 16, 16], "dropout": 0.1},
            "train": {"epochs": epochs, "batch_size": 16, "lr": 0.005, "weight_decay": 0.0,
                      "label_smoothing": 0.0, "patience": 5, "num_workers": 0, "noise_prob": 0.5,
                      "snr_db": [15, 40], "dataset_weights": {}},
            "out_dir": str(tmp_path / "runs")}


def _ds(tmp_path, pad_mode, train=False):
    t = torch.arange(16000) / 16000
    sf.write(tmp_path / "s.wav", (0.3 * torch.sin(2 * torch.pi * 300 * t)).numpy(), 16000)
    df = pd.DataFrame({"proc_path": [str(tmp_path / "s.wav")], "label": [0], "dataset": ["tess"]})
    return EmotionDataset(df, window_s=3.0, train=train, pad_mode=pad_mode)


def test_repeat_padding_loops_instead_of_silence(tmp_path):
    zeros, _, _ = _ds(tmp_path, "zeros")[0]
    rep, _, _ = _ds(tmp_path, "repeat")[0]
    assert zeros[-8000:].abs().sum() == 0            # silent tail
    assert rep.shape == (48000,) and rep[-8000:].abs().sum() > 0
    assert _ds(tmp_path, "repeat", train=True)[0][0].shape == (48000,)
    with pytest.raises(ValueError):
        _ds(tmp_path, "bogus")


def test_band_norm_standardizes_each_band():
    x = LogMel(norm="band", spec_augment=False)(torch.randn(2, 48000))[:, 0]
    assert x.mean(dim=2).abs().max() < 1e-4
    assert (x.std(dim=2) - 1).abs().max() < 1e-3
    y = LogMel(norm="instance", spec_augment=False)(torch.randn(2, 48000))[:, 0]
    assert y.mean(dim=(1, 2)).abs().max() < 1e-4
    with pytest.raises(ValueError):
        LogMel(norm="nope")


def test_deep_update_merges_nested_dicts():
    out = E.deep_update({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 9}, "c": {"z": 3}})
    assert out == {"a": {"x": 1, "y": 9}, "b": 1, "c": {"z": 3}}


def test_grid_runs_summarizes_and_skips_finished(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, epochs=1)
    df = E.run_grid(cfg, ["baseline", "repeat_pad"], [0], epochs=1)
    out = Path(cfg["out_dir"])
    assert len(df) == 2 and set(E.METRICS) <= set(df.columns)
    assert (out / "t_ablation_runs.csv").exists() and (out / "t_ablation_summary.csv").exists()
    assert df["xcorpus_disgust_share"].between(0, 1).all()

    def boom(*a, **k):
        raise AssertionError("should not retrain a finished run")
    monkeypatch.setattr(E, "run", boom)
    assert len(E.run_grid(cfg, ["baseline", "repeat_pad"], [0], epochs=1)) == 2
