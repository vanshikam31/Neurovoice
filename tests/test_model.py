import pandas as pd
import soundfile as sf
import torch

from neurovoice.data.dataset import EmotionDataset
from neurovoice.features import LogMel
from neurovoice.models.cnn import EmotionCNN


def test_logmel_shape():
    x = LogMel(spec_augment=False)(torch.randn(2, 48000))
    assert x.shape == (2, 1, 64, 188)
    assert torch.isfinite(x).all()


def test_model_forward_backward():
    m = EmotionCNN(n_classes=7, channels=(8, 16, 32, 32))
    out = m(torch.randn(3, 48000))
    assert out.shape == (3, 7)
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_dataset_crops_and_pads(tmp_path):
    for name, secs in (("short", 1.0), ("long", 5.0)):
        sf.write(tmp_path / f"{name}.wav", torch.randn(int(16000 * secs)).numpy() * 0.1, 16000)
    df = pd.DataFrame({"proc_path": [str(tmp_path / "short.wav"), str(tmp_path / "long.wav")],
                       "label": [0, 1], "dataset": ["tess", "cremad"]})
    ds = EmotionDataset(df, window_s=3.0, train=False, dataset_weights={"tess": 0.5})
    short, _, w_short = ds[0]
    long, _, w_long = ds[1]
    assert short.shape == long.shape == (48000,)
    assert short[16000:].abs().sum() == 0          # zero-padded tail
    assert (w_short, w_long) == (0.5, 1.0)


def test_windows_style_paths_are_accepted(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(wav, torch.randn(16000).numpy() * 0.1, 16000)
    df = pd.DataFrame({"proc_path": [str(wav).replace("/", "\\")], "label": [0], "dataset": ["tess"]})
    y, label, _ = EmotionDataset(df, window_s=1.0)[0]
    assert y.shape == (16000,) and label == 0
