"""Tiny end-to-end training run on synthetic tones (each class = a different pitch)."""
import json

import numpy as np
import pandas as pd
import soundfile as sf

from neurovoice.train import class_weights, run


def make_manifest(tmp_path):
    rows, rng = [], np.random.default_rng(0)
    for split, n_per_class in (("train", 6), ("val", 2), ("test", 2), ("xcorpus_test", 2)):
        for label in range(7):
            for k in range(n_per_class):
                t = np.arange(int(16000 * rng.uniform(1.0, 2.0))) / 16000
                y = 0.3 * np.sin(2 * np.pi * (200 + 150 * label) * t) + 0.01 * rng.standard_normal(len(t))
                path = tmp_path / f"{split}_{label}_{k}.wav"
                sf.write(path, y.astype(np.float32), 16000)
                rows.append({"proc_path": str(path), "label": label, "split": split,
                             "dataset": "savee" if split == "xcorpus_test" else "ravdess",
                             "gender": "male" if k % 2 else "female", "speaker_id": f"{split}_{k}"})
    pd.DataFrame(rows).to_csv(tmp_path / "m.csv", index=False)
    return tmp_path / "m.csv"


def test_class_weights_favor_rare_classes():
    w = class_weights([0] * 90 + [1] * 10, 3)
    assert w[1] > w[0] and w[2] == 0


def make_cfg(tmp_path, epochs=3):
    return {"name": "t", "seed": 0, "manifest": str(make_manifest(tmp_path)), "sample_rate": 16000,
            "window_s": 2.0,
            "features": {"n_fft": 512, "hop": 256, "n_mels": 32, "fmin": 50, "fmax": 8000, "spec_augment": True},
            "model": {"channels": [8, 16, 16, 16], "dropout": 0.1},
            "train": {"epochs": epochs, "batch_size": 16, "lr": 0.005, "weight_decay": 0.0,
                      "label_smoothing": 0.0, "patience": 5, "num_workers": 0, "noise_prob": 0.5,
                      "snr_db": [15, 40], "dataset_weights": {}},
            "out_dir": str(tmp_path / "runs")}


def test_training_runs_and_writes_results(tmp_path):
    res = run(make_cfg(tmp_path))
    out = tmp_path / "runs" / "t"
    for f in ("best.pt", "last.pt", "history.csv", "metrics.json", "confusion_test.png", "confusion_xcorpus_test.png"):
        assert (out / f).exists(), f
    m = json.loads((out / "metrics.json").read_text())
    assert set(m) >= {"val", "test", "xcorpus_test"}
    assert 0.0 <= m["test"]["uar"] <= 1.0 and "by_gender" in m["test"]
    assert res["val"]["n"] == 14


def test_eval_only_reproduces_results_from_checkpoint(tmp_path):
    cfg = make_cfg(tmp_path)
    first = run(cfg)
    (tmp_path / "runs" / "t" / "metrics.json").unlink()
    again = run(cfg, eval_only=True)
    assert (tmp_path / "runs" / "t" / "metrics.json").exists()
    assert again["test"]["uar"] == first["test"]["uar"]


def test_ctrl_c_then_resume_completes_schedule(tmp_path, monkeypatch):
    import neurovoice.train as T
    cfg, real, calls = make_cfg(tmp_path), T.weighted_ce, {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 5:          # 3 batches per epoch -> interrupt during epoch 2
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(T, "weighted_ce", flaky)
    assert T.run(cfg) == {}
    out = tmp_path / "runs" / "t"
    assert (out / "last.pt").exists() and len(pd.read_csv(out / "history.csv")) == 1

    monkeypatch.setattr(T, "weighted_ce", real)
    res = T.run(cfg, resume=True)
    assert len(pd.read_csv(out / "history.csv")) == 3      # epochs 1 (before) + 2, 3 (after resume)
    assert "test" in res
