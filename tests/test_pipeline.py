"""End-to-end smoke test on tiny synthetic datasets (no real data needed)."""
import numpy as np
import pandas as pd
import soundfile as sf

from neurovoice.data.build_manifest import build_manifest
from neurovoice.data.preprocess import run as run_preprocess


def tone(path, sr=22050):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(sr) / sr
    y = np.concatenate([np.zeros(sr // 4), 0.3 * np.sin(2 * np.pi * 300 * t), np.zeros(sr // 4)])
    sf.write(path, y, sr)


def test_full_pipeline(tmp_path):
    raw = tmp_path / "raw"
    for actor in range(1, 9):
        for emo in ("01", "02", "05"):
            tone(raw / "rav" / f"Actor_{actor:02d}" / f"03-01-{emo}-01-01-01-{actor:02d}.wav")
    for actor in (1001, 1002, 1003, 1004, 1005, 1006):
        for emo in ("ANG", "HAP"):
            tone(raw / "cre" / f"{actor}_DFA_{emo}_XX.wav")
    for spk in ("OAF", "YAF"):
        tone(raw / "tess" / f"{spk}_back_ps.wav")
    for spk in ("DC", "JE"):
        tone(raw / "sav" / f"{spk}_su01.wav")

    cfg = {
        "seed": 0, "sample_rate": 16000,
        "datasets": {"ravdess": str(raw / "rav"), "cremad": str(raw / "cre"),
                     "tess": str(raw / "tess"), "savee": str(raw / "sav")},
        "label_map": {"calm": "neutral"},
        "split": {"val_frac": 0.15, "test_frac": 0.15, "cross_corpus_holdout": ["savee"]},
        "preprocess": {"trim_top_db": 30, "peak_norm": 0.95, "min_duration_s": 0.5,
                       "out_dir": str(tmp_path / "proc"), "workers": 2},
        "manifest": str(tmp_path / "manifest.csv"),
        "manifest_processed": str(tmp_path / "manifest_proc.csv"),
    }

    df = build_manifest(cfg)
    df.to_csv(cfg["manifest"], index=False)
    assert "calm" not in set(df.emotion)               # merged into neutral
    assert set(df[df.dataset == "savee"].split) == {"xcorpus_test"}
    assert set(df.emotion) <= {"neutral", "happy", "angry", "surprised"}

    out = run_preprocess(cfg)
    assert len(out) == len(df)
    info = sf.info(out.proc_path.iloc[0])
    assert info.samplerate == 16000 and info.channels == 1
    assert out.duration_s.between(0.9, 1.3).all()
    assert pd.read_csv(cfg["manifest_processed"]).shape[0] == len(out)
