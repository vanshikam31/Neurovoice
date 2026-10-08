"""Standardize audio: mono, 16 kHz, trim silence, peak-normalize.

    python -m neurovoice.data.preprocess --config configs/data.yaml

Long clips are NOT cropped here; windowing happens at training/inference time.
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from ..config import load_config

log = logging.getLogger(__name__)


def preprocess_waveform(y: np.ndarray, top_db: float = 30, peak: float = 0.95) -> np.ndarray:
    """Trim leading/trailing silence, then peak-normalize. Returns empty array if silent."""
    if y.size == 0 or float(np.max(np.abs(y))) == 0.0:
        return y[:0]  # librosa.trim returns all-zero audio unchanged, so check first
    y, _ = librosa.effects.trim(y, top_db=top_db)
    if y.size == 0:
        return y
    return y / float(np.max(np.abs(y))) * peak


def process_file(src: str, dst: str, sr: int, top_db: float, peak: float) -> float:
    """Load -> mono/resample -> clean -> save 16-bit wav. Returns duration in seconds."""
    y, _ = librosa.load(src, sr=sr, mono=True)
    y = preprocess_waveform(y, top_db, peak)
    if y.size == 0:
        return 0.0
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, y, sr, subtype="PCM_16")
    return len(y) / sr


def _work(job: tuple) -> float:
    try:
        return process_file(*job)
    except Exception as e:  # corrupt file etc.; don't kill the whole run
        log.warning("Failed %s: %s", job[0], e)
        return 0.0


def run(cfg: dict) -> pd.DataFrame:
    pc, sr = cfg["preprocess"], cfg["sample_rate"]
    df = pd.read_csv(cfg["manifest"])
    out_dir = Path(pc["out_dir"])
    jobs = [
        (r.path, str(out_dir / r.dataset / f"{r.uid}.wav"), sr, pc["trim_top_db"], pc["peak_norm"])
        for r in df.itertuples()
    ]
    with ProcessPoolExecutor(max_workers=pc.get("workers")) as ex:
        durs = list(tqdm(ex.map(_work, jobs, chunksize=16), total=len(jobs), desc="preprocess"))

    df["proc_path"] = [j[1] for j in jobs]
    df["duration_s"] = durs
    keep = df["duration_s"] >= pc["min_duration_s"]
    log.info("Dropped %d clips (failed or shorter than %.2fs)", int((~keep).sum()), pc["min_duration_s"])
    df = df[keep].reset_index(drop=True)
    Path(cfg["manifest_processed"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["manifest_processed"], index=False)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    df = run(load_config(ap.parse_args().config))
    print(df.groupby("dataset")["duration_s"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
