"""Speaker-independent splitting.

Splitting by clip would leak speaker identity (same voice in train and test) and
inflate accuracy. Here every speaker lives in exactly one split.

Splits produced:
  train / val / test : speakers divided within each (dataset, gender) group
  xcorpus_test       : whole datasets that are never used for training
Groups with fewer than 3 speakers (e.g. TESS has 2) go entirely to train.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_splits(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    holdout_datasets: list[str] | None = None,
    seed: int = 42,
) -> pd.Series:
    holdout = set(holdout_datasets or [])
    rng = np.random.default_rng(seed)
    speakers = df[["speaker_id", "dataset", "gender"]].drop_duplicates("speaker_id")
    mapping: dict[str, str] = {}

    for (dataset, _gender), grp in speakers.groupby(["dataset", "gender"]):
        ids = sorted(grp["speaker_id"])
        if dataset in holdout:
            mapping.update({s: "xcorpus_test" for s in ids})
            continue
        n = len(ids)
        if n < 3:
            mapping.update({s: "train" for s in ids})
            continue
        rng.shuffle(ids)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, min(round(n * val_frac), n - n_test - 1))
        mapping.update({s: "test" for s in ids[:n_test]})
        mapping.update({s: "val" for s in ids[n_test:n_test + n_val]})
        mapping.update({s: "train" for s in ids[n_test + n_val:]})

    return df["speaker_id"].map(mapping)


def verify_no_leakage(df: pd.DataFrame) -> None:
    per_speaker = df.groupby("speaker_id")["split"].nunique()
    leaked = per_speaker[per_speaker > 1]
    if len(leaked):
        raise AssertionError(f"Speakers in multiple splits: {list(leaked.index)}")
    if df["split"].isna().any():
        raise AssertionError("Some rows have no split assigned")
    train_spk = set(df.loc[df.split == "train", "speaker_id"])
    for s in ("val", "test", "xcorpus_test"):
        overlap = train_spk & set(df.loc[df.split == s, "speaker_id"])
        if overlap:
            raise AssertionError(f"Train/{s} speaker overlap: {sorted(overlap)}")
