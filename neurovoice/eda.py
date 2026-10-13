"""Exploratory data analysis: saves figures to reports/figures/ and prints summary tables.

    python -m neurovoice.eda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data.labels import EMOTIONS

SPLIT_ORDER = ["train", "val", "test", "xcorpus_test"]


def class_balance(df: pd.DataFrame, out: Path) -> None:
    ct = pd.crosstab(df["emotion"], df["dataset"]).reindex([e for e in EMOTIONS if e in set(df.emotion)])
    ax = ct.plot(kind="bar", stacked=True, figsize=(8, 4.5), width=0.8)
    ax.set_title("Clips per emotion, by dataset")
    ax.set_xlabel("")
    ax.set_ylabel("clips")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out / "class_balance.png", dpi=150)
    plt.close()


def durations(df: pd.DataFrame, out: Path, window_s: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, g in df.groupby("dataset"):
        ax.hist(g["duration_s"], bins=30, alpha=0.6, label=f"{name} (mean {g.duration_s.mean():.1f}s)")
    ax.axvline(window_s, color="k", linestyle="--", label=f"model window ({window_s:g}s)")
    ax.set_xlabel("clip duration after trimming (s)")
    ax.set_ylabel("clips")
    ax.set_title("Clip durations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "durations.png", dpi=150)
    plt.close(fig)


def split_overview(df: pd.DataFrame, out: Path) -> None:
    splits = [s for s in SPLIT_ORDER if s in set(df.split)]
    clips = pd.crosstab(df["split"], df["emotion"]).reindex(index=splits, columns=[e for e in EMOTIONS if e in set(df.emotion)])
    spk = df.drop_duplicates("speaker_id").groupby("split").size().reindex(splits)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    clips.plot(kind="bar", stacked=True, ax=a1, width=0.8)
    a1.set_title("Clips per split, by emotion")
    a1.set_xlabel("")
    a1.tick_params(axis="x", rotation=0)
    spk.plot(kind="bar", ax=a2, color="slategray", width=0.6)
    overlap = int((df.groupby("speaker_id")["split"].nunique() > 1).sum())
    a2.set_title(f"Speakers per split (speakers in >1 split: {overlap})")
    a2.set_xlabel("")
    a2.tick_params(axis="x", rotation=0)
    for i, v in enumerate(spk):
        a2.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out / "split_overview.png", dpi=150)
    plt.close(fig)


def example_mels(df: pd.DataFrame, out: Path, sr: int = 16000) -> None:
    src = df[df.dataset == "ravdess"] if (df.dataset == "ravdess").any() else df
    emotions = [e for e in EMOTIONS if e in set(src.emotion)]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, emo in zip(axes.ravel(), emotions):
        path = src[src.emotion == emo].sort_values("uid").iloc[0]["proc_path"]
        y, _ = librosa.load(path, sr=sr)
        m = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64), ref=np.max)
        ax.axis("on")
        ax.imshow(m, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(emo)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Example log-mel spectrogram per emotion (one clip each)")
    fig.tight_layout()
    fig.savefig(out / "example_mels.png", dpi=150)
    plt.close(fig)


def summary(df: pd.DataFrame) -> str:
    parts = [
        f"Clips: {len(df)} | speakers: {df.speaker_id.nunique()} | total audio: {df.duration_s.sum() / 3600:.2f} h",
        "\nDuration (s) by dataset:",
        df.groupby("dataset")["duration_s"].describe().round(2).to_string(),
        "\nGender split (clips):",
        pd.crosstab(df["dataset"], df["gender"]).to_string(),
        "\nMissing values per column:",
        df.isna().sum().to_string(),
        f"\nDuplicate audio paths: {int(df.proc_path.duplicated().sum())}",
    ]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest_processed.csv")
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--window", type=float, default=3.0)
    a = ap.parse_args()
    df = pd.read_csv(a.manifest)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    class_balance(df, out)
    durations(df, out, a.window)
    split_overview(df, out)
    example_mels(df, out)
    text = summary(df)
    (out.parent / "eda_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nFigures saved to {out}/")


if __name__ == "__main__":
    main()
