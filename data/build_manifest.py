"""Scan raw datasets -> unified manifest CSV with speaker-independent splits.

    python -m neurovoice.data.build_manifest --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..config import load_config
from .labels import EMOTIONS, LABEL2ID
from .parsers import PARSERS
from .splits import assign_splits, verify_no_leakage

log = logging.getLogger(__name__)

EXPECTED_SPEAKERS = {"ravdess": 24, "cremad": 91, "tess": 2, "savee": 4}
COLUMNS = ["uid", "path", "dataset", "speaker_id", "gender",
           "emotion_raw", "emotion", "label", "split"]


def summarize(df: pd.DataFrame) -> str:
    order = [e for e in EMOTIONS if e in set(df.emotion)]
    parts = [
        "# NeuroVoice data report\n",
        f"Total clips: {len(df)}  |  Speakers: {df.speaker_id.nunique()}\n",
        "## Clips per dataset x emotion\n```",
        pd.crosstab(df.dataset, df.emotion)[order].to_string(),
        "```\n## Clips per split x emotion\n```",
        pd.crosstab(df.split, df.emotion)[order].to_string(),
        "```\n## Speakers per split x dataset\n```",
        df.drop_duplicates("speaker_id").pipe(lambda d: pd.crosstab(d.split, d.dataset)).to_string(),
        "```\n## Clips per split x gender\n```",
        pd.crosstab(df.split, df.gender).to_string(),
        "```",
    ]
    return "\n".join(parts) + "\n"


def build_manifest(cfg: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for name, root in cfg["datasets"].items():
        root = Path(root)
        if not root.exists():
            log.warning("[skip] %s: %s not found", name, root)
            continue
        recs = PARSERS[name](root)
        log.info("[ok]   %s: %d clips", name, len(recs))
        rows.extend(recs)
    if not rows:
        raise SystemExit("No audio found. Check `datasets:` paths in the config.")

    df = pd.DataFrame(rows)

    # Some mirrors (e.g. Kaggle TESS/RAVDESS) contain the whole dataset twice in nested
    # folders. Filenames are unique within each real dataset, so drop repeats.
    before = len(df)
    df = df.drop_duplicates(subset=["dataset", "stem"], keep="first").copy()
    if len(df) < before:
        log.warning("Removed %d duplicate files (same dataset + filename), "
                    "probably a nested copy of the dataset folder", before - len(df))

    for name, n in df.groupby("dataset")["speaker_id"].nunique().items():
        exp = EXPECTED_SPEAKERS.get(name)
        if exp is not None and n != exp:
            log.warning("%s: found %d speakers, expected %d. Check filenames/folders.", name, n, exp)

    label_map = cfg.get("label_map") or {}
    df["emotion"] = [label_map[e] if e in label_map else e for e in df["emotion_raw"]]
    dropped = df["emotion"].isna().sum()
    if dropped:
        log.info("Dropping %d clips mapped to null", dropped)
    df = df[df["emotion"].notna()].copy()
    unknown = set(df["emotion"]) - set(EMOTIONS)
    if unknown:
        raise ValueError(f"Labels not in unified scheme: {unknown}")
    df["label"] = df["emotion"].map(LABEL2ID)

    # Unique, filesystem-safe id (also used as processed filename)
    df["uid"] = df["dataset"] + "_" + df["stem"]
    dup = df.groupby("uid").cumcount()
    df.loc[dup > 0, "uid"] += "_dup" + dup[dup > 0].astype(str)

    sp = cfg.get("split", {})
    df["split"] = assign_splits(
        df,
        val_frac=sp.get("val_frac", 0.15),
        test_frac=sp.get("test_frac", 0.15),
        holdout_datasets=sp.get("cross_corpus_holdout", []),
        seed=cfg.get("seed", 42),
    )
    verify_no_leakage(df)
    return df[COLUMNS].reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    cfg = load_config(ap.parse_args().config)

    df = build_manifest(cfg)
    Path(cfg["manifest"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["manifest"], index=False)

    report = summarize(df)
    Path(cfg["report"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["report"]).write_text(report, encoding="utf-8")
    print(report)
    log.info("Wrote %s and %s", cfg["manifest"], cfg["report"])


if __name__ == "__main__":
    main()
