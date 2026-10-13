"""Run config variants x seeds and summarize mean/std, to separate real effects from run-to-run noise.

    python -m neurovoice.experiments --config configs/train_cnn.yaml --seeds 0 1 2
    python -m neurovoice.experiments --variants baseline repeat_pad --seeds 0

Finished runs (those with a metrics.json) are skipped, so you can stop and re-run after a disconnect.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data.labels import LABEL2ID
from .train import run

log = logging.getLogger(__name__)

VARIANTS = {
    "baseline": {},
    "repeat_pad": {"data": {"pad_mode": "repeat"}},
    "band_norm": {"features": {"norm": "band"}},
    "repeat+band": {"data": {"pad_mode": "repeat"}, "features": {"norm": "band"}},
    # optional extras (not in the default set)
    "baseline_50ep": {"train": {"epochs": 50}},
    "wide": {"model": {"channels": [48, 96, 192, 384]}},
    "tess_half": {"train": {"dataset_weights": {"tess": 0.5}}},
}
DEFAULT_VARIANTS = ["baseline", "repeat_pad", "band_norm", "repeat+band"]
METRICS = ["val_uar", "test_uar", "xcorpus_uar", "xcorpus_disgust_share"]


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _row(variant: str, seed: int, res: dict) -> dict:
    x = res.get("xcorpus_test")
    share = np.nan
    if x:
        cm = np.asarray(x["confusion_matrix"])
        share = float(cm[:, LABEL2ID["disgust"]].sum() / max(cm.sum(), 1))   # how often SAVEE is called 'disgust'
    return {"variant": variant, "seed": seed, "best_epoch": res["best_epoch"],
            "val_uar": res["val"]["uar"], "test_uar": res["test"]["uar"],
            "xcorpus_uar": x["uar"] if x else np.nan, "xcorpus_disgust_share": share}


def run_grid(base_cfg: dict, variants: list[str], seeds: list[int], epochs: int | None = None) -> pd.DataFrame:
    out_root, base_name = Path(base_cfg["out_dir"]), base_cfg["name"]
    rows = []
    for variant in variants:
        for seed in seeds:
            cfg = deep_update(base_cfg, VARIANTS[variant])
            cfg["seed"], cfg["name"] = seed, f"{base_name}_{variant}_s{seed}"
            metrics_path = out_root / cfg["name"] / "metrics.json"
            if metrics_path.exists():
                log.info("[skip] %s already finished", cfg["name"])
                res = json.loads(metrics_path.read_text())
            else:
                log.info("[run ] %s", cfg["name"])
                res = run(cfg, epochs=epochs)
                if not res:          # interrupted with Ctrl+C
                    log.info("Interrupted; re-run the same command to continue.")
                    return pd.DataFrame(rows)
            rows.append(_row(variant, seed, res))

    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / f"{base_name}_ablation_runs.csv", index=False)
    summary = df.groupby("variant", sort=False)[METRICS].agg(["mean", "std"]).round(3)
    summary.to_csv(out_root / f"{base_name}_ablation_summary.csv")
    log.info("\n%s\n(std over %d seed(s); disgust_share = fraction of SAVEE clips predicted 'disgust', chance ~0.14)",
             summary.to_string(), len(seeds))
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_cnn.yaml")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    run_grid(load_config(a.config), a.variants, a.seeds, a.epochs)


if __name__ == "__main__":
    main()
