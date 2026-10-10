"""Train and evaluate the CNN baseline.

    python -m neurovoice.train --config configs/train_cnn.yaml
    python -m neurovoice.train --config configs/train_cnn.yaml --smoke   # 1-minute sanity check

Model selection uses validation UAR only. Test and cross-corpus sets are touched once, at the end.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .data.dataset import EmotionDataset
from .data.labels import EMOTIONS
from .metrics import breakdown, plot_confusion, summarize
from .models.cnn import EmotionCNN

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(labels, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights (mean 1 over present classes) so small classes like 'surprised' count more."""
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    present = counts > 0
    w = np.zeros(n_classes)
    w[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(w, dtype=torch.float32)


def make_loader(df: pd.DataFrame, cfg: dict, train: bool) -> DataLoader:
    t = cfg["train"]
    ds = EmotionDataset(df, cfg["sample_rate"], cfg["window_s"], train=train,
                        noise_prob=t.get("noise_prob", 0.5), snr_db=tuple(t.get("snr_db", (15, 40))),
                        dataset_weights=t.get("dataset_weights") or {})
    return DataLoader(ds, batch_size=t["batch_size"], shuffle=train, num_workers=t.get("num_workers", 0))


def weighted_ce(logits, y, sample_w, class_w, smoothing: float):
    ce = F.cross_entropy(logits, y, reduction="none", label_smoothing=smoothing)
    w = class_w[y] * sample_w
    return (ce * w).sum() / w.sum().clamp_min(1e-8)


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = [F.softmax(model(x.to(device)), dim=1).cpu().numpy() for x, _, _ in loader]
    return np.concatenate(out) if out else np.zeros((0, len(EMOTIONS)))


def evaluate_split(model, df: pd.DataFrame, cfg: dict, device, out_dir: Path, name: str) -> dict:
    df = df.reset_index(drop=True)
    pred = predict(model, make_loader(df, cfg, train=False), device).argmax(axis=1)
    y = df["label"].to_numpy()
    s = summarize(y, pred, EMOTIONS)
    s["by_dataset"] = breakdown(df, y, pred, "dataset", EMOTIONS)
    s["by_gender"] = breakdown(df, y, pred, "gender", EMOTIONS)
    plot_confusion(s["confusion_matrix"], EMOTIONS, out_dir / f"confusion_{name}.png",
                   f"{name}  (UAR {s['uar']:.3f}, n={s['n']})")
    return s


def run(cfg: dict, epochs: int | None = None, smoke: bool = False) -> dict:
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = cfg["train"]
    epochs = epochs or t["epochs"]

    df = pd.read_csv(cfg["manifest"])
    splits = {s: g.reset_index(drop=True) for s, g in df.groupby("split")}
    if smoke:
        splits = {s: g.sample(min(len(g), 96), random_state=0).reset_index(drop=True) for s, g in splits.items()}
        epochs = 1
    train_df, val_df = splits["train"], splits["val"]

    out_dir = Path(cfg["out_dir"]) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    model = EmotionCNN(n_classes=len(EMOTIONS), channels=tuple(cfg["model"]["channels"]),
                       dropout=cfg["model"]["dropout"], sample_rate=cfg["sample_rate"],
                       **cfg["features"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("device=%s | params=%.2fM | train=%d val=%d", device, n_params / 1e6, len(train_df), len(val_df))

    cw = class_weights(train_df["label"].to_numpy(), len(EMOTIONS))
    log.info("class weights: %s", {e: round(float(w), 2) for e, w in zip(EMOTIONS, cw)})
    cw = cw.to(device)

    train_loader = make_loader(train_df, cfg, train=True)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t["lr"], epochs=epochs, steps_per_epoch=len(train_loader))

    best_uar, best_epoch, history = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for x, y, sw in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x, y, sw = x.to(device), y.to(device), sw.to(device)
            loss = weighted_ce(model(x), y, sw, cw, t.get("label_smoothing", 0.0))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        val_pred = predict(model, make_loader(val_df, cfg, train=False), device).argmax(axis=1)
        vs = summarize(val_df["label"].to_numpy(), val_pred, EMOTIONS)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_acc": vs["accuracy"], "val_uar": vs["uar"], "val_macro_f1": vs["macro_f1"]})
        improved = vs["uar"] > best_uar
        if improved:
            best_uar, best_epoch = vs["uar"], epoch
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, out_dir / "best.pt")
        log.info("epoch %02d | loss %.3f | val acc %.3f UAR %.3f F1 %.3f | %.0fs%s", epoch,
                 history[-1]["train_loss"], vs["accuracy"], vs["uar"], vs["macro_f1"],
                 time.time() - t0, "  *best*" if improved else "")
        if epoch - best_epoch >= t.get("patience", 6):
            log.info("Early stopping (no val UAR gain for %d epochs)", t.get("patience", 6))
            break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # Final evaluation with the best checkpoint
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["model"])
    results = {"best_epoch": best_epoch, "params": n_params}
    for name in ("val", "test", "xcorpus_test"):
        if name in splits and len(splits[name]):
            results[name] = evaluate_split(model, splits[name], cfg, device, out_dir, name)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    lines = [f"\nBest epoch: {best_epoch}", f"{'split':<14}{'n':>6}{'acc':>8}{'UAR':>8}{'macroF1':>9}"]
    for name in ("val", "test", "xcorpus_test"):
        if name in results:
            r = results[name]
            lines.append(f"{name:<14}{r['n']:>6}{r['accuracy']:>8.3f}{r['uar']:>8.3f}{r['macro_f1']:>9.3f}")
    log.info("\n".join(lines))
    log.info("Saved results to %s", out_dir)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_cnn.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch: checks the pipeline runs")
    a = ap.parse_args()
    run(load_config(a.config), epochs=a.epochs, smoke=a.smoke)


if __name__ == "__main__":
    main()
