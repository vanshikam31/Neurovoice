"""Train and evaluate the CNN baseline.

    python -m neurovoice.train --config configs/train_cnn.yaml
    python -m neurovoice.train --config configs/train_cnn.yaml --smoke   # 1-minute sanity check
    python -m neurovoice.train --resume       # continue after Ctrl+C
    python -m neurovoice.train --eval-only    # evaluate best checkpoint without training

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
from .models.wavlm import EmotionWavLM

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
                        dataset_weights=t.get("dataset_weights") or {},
                        pad_mode=(cfg.get("data") or {}).get("pad_mode", "zeros"))
    return DataLoader(ds, batch_size=t["batch_size"], shuffle=train, num_workers=t.get("num_workers", 0))


def weighted_ce(logits, y, sample_w, class_w, smoothing: float):
    ce = F.cross_entropy(logits, y, reduction="none", label_smoothing=smoothing)
    w = class_w[y] * sample_w
    return (ce * w).sum() / w.sum().clamp_min(1e-8)


def use_amp(cfg: dict, device) -> bool:
    return bool(cfg["train"].get("amp", False)) and device.type == "cuda"


@torch.no_grad()
def predict(model, loader, device, amp: bool = False) -> np.ndarray:
    model.eval()
    out = []
    for x, _, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x.to(device))
        out.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(EMOTIONS)))


def evaluate_split(model, df: pd.DataFrame, cfg: dict, device, out_dir: Path, name: str) -> dict:
    df = df.reset_index(drop=True)
    pred = predict(model, make_loader(df, cfg, train=False), device, use_amp(cfg, device)).argmax(axis=1)
    y = df["label"].to_numpy()
    s = summarize(y, pred, EMOTIONS)
    s["by_dataset"] = breakdown(df, y, pred, "dataset", EMOTIONS)
    s["by_gender"] = breakdown(df, y, pred, "gender", EMOTIONS)
    plot_confusion(s["confusion_matrix"], EMOTIONS, out_dir / f"confusion_{name}.png",
                   f"{name}  (UAR {s['uar']:.3f}, n={s['n']})")
    return s


def build_model(cfg: dict, device) -> torch.nn.Module:
    m = cfg["model"]
    if m.get("type", "cnn") == "wavlm":
        return EmotionWavLM(n_classes=len(EMOTIONS), pretrained=m.get("pretrained"), dropout=m.get("dropout", 0.1),
                            layer_weighting=m.get("layer_weighting", True),
                            freeze_feature_encoder=m.get("freeze_feature_encoder", True),
                            layerdrop=m.get("layerdrop", 0.0), backbone_config=m.get("backbone_config")).to(device)
    return EmotionCNN(n_classes=len(EMOTIONS), channels=tuple(cfg["model"]["channels"]),
                      dropout=cfg["model"]["dropout"], sample_rate=cfg["sample_rate"],
                      **cfg["features"]).to(device)


def make_optimizer(model, cfg: dict):
    """AdamW. Pretrained backbones get a small lr and the new head a larger one (`head_lr`)."""
    t = cfg["train"]
    if hasattr(model, "backbone"):
        head_lr = t.get("head_lr", t["lr"])
        bb = [p for p in model.backbone.parameters() if p.requires_grad]
        rest = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
        groups, max_lr = [{"params": bb, "lr": t["lr"]}, {"params": rest, "lr": head_lr}], [t["lr"], head_lr]
    else:
        groups, max_lr = [{"params": [p for p in model.parameters() if p.requires_grad]}], t["lr"]
    return torch.optim.AdamW(groups, lr=t["lr"], weight_decay=t["weight_decay"]), max_lr


def final_eval(model, splits: dict, cfg: dict, device, out_dir: Path, best_epoch: int, n_params: int) -> dict:
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


def run(cfg: dict, epochs: int | None = None, smoke: bool = False,
        resume: bool = False, eval_only: bool = False) -> dict:
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

    out_dir = Path(cfg["out_dir"]) / (cfg["name"] + ("_smoke" if smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate a saved checkpoint without training (e.g. after stopping a run early).
    if eval_only:
        if not (out_dir / "best.pt").exists():
            raise FileNotFoundError(f"{out_dir / 'best.pt'} not found. Train first.")
        ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        cfg = {**cfg, **{k: ck["config"][k] for k in ("sample_rate", "window_s", "features", "model")
                                 if k in ck["config"]},
               "data": ck["config"].get("data", {})}      # old checkpoints: zero padding
        model = build_model(cfg, device)
        model.load_state_dict(ck["model"])
        log.info("Evaluating best checkpoint (epoch %d)", ck["epoch"])
        return final_eval(model, splits, cfg, device, out_dir, ck["epoch"], sum(p.numel() for p in model.parameters()))

    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("device=%s | params=%.2fM (trainable %.2fM) | train=%d val=%d | amp=%s", device, n_params / 1e6,
             n_train / 1e6, len(train_df), len(val_df), use_amp(cfg, device))

    cw = class_weights(train_df["label"].to_numpy(), len(EMOTIONS))
    log.info("class weights: %s", {e: round(float(w), 2) for e, w in zip(EMOTIONS, cw)})
    cw = cw.to(device)

    train_loader = make_loader(train_df, cfg, train=True)
    opt, max_lr = make_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_loader),
                                                pct_start=t.get("pct_start", 0.3))
    amp = use_amp(cfg, device)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    best_uar, best_epoch, history, start_epoch = -1.0, 0, [], 1
    last_path = out_dir / "last.pt"
    if resume:
        if last_path.exists():
            ck = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            if amp and ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            start_epoch, best_uar, best_epoch, history = ck["epoch"] + 1, ck["best_uar"], ck["best_epoch"], ck["history"]
            log.info("Resumed after epoch %d (best val UAR %.3f at epoch %d)", ck["epoch"], best_uar, best_epoch)
        else:
            log.warning("--resume given but %s not found; starting from scratch", last_path)

    done = start_epoch - 1
    try:
        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            model.train()
            losses = []
            for x, y, sw in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
                x, y, sw = x.to(device), y.to(device), sw.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(x)
                loss = weighted_ce(logits.float(), y, sw, cw, t.get("label_smoothing", 0.0))
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("grad_clip", 5.0))
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(loss.item())

            val_pred = predict(model, make_loader(val_df, cfg, train=False), device, amp).argmax(axis=1)
            vs = summarize(val_df["label"].to_numpy(), val_pred, EMOTIONS)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                            "val_acc": vs["accuracy"], "val_uar": vs["uar"], "val_macro_f1": vs["macro_f1"]})
            improved = vs["uar"] > best_uar
            if improved:
                best_uar, best_epoch = vs["uar"], epoch
                torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, out_dir / "best.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict() if amp else None, "epoch": epoch, "best_uar": best_uar, "best_epoch": best_epoch, "history": history}, last_path)
            pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
            done = epoch
            mem = f" | peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f}GB" if device.type == "cuda" else ""
            log.info("epoch %02d | loss %.3f | val acc %.3f UAR %.3f F1 %.3f | %.0fs%s%s", epoch,
                     history[-1]["train_loss"], vs["accuracy"], vs["uar"], vs["macro_f1"],
                     time.time() - t0, mem, "  *best*" if improved else "")
            if epoch - best_epoch >= t.get("patience", 6):
                log.info("Early stopping (no val UAR gain for %d epochs)", t.get("patience", 6))
                break
    except KeyboardInterrupt:
        log.info("\nStopped. Finished epochs are saved (last: %d, best: %d).", done, best_epoch)
        log.info("Continue training:      python -m neurovoice.train --resume")
        log.info("Evaluate best so far:   python -m neurovoice.train --eval-only")
        return {}

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=False)["model"])
    results = final_eval(model, splits, cfg, device, out_dir, best_epoch, n_params)
    if t.get("delete_last_when_done", False):
        last_path.unlink(missing_ok=True)      # resume state is large for WavLM; not needed once finished
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_cnn.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch: checks the pipeline runs")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.pt")
    ap.add_argument("--eval-only", action="store_true", help="evaluate runs/<name>/best.pt, no training")
    a = ap.parse_args()
    run(load_config(a.config), epochs=a.epochs, smoke=a.smoke, resume=a.resume, eval_only=a.eval_only)


if __name__ == "__main__":
    main()
