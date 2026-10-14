import json
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).parent))
from test_train_smoke import make_manifest  # noqa: E402

from neurovoice.models.wavlm import EmotionWavLM  # noqa: E402
from neurovoice.train import make_optimizer, run  # noqa: E402

TINY = {"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 64,
        "conv_dim": [16] * 7, "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 2,
        "num_buckets": 16, "max_bucket_distance": 64}


def tiny_model(**kw):
    return EmotionWavLM(n_classes=7, pretrained=None, backbone_config=TINY, **kw)


def test_forward_backward_and_freezing():
    m = tiny_model()
    m.train()
    out = m(torch.randn(3, 32000))
    assert out.shape == (3, 7)
    out.sum().backward()
    assert m.layer_logits.grad is not None and m.head[1].weight.grad is not None
    assert all(p.grad is None for p in m.backbone.feature_extractor.parameters())      # frozen
    assert any(p.grad is not None for p in m.backbone.encoder.parameters())            # fine-tuned
    assert torch.isclose(m.layer_weights().sum(), torch.tensor(1.0))
    assert m.layer_weights().numel() == TINY["num_hidden_layers"] + 1


def test_without_layer_weighting():
    m = tiny_model(layer_weighting=False)
    assert m.layer_logits is None and m(torch.randn(2, 32000)).shape == (2, 7)


def test_optimizer_has_two_learning_rates_and_skips_frozen_params():
    cfg = {"train": {"lr": 5e-5, "head_lr": 1e-3, "weight_decay": 0.01}}
    m = tiny_model()
    opt, max_lr = make_optimizer(m, cfg)
    assert max_lr == [5e-5, 1e-3] and len(opt.param_groups) == 2
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) not in in_opt for p in m.backbone.feature_extractor.parameters())
    assert id(m.layer_logits) in {id(p) for p in opt.param_groups[1]["params"]}


def test_wavlm_training_and_eval_only(tmp_path):
    cfg = {"name": "w", "seed": 0, "manifest": str(make_manifest(tmp_path)), "sample_rate": 16000,
           "window_s": 2.0, "data": {"pad_mode": "repeat"},
           "model": {"type": "wavlm", "pretrained": None, "backbone_config": TINY, "dropout": 0.1},
           "train": {"epochs": 2, "batch_size": 16, "lr": 5e-4, "head_lr": 5e-3, "pct_start": 0.2,
                     "weight_decay": 0.0, "grad_clip": 1.0, "label_smoothing": 0.0, "patience": 5,
                     "num_workers": 0, "noise_prob": 0.5, "snr_db": [15, 40], "amp": True,
                     "dataset_weights": {}, "delete_last_when_done": True},
           "out_dir": str(tmp_path / "runs")}
    res = run(cfg)
    out = tmp_path / "runs" / "w"
    assert (out / "best.pt").exists() and not (out / "last.pt").exists()        # last.pt cleaned up
    assert set(json.loads((out / "metrics.json").read_text())) >= {"val", "test", "xcorpus_test"}
    again = run(cfg, eval_only=True)
    assert again["test"]["uar"] == res["test"]["uar"]
    run(cfg, smoke=True)
    assert (tmp_path / "runs" / "w_smoke" / "metrics.json").exists()           # smoke runs use their own folder
