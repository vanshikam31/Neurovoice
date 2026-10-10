import numpy as np
import pandas as pd

from neurovoice.metrics import breakdown, confusion, summarize

NAMES = ["a", "b", "c"]


def test_perfect_and_known_values():
    s = summarize([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], NAMES)
    assert s["accuracy"] == s["uar"] == s["macro_f1"] == 1.0
    # class a: 1/2 recall, class b: 2/2, class c: 0/1  -> UAR = 0.5
    s = summarize([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], NAMES)
    assert np.isclose(s["uar"], (0.5 + 1.0 + 0.0) / 3)
    assert np.isclose(s["accuracy"], 3 / 5)


def test_uar_ignores_classes_without_true_samples():
    s = summarize([0, 0, 1, 1], [0, 0, 1, 2], NAMES)      # class c never occurs as truth
    assert np.isclose(s["uar"], (1.0 + 0.5) / 2)


def test_confusion_matrix_counts():
    cm = confusion([0, 1, 1], [0, 1, 0], 3)
    assert cm.tolist() == [[1, 0, 0], [1, 1, 0], [0, 0, 0]]


def test_breakdown_by_group():
    meta = pd.DataFrame({"gender": ["m", "m", "f", "f"]})
    out = breakdown(meta, [0, 1, 0, 1], [0, 1, 1, 0], "gender", NAMES)
    assert out["m"]["accuracy"] == 1.0 and out["f"]["accuracy"] == 0.0
