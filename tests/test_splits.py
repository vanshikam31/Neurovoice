import pandas as pd
import pytest

from neurovoice.data.splits import assign_splits, verify_no_leakage


def make_df():
    rows = []
    for i in range(20):  # dataset "a": 10 male, 10 female speakers
        for k in range(5):
            rows.append(("a", f"a_{i}", "male" if i % 2 else "female"))
    for s in ("b_0", "b_1"):  # tiny dataset -> all train
        rows += [("b", s, "female")] * 5
    for s in ("c_0", "c_1", "c_2", "c_3"):  # held-out dataset
        rows += [("c", s, "male")] * 5
    return pd.DataFrame(rows, columns=["dataset", "speaker_id", "gender"])


def run(seed=42):
    df = make_df()
    df["split"] = assign_splits(df, 0.15, 0.15, ["c"], seed)
    return df


def test_no_speaker_leakage():
    verify_no_leakage(run())


def test_holdout_and_tiny_dataset():
    df = run()
    assert set(df[df.dataset == "c"].split) == {"xcorpus_test"}
    assert set(df[df.dataset == "b"].split) == {"train"}


def test_all_splits_present_and_gender_balanced():
    a = run().query("dataset == 'a'")
    assert {"train", "val", "test"} <= set(a.split)
    for s in ("val", "test"):
        assert set(a[a.split == s].gender) == {"male", "female"}


def test_deterministic_and_seed_sensitive():
    assert run(1).split.equals(run(1).split)
    assert not run(1).split.equals(run(2).split)


def test_leakage_detector_fires():
    df = run()
    df.loc[df.index[0], "split"] = "test"
    df.loc[df.speaker_id == df.speaker_id.iloc[0], "split"] = ["test"] + ["train"] * 4
    with pytest.raises(AssertionError):
        verify_no_leakage(df)
