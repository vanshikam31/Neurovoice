import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from test_train_smoke import make_manifest  # noqa: E402

from neurovoice.data.labels import ID2LABEL  # noqa: E402
from neurovoice.eda import class_balance, durations, example_mels, split_overview, summary  # noqa: E402


def test_eda_makes_all_figures(tmp_path):
    df = pd.read_csv(make_manifest(tmp_path))
    df["emotion"] = df["label"].map(ID2LABEL)
    df["uid"] = [f"u{i}" for i in range(len(df))]
    df["duration_s"] = 1.5
    out = tmp_path / "fig"
    out.mkdir()
    class_balance(df, out)
    durations(df, out, 3.0)
    split_overview(df, out)
    example_mels(df, out)
    for f in ("class_balance", "durations", "split_overview", "example_mels"):
        assert (out / f"{f}.png").stat().st_size > 5000, f
    assert "Duplicate audio paths: 0" in summary(df)
