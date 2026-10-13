from pathlib import Path

from neurovoice.data.build_manifest import build_manifest


def touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_nested_duplicate_folders_are_removed(tmp_path):
    for sub in ("TESS/OAF_angry", "TESS/TESS/OAF_angry"):  # same dataset twice
        touch(tmp_path / sub / "OAF_back_angry.wav")
        touch(tmp_path / sub / "OAF_bar_angry.wav")
    cfg = {"seed": 0, "datasets": {"tess": str(tmp_path)}, "label_map": {},
           "split": {"cross_corpus_holdout": []}}
    df = build_manifest(cfg)
    assert len(df) == 2
    assert df.uid.is_unique


def test_tess_with_typo_and_duplicates_stays_in_train(tmp_path):
    for sub in ("TESS/a", "TESS/TESS/a"):
        for f in ("OAF_back_angry", "OA_bite_fear", "YAF_back_ps"):
            touch(tmp_path / sub / f"{f}.wav")
    cfg = {"seed": 0, "datasets": {"tess": str(tmp_path)}, "label_map": {},
           "split": {"cross_corpus_holdout": []}}
    df = build_manifest(cfg)
    assert df.speaker_id.nunique() == 2
    assert set(df.split) == {"train"}      # 2 speakers -> never split
    assert len(df) == 3
