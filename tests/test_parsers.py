from pathlib import Path

from neurovoice.data.parsers import parse_cremad, parse_ravdess, parse_savee, parse_tess


def touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def by_stem(recs):
    return {r["stem"]: r for r in recs}


def test_ravdess(tmp_path):
    touch(tmp_path / "Actor_01" / "03-01-06-01-02-01-01.wav")  # fearful, male
    touch(tmp_path / "Actor_02" / "03-01-02-01-01-01-02.wav")  # calm, female
    touch(tmp_path / "Actor_02" / "03-02-02-01-01-01-02.wav")  # song -> skipped
    recs = by_stem(parse_ravdess(tmp_path))
    assert len(recs) == 2
    a = recs["03-01-06-01-02-01-01"]
    assert (a["emotion_raw"], a["gender"], a["speaker_id"]) == ("fearful", "male", "ravdess_01")
    assert recs["03-01-02-01-01-01-02"]["emotion_raw"] == "calm"
    assert recs["03-01-02-01-01-01-02"]["gender"] == "female"


def test_cremad_with_demographics(tmp_path):
    touch(tmp_path / "AudioWAV" / "1001_DFA_ANG_XX.wav")
    touch(tmp_path / "AudioWAV" / "1002_IEO_HAP_HI.wav")
    (tmp_path / "VideoDemographics.csv").write_text(
        "ActorID,Age,Sex,Race,Ethnicity\n1001,51,Male,Caucasian,Not Hispanic\n"
        "1002,21,Female,Caucasian,Not Hispanic\n")
    recs = by_stem(parse_cremad(tmp_path))
    assert recs["1001_DFA_ANG_XX"]["gender"] == "male"
    assert recs["1002_IEO_HAP_HI"]["emotion_raw"] == "happy"
    assert recs["1002_IEO_HAP_HI"]["speaker_id"] == "cremad_1002"


def test_cremad_without_demographics(tmp_path):
    touch(tmp_path / "1001_DFA_SAD_XX.wav")
    assert parse_cremad(tmp_path)[0]["gender"] == "unknown"


def test_tess_maps_ps_and_fear(tmp_path):
    touch(tmp_path / "OAF_Pleasant_surprise" / "OAF_back_ps.wav")
    touch(tmp_path / "YAF_fear" / "YAF_bar_fear.wav")
    recs = by_stem(parse_tess(tmp_path))
    assert recs["OAF_back_ps"]["emotion_raw"] == "surprised"
    assert recs["YAF_bar_fear"]["emotion_raw"] == "fearful"
    assert recs["YAF_bar_fear"]["speaker_id"] == "tess_YAF"


def test_savee_both_layouts(tmp_path):
    touch(tmp_path / "ALL" / "DC_sa02.wav")
    touch(tmp_path / "AudioData" / "JE" / "su05.wav")
    touch(tmp_path / "ALL" / "readme.wav")  # junk -> skipped
    recs = by_stem(parse_savee(tmp_path))
    assert recs["DC_sa02"]["emotion_raw"] == "sad"
    assert recs["su05"]["emotion_raw"] == "surprised"
    assert recs["su05"]["speaker_id"] == "savee_JE"
    assert len(recs) == 2


def test_tess_typo_prefix_is_not_a_new_speaker(tmp_path):
    touch(tmp_path / "OAF_Fear" / "OAF_bar_fear.wav")
    touch(tmp_path / "OAF_Fear" / "OA_bite_fear.wav")      # typo'd prefix
    touch(tmp_path / "YAF_fear" / "YAF_bar_fear.wav")
    touch(tmp_path / "YAF_fear" / "XYZ_bar_fear.wav")      # unknown -> skipped
    recs = parse_tess(tmp_path)
    assert {r["speaker_id"] for r in recs} == {"tess_OAF", "tess_YAF"}
    assert len(recs) == 3
