"""One parser per dataset. Each returns a list of dict records.

Parsers only read file/folder NAMES (no audio), so they are fast and testable.
speaker_id is globally unique: "<dataset>_<speaker>".
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from .labels import CREMAD_CODES, RAVDESS_CODES, SAVEE_CODES, TESS_CODES

log = logging.getLogger(__name__)


def _wavs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")


def _rec(path: Path, dataset: str, speaker: str, gender: str, emotion_raw: str) -> dict:
    return {
        "path": str(path),
        "stem": path.stem,
        "dataset": dataset,
        "speaker_id": f"{dataset}_{speaker}",
        "gender": gender,
        "emotion_raw": emotion_raw,
    }


def parse_ravdess(root: Path) -> list[dict]:
    """03-01-06-01-02-01-12.wav = modality-channel-emotion-intensity-statement-repetition-actor.
    Odd actor IDs are male, even are female. Only speech (channel 01) is kept."""
    out = []
    for p in _wavs(root):
        parts = p.stem.split("-")
        if len(parts) != 7:
            log.debug("RAVDESS: skipping %s", p.name)
            continue
        _, channel, emo, _, _, _, actor = parts
        if channel != "01" or emo not in RAVDESS_CODES:
            continue
        gender = "male" if int(actor) % 2 == 1 else "female"
        out.append(_rec(p, "ravdess", actor, gender, RAVDESS_CODES[emo]))
    return out


def _load_cremad_gender(root: Path) -> dict[str, str]:
    for csv_path in root.rglob("VideoDemographics.csv"):
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            return {str(r["ActorID"]).strip(): r["Sex"].strip().lower() for r in csv.DictReader(f)}
    log.warning("CREMA-D: VideoDemographics.csv not found; gender will be 'unknown'")
    return {}


def parse_cremad(root: Path) -> list[dict]:
    """1001_DFA_ANG_XX.wav = actor_sentence_emotion_level."""
    genders = _load_cremad_gender(root)
    out = []
    for p in _wavs(root):
        parts = p.stem.split("_")
        if len(parts) < 3 or parts[2] not in CREMAD_CODES:
            continue
        actor = parts[0]
        out.append(_rec(p, "cremad", actor, genders.get(actor, "unknown"), CREMAD_CODES[parts[2]]))
    return out


# TESS has exactly two speakers. Some files carry typo'd prefixes (e.g. "OA_..."),
# which would otherwise look like extra speakers and break speaker-independent splits.
_TESS_SPEAKERS = {"OA": "OAF", "YA": "YAF"}


def parse_tess(root: Path) -> list[dict]:
    """OAF_back_angry.wav = speaker_word_emotion. Two female speakers (OAF, YAF).
    Folder names differ between mirrors, so only the filename is used."""
    out, unknown = [], {}
    for p in _wavs(root):
        parts = p.stem.split("_")
        if len(parts) < 3:
            continue
        emo = parts[-1].lower()
        if emo not in TESS_CODES:
            continue
        spk = _TESS_SPEAKERS.get(parts[0].upper()[:2])
        if spk is None:
            unknown[parts[0]] = unknown.get(parts[0], 0) + 1
            continue
        out.append(_rec(p, "tess", spk, "female", TESS_CODES[emo]))
    if unknown:
        log.warning("TESS: skipped files with unrecognised speaker prefix: %s", unknown)
    return out


_SAVEE_SPEAKERS = {"DC", "JE", "JK", "KL"}
_SAVEE_EMO = re.compile(r"^(sa|su|a|d|f|h|n)(\d+)$")


def parse_savee(root: Path) -> list[dict]:
    """Handles both layouts: ALL/DC_a01.wav and AudioData/DC/a01.wav. Four male speakers."""
    out = []
    for p in _wavs(root):
        stem = p.stem
        if "_" in stem:
            spk, code = stem.split("_", 1)
        else:
            spk, code = p.parent.name, stem
        spk = spk.upper()
        m = _SAVEE_EMO.match(code.lower())
        if spk not in _SAVEE_SPEAKERS or not m:
            continue
        out.append(_rec(p, "savee", spk, "male", SAVEE_CODES[m.group(1)]))
    return out


PARSERS = {
    "ravdess": parse_ravdess,
    "cremad": parse_cremad,
    "tess": parse_tess,
    "savee": parse_savee,
}
