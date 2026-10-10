# NeuroVoice: Speech Emotion Recognition

Upload audio or record from your microphone and get the predicted emotion.
This repo is built in phases. **Phase 1: data pipeline. Phase 2: CNN baseline.**

## Phase 1: what it does
1. Parses RAVDESS, CREMA-D, TESS and SAVEE into one manifest with a **shared 7-class label scheme**
   (neutral, happy, sad, angry, fearful, disgust, surprised).
2. Creates **speaker-independent** train/val/test splits and a held-out **cross-corpus** test set.
3. Standardizes audio: mono, 16 kHz, silence trimmed, peak-normalized.
4. Writes a data report (class balance, speakers per split) to `reports/data_report.md`.

## Setup
```bash
pip install -r requirements.txt
```
Download each dataset and unpack it under `data/raw/` (paths are set in `configs/data.yaml`):

| Dataset | Folder | Notes |
|---|---|---|
| RAVDESS (speech) | `data/raw/ravdess` | Zenodo, `Audio_Speech_Actors_01-24.zip`. Non-commercial license |
| CREMA-D | `data/raw/crema-d` | GitHub: CheyneyComputerScience/CREMA-D. Keep `VideoDemographics.csv` for gender |
| TESS | `data/raw/tess` | Toronto Emotional Speech Set |
| SAVEE | `data/raw/savee` | Requires free registration |

Check each dataset's license before redistributing anything. You can start with just one or two
datasets; missing folders are skipped.

## Run
```bash
python -m neurovoice.data.build_manifest --config configs/data.yaml   # manifest + report
python -m neurovoice.data.preprocess     --config configs/data.yaml   # processed audio
python -m pytest -q                                                   # tests
```

## Design decisions (worth mentioning in interviews)
- **Split by speaker, not by clip.** A random clip split puts the same voice in train and test and
  inflates accuracy. `verify_no_leakage` asserts no speaker appears in two splits.
- **Splits are gender-balanced** within each dataset where gender is known.
- **Cross-corpus holdout.** SAVEE (4 speakers, different recording setup) is never used for training;
  it measures generalization. Add more datasets under `cross_corpus_holdout` for a stricter test.
- **TESS has only 2 speakers**, so it can't be split and always goes to train. Treat its clips with
  care in later phases (it is very clean and easy; consider down-weighting it).
- **"calm" (RAVDESS) is merged into neutral** by default. Set `calm: null` in the config to drop it.
- **Class imbalance:** CREMA-D has no "surprised". That class comes from RAVDESS and TESS only, so
  expect it to be the weakest class. Phase 2 should use class weights or balanced sampling.
- Long clips are not cropped here; sliding windows are applied in training and inference.

## Layout
```
configs/data.yaml            all paths and parameters
neurovoice/data/labels.py    unified label scheme + per-dataset code maps
neurovoice/data/parsers.py   one parser per dataset (filename -> speaker, emotion, gender)
neurovoice/data/splits.py    speaker-independent splitting + leakage check
neurovoice/data/build_manifest.py
neurovoice/data/preprocess.py
tests/                       parsers, splits, audio, and an end-to-end smoke test
```

## Phase 2: CNN baseline
Raw 3-second windows -> log-mel (computed inside the model) -> 4-block CNN -> 7 emotions.

```bash
# PyTorch first (CPU build is much smaller; pick the right command at pytorch.org if you have a GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m neurovoice.train --smoke      # ~1 minute sanity check on a tiny subset
python -m neurovoice.train              # full run (CPU: roughly 1-2 minutes per epoch)
```
Outputs go to `runs/cnn_baseline/`: `metrics.json`, `history.csv`, `confusion_*.png`, `best.pt`
(the checkpoint is git-ignored).

**Training details**
- Class-weighted loss (inverse frequency) so small classes like "surprised" are not ignored.
- Augmentation: random crop, Gaussian noise at random SNR, SpecAugment masks.
- Model selection by **validation UAR** only. Test and cross-corpus (SAVEE) sets are evaluated once at the end.
- Reported: accuracy, UAR, macro-F1, per-class recall, confusion matrices, and breakdowns by dataset and gender.
- Optional `train.dataset_weights` (e.g. `{tess: 0.5}`) down-weights the very clean TESS corpus. Good for an ablation.

**Known limitations**
- In-corpus val/test have only ~32 "surprised" clips (CREMA-D has none), so that class's score is noisy.
  Use the SAVEE cross-corpus result as the stronger signal.
- Clips are center-cropped/padded to 3 s at evaluation; sliding-window inference comes with the demo app.
