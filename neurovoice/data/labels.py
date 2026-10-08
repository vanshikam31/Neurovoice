"""Unified emotion label scheme shared by every dataset."""

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL = {i: e for e, i in LABEL2ID.items()}

# Dataset-native codes -> names. "calm" (RAVDESS only) is resolved by
# `label_map` in configs/data.yaml, since whether it equals neutral is debatable.
RAVDESS_CODES = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
CREMAD_CODES = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fearful",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}
TESS_CODES = {
    "angry": "angry", "disgust": "disgust", "fear": "fearful",
    "happy": "happy", "neutral": "neutral", "ps": "surprised", "sad": "sad",
}
SAVEE_CODES = {
    "a": "angry", "d": "disgust", "f": "fearful", "h": "happy",
    "n": "neutral", "sa": "sad", "su": "surprised",
}
