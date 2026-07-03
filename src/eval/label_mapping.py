"""Maps a VLM's short free-text answer to the canonical FER2013 7-class label
set, so free-text (VLM) and closed-set (CV/FER) models can be scored by the
same accuracy metric on the same unified sample. Unmappable text counts as a
miss (scored against ground truth as a wrong label), not excluded from the
denominator - a VLM that can't produce a clean label is a real accuracy cost.
"""
import re

FER2013_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

_SYNONYMS = {
    "angry": "angry", "anger": "angry", "mad": "angry", "furious": "angry", "irritated": "angry",
    "disgust": "disgust", "disgusted": "disgust", "revolted": "disgust",
    "fear": "fear", "fearful": "fear", "scared": "fear", "afraid": "fear", "terrified": "fear",
    "happy": "happy", "happiness": "happy", "joy": "happy", "joyful": "happy", "content": "happy",
    "sad": "sad", "sadness": "sad", "unhappy": "sad", "sorrowful": "sad", "down": "sad",
    "surprise": "surprise", "surprised": "surprise", "shocked": "surprise", "astonished": "surprise",
    "neutral": "neutral", "calm": "neutral", "indifferent": "neutral", "expressionless": "neutral",
}


def map_to_fer2013_label(raw_text: str) -> str | None:
    """Returns one of FER2013_LABELS, or None if no known word is found."""
    words = re.findall(r"[a-zA-Z]+", raw_text.lower())
    for word in words:
        if word in _SYNONYMS:
            return _SYNONYMS[word]
    return None
