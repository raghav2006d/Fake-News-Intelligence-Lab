import re
from typing import Iterable, List


def clean_text(text: str) -> str:
    """Normalize article text for classical ML models."""
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_texts(texts: Iterable[str]) -> List[str]:
    return [clean_text(text) for text in texts]


def article_stats(text: str) -> dict:
    cleaned = clean_text(text)
    words = cleaned.split()
    sentences = [part for part in re.split(r"[.!?]+", str(text)) if part.strip()]
    unique_words = set(words)

    return {
        "characters": len(str(text)),
        "words": len(words),
        "sentences": len(sentences),
        "unique_words": len(unique_words),
        "lexical_diversity": round(len(unique_words) / max(len(words), 1), 4),
    }
