from pathlib import Path
from typing import List

import joblib
import numpy as np

from src.explain import coefficient_explanation, lime_explanation
from src.text_processing import article_stats, clean_text


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "models" / "fake_news_pipeline.joblib"


class FakeNewsModelService:
    def __init__(self) -> None:
        self.pipeline = None
        self.metrics = {}
        self.model_name = "keyword_fallback"
        self.load()

    def load(self) -> None:
        if PIPELINE_PATH.exists():
            bundle = joblib.load(PIPELINE_PATH)
            self.pipeline = bundle["pipeline"]
            self.metrics = bundle.get("metrics", {})
            self.model_name = "tfidf_logistic_regression"
            return

        self.model_name = "keyword_fallback_train_model_first"

    def predict_one(self, text: str, explanation_method: str = "linear") -> dict:
        cleaned = clean_text(text)

        if self.pipeline is not None and hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba([cleaned])[0]
            fake_probability = float(probabilities[0])
            real_probability = float(probabilities[1])
        else:
            fake_probability, real_probability = self._fallback_score(cleaned)

        label = "Real News" if real_probability >= fake_probability else "Fake News"
        confidence = max(fake_probability, real_probability)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "fake_probability": round(fake_probability, 4),
            "real_probability": round(real_probability, 4),
            "stats": article_stats(text),
            "explanation": self.explain(cleaned, explanation_method=explanation_method),
            "model_name": self.model_name,
        }

    def predict_batch(self, texts: List[str]) -> List[dict]:
        return [self.predict_one(text) for text in texts]

    def explain(self, cleaned_text: str, explanation_method: str = "linear") -> list:
        if explanation_method == "lime":
            lime_terms = lime_explanation(self.pipeline, cleaned_text)
            if lime_terms:
                return lime_terms

        linear_terms = coefficient_explanation(self.pipeline, cleaned_text)
        if linear_terms:
            return linear_terms

        cleaned_words = cleaned_text.split()
        if not cleaned_words:
            return []

        suspicious_terms = {
            "shocking",
            "secret",
            "exposed",
            "hoax",
            "unbelievable",
            "urgent",
            "viral",
            "conspiracy",
            "breaking",
            "claim",
        }

        counts = {}
        for word in cleaned_words:
            if word in suspicious_terms or len(word) > 12:
                counts[word] = counts.get(word, 0) + 1

        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10]
        return [{"term": term, "weight": count} for term, count in ranked]

    def _fallback_score(self, cleaned: str) -> tuple:
        words = cleaned.split()
        if not words:
            return 0.5, 0.5

        clickbait_words = {
            "shocking",
            "secret",
            "exposed",
            "unbelievable",
            "miracle",
            "conspiracy",
            "urgent",
            "viral",
        }
        score = sum(1 for word in words if word in clickbait_words) / max(len(words), 1)
        fake_probability = float(np.clip(0.35 + score * 10, 0.05, 0.95))
        return fake_probability, 1 - fake_probability
