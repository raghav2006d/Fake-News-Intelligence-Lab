from functools import lru_cache

import numpy as np


def coefficient_explanation(pipeline, text: str, top_k: int = 12) -> list[dict]:
    """Fast explanation for TF-IDF linear models."""
    if pipeline is None:
        return []

    try:
        vectorizer = pipeline.named_steps["tfidf"]
        model = pipeline.named_steps["model"]
        transformed = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]
        active = transformed.nonzero()[1]

        scored_terms = []
        for index in active:
            contribution = float(transformed[0, index] * coefs[index])
            scored_terms.append(
                {
                    "term": feature_names[index],
                    "weight": round(abs(contribution), 5),
                    "direction": "real" if contribution > 0 else "fake",
                }
            )

        return sorted(scored_terms, key=lambda item: item["weight"], reverse=True)[:top_k]
    except Exception:
        return []


@lru_cache(maxsize=1)
def _lime_available():
    try:
        from lime.lime_text import LimeTextExplainer

        return LimeTextExplainer(class_names=["Fake News", "Real News"])
    except Exception:
        return None


def lime_explanation(pipeline, text: str, top_k: int = 10) -> list[dict]:
    """Optional LIME explanation, slower but model-agnostic."""
    explainer = _lime_available()
    if pipeline is None or explainer is None or not hasattr(pipeline, "predict_proba"):
        return []

    try:
        explanation = explainer.explain_instance(
            text,
            pipeline.predict_proba,
            num_features=top_k,
            labels=[0, 1],
        )
        label = int(np.argmax(pipeline.predict_proba([text])[0]))
        return [
            {
                "term": term,
                "weight": round(abs(float(weight)), 5),
                "direction": "real" if weight > 0 and label == 1 else "fake",
            }
            for term, weight in explanation.as_list(label=label)
        ]
    except Exception:
        return []
