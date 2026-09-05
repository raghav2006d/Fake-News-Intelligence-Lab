from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data import load_fake_real_dataset
from text_processing import clean_text


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "fake_news_pipeline.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"


def train_model() -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_fake_real_dataset()
    df["clean_content"] = df["content"].map(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_content"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    stop_words="english",
                ),
            ),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    mlflow.set_experiment("fake-news-detection")
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "dataset_samples": int(len(df)),
        }

        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        joblib.dump({"pipeline": pipeline, "metrics": metrics}, MODEL_PATH)
        pd.Series(metrics).to_json(METRICS_PATH, indent=2)

    return metrics


if __name__ == "__main__":
    result = train_model()
    print("Training complete")
    for key, value in result.items():
        print(f"{key}: {value}")
