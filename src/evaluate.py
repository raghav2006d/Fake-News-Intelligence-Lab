from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fake_news_pipeline.joblib"


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    metrics = bundle.get("metrics", {})
    print("Model metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
