from pathlib import Path
from zipfile import ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = ROOT / "fake news dataset ml.zip"


def load_fake_real_dataset(zip_path: Path = DEFAULT_ZIP) -> pd.DataFrame:
    """Load the Fake.csv/True.csv dataset from the local zip file."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")

    with ZipFile(zip_path) as archive:
        with archive.open("Fake.csv") as fake_file:
            fake_df = pd.read_csv(fake_file)
        with archive.open("True.csv") as true_file:
            true_df = pd.read_csv(true_file)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = (df["title"] + " " + df["text"]).str.strip()
    return df[["title", "text", "subject", "date", "content", "label"]]
