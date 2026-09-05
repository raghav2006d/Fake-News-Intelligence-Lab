from pathlib import Path
import argparse
import os

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "models" / "distilbert_fake_news"
CACHE_DIR = ROOT / ".hf_cache"

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_DIR / "datasets"))

import evaluate
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

try:
    from src.data import load_fake_real_dataset
    from src.text_processing import clean_text
except ImportError:
    from data import load_fake_real_dataset
    from text_processing import clean_text


def prepare_dataset(sample_size: int | None):
    df = load_fake_real_dataset()
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42)
    df["text"] = df["content"].map(clean_text)
    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False).train_test_split(
        test_size=0.2,
        seed=42,
        stratify_by_column="label",
    )


def train_distilbert(sample_size: int | None, epochs: float, max_length: int, batch_size: int) -> None:
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = prepare_dataset(sample_size=sample_size)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "f1": f1_metric.compute(predictions=predictions, references=labels)["f1"],
        }

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=["mlflow"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for fake-news classification.")
    parser.add_argument("--sample-size", type=int, default=8000)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    train_distilbert(
        sample_size=args.sample_size,
        epochs=args.epochs,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
