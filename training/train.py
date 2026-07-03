"""
CivicGuard AI — Model Training Script
Fine-tunes XLM-R on multilingual hate speech data.
Optimized for standard laptop (CPU or limited GPU).
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force offline mode — no downloads during training
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

from training.config import TrainingConfig


class HateSpeechDataset(Dataset):
    """Custom dataset for hate speech classification."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def compute_metrics(eval_pred):
    """Compute accuracy, F1, precision, and recall."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
    }


def train():
    """Main training function."""
    config = TrainingConfig()

    print("=" * 60)
    print("  CivicGuard AI - XLM-R Fine-tuning")
    print("=" * 60)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[DEVICE] Using: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        config.fp16 = True  # Enable FP16 for GPU training

    # Load data
    print(f"\n[DATA] Loading datasets...")
    train_df = pd.read_csv(Path(config.data_dir) / "train.csv")
    val_df = pd.read_csv(Path(config.data_dir) / "val.csv")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")

    # Load tokenizer from LOCAL CACHE — no download
    print(f"\n[MODEL] Loading from LOCAL CACHE (offline mode)...")
    print(f"  Model:     {config.model_name}")
    print(f"  Cache dir: {config.model_cache_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.model_cache_dir,
        local_files_only=True,
    )
    print("  Tokenizer loaded from local cache.")

    # Create datasets
    train_dataset = HateSpeechDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    val_dataset = HateSpeechDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    # Load model from LOCAL CACHE — no download
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        cache_dir=config.model_cache_dir,
        local_files_only=True,
        num_labels=config.num_labels,
        id2label=config.id2label,
        label2id=config.label2id,
    )
    print(f"  Model loaded from local cache.")
    print(f"  Parameters: {model.num_parameters():,}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=True,
        seed=config.seed,
        report_to="none",  # Disable W&B etc.
        dataloader_num_workers=0,  # Safe for Windows
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    print(f"\n[TRAIN] Starting training...")
    print(f"  Epochs:        {config.num_epochs}")
    print(f"  Batch size:    {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max length:    {config.max_length}")
    print(f"  FP16:          {config.fp16}")
    print()

    train_result = trainer.train()

    # Print results
    print(f"\n[RESULTS] Training complete!")
    print(f"  Training loss: {train_result.training_loss:.4f}")

    # Evaluate
    print(f"\n[EVAL] Evaluating on validation set...")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # Save model
    print(f"\n[SAVE] Saving model to {config.output_dir}...")
    model_save_path = Path(config.output_dir) / "civicguard-xlmr"
    trainer.save_model(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))

    # Save config
    with open(model_save_path / "training_config.json", "w") as f:
        json.dump({
            "model_name": config.model_name,
            "num_labels": config.num_labels,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "label2id": config.label2id,
            "id2label": {str(k): v for k, v in config.id2label.items()},
            "final_metrics": {k: float(v) for k, v in eval_results.items() if isinstance(v, float)},
        }, f, indent=2)

    print(f"\n[OK] Model saved to: {model_save_path}")
    print("[OK] Training pipeline complete!")

    return trainer, eval_results


if __name__ == "__main__":
    train()
