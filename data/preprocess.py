"""
CivicGuard AI — Data Preprocessing Module
Handles text cleaning, normalization, and train/val split for multilingual data.
"""

import re
import pandas as pd
import emoji
from pathlib import Path
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).parent / "datasets"


def clean_text(text: str) -> str:
    """
    Preprocess text while preserving emojis and code-mixed content.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove mentions (@user)
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol but keep the text
    text = re.sub(r"#(\w+)", r"\1", text)

    # Normalize repeated characters (e.g., "haaappy" → "haappy")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Keep emojis — they're important for sentiment
    # emoji library handles this natively

    return text


def normalize_transliteration(text: str) -> str:
    """
    Normalize common transliteration variations in Romanized text.
    Handles Hindi, Telugu, Bengali romanized forms.
    """
    # Common normalization patterns
    replacements = {
        r"\baa\b": "a",
        r"\bee\b": "i",
        r"\boo\b": "u",
        "shri": "sri",
        "th": "t",  # Only for specific cases
    }
    # NOTE: Light normalization only — aggressive normalization can hurt
    # code-mixed text understanding

    # Normalize common variations
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)  # Reduce letter repetition

    return text


def preprocess_dataset(input_path: str = None, output_dir: str = None):
    """
    Load, clean, and split the merged dataset into train/validation sets.
    """
    if input_path is None:
        input_path = DATA_DIR / "merged_dataset.csv"
    if output_dir is None:
        output_dir = DATA_DIR

    print("=" * 60)
    print("  CivicGuard AI — Data Preprocessing")
    print("=" * 60)

    # Load dataset
    print(f"\n[LOAD] Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  → Loaded {len(df)} samples")

    # Clean text
    print("[CLEAN] Cleaning text...")
    df["text"] = df["text"].apply(clean_text)

    # Remove empty texts
    df = df[df["text"].str.len() > 0]

    # Remove duplicates after cleaning
    df = df.drop_duplicates(subset=["text"], keep="first")
    print(f"  → {len(df)} samples after cleaning")

    # Map labels to integers
    label2id = {"hate": 0, "offensive": 1, "neutral": 2}
    id2label = {v: k for k, v in label2id.items()}

    df["label_id"] = df["label"].map(label2id)

    # Remove any rows with unmapped labels
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    # Print distribution
    print(f"\n[DIST] Label distribution:")
    for label, count in df["label"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  → {label}: {count} ({pct:.1f}%)")

    # Train/Validation split (80/20)
    print(f"\n[SPLIT] Splitting dataset (80/20)...")
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label_id"],
    )

    print(f"  → Train: {len(train_df)} samples")
    print(f"  → Validation: {len(val_df)} samples")

    # Save processed datasets
    output_dir = Path(output_dir)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)

    # Save label mappings
    import json
    with open(output_dir / "label_map.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    print(f"\n✓ Preprocessing complete!")
    print(f"  Train: {output_dir / 'train.csv'}")
    print(f"  Val:   {output_dir / 'val.csv'}")
    print(f"  Labels: {output_dir / 'label_map.json'}")

    return train_df, val_df, label2id, id2label


if __name__ == "__main__":
    preprocess_dataset()
