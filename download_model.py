"""
CivicGuard AI - One-Time Model Download Script
Downloads XLM-R model and saves to local ./model_cache/ directory.
Run this ONCE, then all training/inference works fully offline.

Usage:
    python download_model.py
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = str(PROJECT_ROOT / "model_cache")


def download_and_cache():
    print("=" * 60)
    print("  CivicGuard AI - Model Download (One-Time)")
    print("=" * 60)
    print(f"\n  Model:  xlm-roberta-base")
    print(f"  Cache:  {CACHE_DIR}")
    print(f"  Size:   ~1.12 GB\n")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Download tokenizer to local cache_dir
    print("[1/2] Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "xlm-roberta-base",
        cache_dir=CACHE_DIR,
    )
    print("  -> Tokenizer cached.")

    # Download model to local cache_dir
    print("[2/2] Downloading model (this may take 10-20 minutes)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        cache_dir=CACHE_DIR,
        num_labels=3,
        id2label={0: "hate", 1: "offensive", 2: "neutral"},
        label2id={"hate": 0, "offensive": 1, "neutral": 2},
    )
    print("  -> Model cached.")

    # Verify
    print("\n[VERIFY] Testing offline loading...")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    tokenizer2 = AutoTokenizer.from_pretrained(
        "xlm-roberta-base",
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )
    model2 = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        cache_dir=CACHE_DIR,
        local_files_only=True,
        num_labels=3,
    )
    print(f"  -> Offline loading OK! ({model2.num_parameters():,} parameters)")

    # Show cache contents
    cache_path = Path(CACHE_DIR)
    total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
    print(f"\n  Total cache size: {total_size / (1024*1024):.1f} MB")

    print("\n[OK] Model cached successfully!")
    print("     You can now train fully OFFLINE:")
    print("     set PYTHONUTF8=1")
    print("     python run_pipeline.py --train")


if __name__ == "__main__":
    download_and_cache()
