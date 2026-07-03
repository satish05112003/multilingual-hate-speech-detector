"""
CivicGuard AI - Master Pipeline Runner
Orchestrates the full pipeline: Dataset -> Preprocess -> Train -> Serve

Usage:
    python run_pipeline.py --all          # Run everything
    python run_pipeline.py --data         # Only create datasets
    python run_pipeline.py --preprocess   # Only preprocess
    python run_pipeline.py --train        # Only train
    python run_pipeline.py --serve        # Only start backend
    python run_pipeline.py --test         # Only run backtests
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def step_data():
    """Step 1: Create all datasets."""
    print("\n" + "=" * 60)
    print("  STEP 1: DATASET CREATION")
    print("=" * 60)
    from data.collect_datasets import (
        create_english_dataset,
        create_hindi_hinglish_dataset,
        create_bengali_dataset,
        create_telugu_dataset,
        create_context_dataset,
        merge_all_datasets,
    )

    create_english_dataset()
    create_hindi_hinglish_dataset()
    create_bengali_dataset()
    create_telugu_dataset()
    create_context_dataset()
    merge_all_datasets()


def step_preprocess():
    """Step 2: Preprocess and split data."""
    print("\n" + "=" * 60)
    print("  STEP 2: DATA PREPROCESSING")
    print("=" * 60)
    from data.preprocess import preprocess_dataset
    preprocess_dataset()


def step_train():
    """Step 3: Fine-tune XLM-R model."""
    print("\n" + "=" * 60)
    print("  STEP 3: MODEL TRAINING")
    print("=" * 60)
    from training.train import train
    train()


def step_test():
    """Step 4: Run backtests."""
    print("\n" + "=" * 60)
    print("  STEP 4: BACKTESTING")
    print("=" * 60)
    from tests.test_model import run_backtest
    run_backtest()


def step_serve():
    """Step 5: Start FastAPI backend."""
    print("\n" + "=" * 60)
    print("  STEP 5: STARTING BACKEND")
    print("=" * 60)
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )


def main():
    parser = argparse.ArgumentParser(
        description="CivicGuard AI — Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --all          Run full pipeline
  python run_pipeline.py --data         Create datasets only
  python run_pipeline.py --train        Train model only
  python run_pipeline.py --serve        Start API server only
  python run_pipeline.py --test         Run backtests only
        """,
    )

    parser.add_argument("--all", action="store_true", help="Run full pipeline (data → preprocess → train → test)")
    parser.add_argument("--data", action="store_true", help="Create datasets")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--test", action="store_true", help="Run backtests")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI backend")

    args = parser.parse_args()

    # Default to --all if no arguments
    if not any(vars(args).values()):
        print("No arguments provided. Use --help for usage.")
        print("Running --all by default...\n")
        args.all = True

    print("+" + "=" * 58 + "+")
    print("|" + "   Multilingual Hate Speech Detection".center(58) + "|")
    print("|" + "  Powered by XLM-RoBERTa".center(58) + "|")
    print("+" + "=" * 58 + "+")

    if args.all or args.data:
        step_data()

    if args.all or args.preprocess:
        step_preprocess()

    if args.all or args.train:
        step_train()

    if args.all or args.test:
        step_test()

    if args.serve:
        step_serve()

    if not args.serve:
        print("\n" + "=" * 60)
        print("  [OK] Pipeline complete!")
        print("  -> To start the backend: python run_pipeline.py --serve")
        print("  -> API docs: http://127.0.0.1:8000/docs")
        print("=" * 60)


if __name__ == "__main__":
    main()
