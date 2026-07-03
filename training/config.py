"""
CivicGuard AI — Training Configuration
Laptop-friendly settings for XLM-R fine-tuning.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Local model cache (HuggingFace blob format)
MODEL_CACHE_DIR = str(PROJECT_ROOT / "model_cache")


@dataclass
class TrainingConfig:
    """Configuration for model training — optimized for standard laptop."""

    # Model — always use the HF model name, cache_dir handles local loading
    model_name: str = "xlm-roberta-base"
    model_cache_dir: str = MODEL_CACHE_DIR
    num_labels: int = 3

    # Label mappings
    label2id: dict = field(default_factory=lambda: {
        "hate": 0,
        "offensive": 1,
        "neutral": 2,
    })
    id2label: dict = field(default_factory=lambda: {
        0: "hate",
        1: "offensive",
        2: "neutral",
    })

    # Tokenizer
    max_length: int = 128

    # Training hyperparameters (laptop-friendly)
    batch_size: int = 8
    eval_batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = False  # Disabled for CPU training

    # Paths
    data_dir: str = str(PROJECT_ROOT / "data" / "datasets")
    output_dir: str = str(PROJECT_ROOT / "model")
    logging_dir: str = str(Path(__file__).parent / "logs")

    # Logging & Saving
    logging_steps: int = 50
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Create output directories."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging_dir).mkdir(parents=True, exist_ok=True)
