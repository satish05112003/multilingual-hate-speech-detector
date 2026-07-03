"""
CivicGuard AI — Inference Engine
Handles model loading, prediction, and post-processing for real-time inference.
Optimized with torch.no_grad() and model caching.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Project paths
_project_root = Path(__file__).parent.parent


class CivicGuardInference:
    """
    Singleton-style inference engine for CivicGuard AI.
    Loads model once and caches in memory for fast predictions.
    """

    _instance = None
    _model = None
    _tokenizer = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = None):
        if self._model is not None:
            return  # Already initialized

        self._model_name = "Hate-speech-CNERG/indic-abusive-allInOne-MuRIL"
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_model()

    def _load_model(self):
        """Load model and tokenizer from HuggingFace."""
        print(f"[LOAD] Loading CivicGuard model...")
        print(f"[DEVICE] Using: {self._device}")

        # STEP 1: LOAD MODEL
        print(f"[MODEL] Loading model from: {self._model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        
        # STEP 2: CHECK LABEL ORDER (IMPORTANT)
        print("Label Order Config:", self._model.config.id2label)

        self._model.to(self._device)
        self._model.eval()  # Set to evaluation mode
        print(f"[OK] Model loaded successfully ({self._model.num_parameters():,} parameters)")

    def predict(self, text: str) -> Dict:
        """
        Predict hate speech label for input text.
        """
        # STEP 3: USE PROBABILITIES (CRITICAL FIX)
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

        normal_prob = probs[0].item()
        abusive_prob = probs[1].item()

        # CHECKPOINT 9: DEBUG SUPPORT
        print(f"TEXT: {text}")
        print(f"PROBS: normal={normal_prob:.3f}, abusive={abusive_prob:.3f}")

        # TASK 5: FINAL THRESHOLDS
        if abusive_prob > 0.90:
            label = "hate"
            confidence = abusive_prob

        elif abusive_prob > 0.60:
            label = "offensive"
            confidence = abusive_prob

        else:
            label = "neutral"
            confidence = normal_prob

        # TASK 4: CONTEXT-AWARE RULES — applied in strict priority order
        text_lower = text.lower()

        # PRIORITY 1: Safe figurative phrases → always neutral
        safe_phrases = [
            "killing me",
            "kill time",
            "kill opportunities",
            "this exam is killing me",
            "dead tired",
            "dying laughing"
        ]

        # PRIORITY 2: Strong direct threats / hate patterns → hate
        hate_patterns = [
            "kill you",
            "i will kill",
            "should be eliminated",
            "deserve to die",
            "go die",
            "rape you",
            "exterminate you",
            # Telugu threat phrases
            "champestha",
            "champesta",
            "champi padata",
            "ninnu champestha"
        ]

        # PRIORITY 3: Soft abuse → offensive (only if not already hate)
        abuse_words = [
            # English
            "idiot", "stupid", "fuck you", "moron", "bitch", "asshole", "retard",
            # Hindi/Hinglish
            "chutiya", "chuthiya", "madarchod", "bhenchod",
            # Telugu slang
            "lanja", "puka", "dengey", "dengay", "dengudam",
            "sulli", "pichi", "vedhava", "erripuka",
            "nee amma", "nee ayya", "ra puka", "ra lanja"
        ]

        # Apply rules in priority order
        if any(p in text_lower for p in safe_phrases):
            label = "neutral"
            confidence = max(confidence, 0.7)

        elif any(p in text_lower for p in hate_patterns):
            label = "hate"
            confidence = max(confidence, 0.85)

        elif any(w in text_lower for w in abuse_words):
            if label != "hate":
                label = "offensive"
                confidence = max(confidence, 0.75)

        print(f"FINAL: {label}")

        # CHECKPOINT 7: CLEAN OUTPUT FORMAT
        return {
            "label": label,
            "confidence": round(confidence, 3),
            "raw_probabilities": {
                "normal": round(normal_prob, 3),
                "abusive": round(abusive_prob, 3)
            }
        }

    def predict_batch(self, texts: list) -> list:
        """Predict labels for multiple texts."""
        return [self.predict(text) for text in texts]

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# Global singleton instance
_engine: Optional[CivicGuardInference] = None

def get_engine(model_path: str = None) -> CivicGuardInference:
    """Get or create the global inference engine."""
    global _engine
    if _engine is None:
        _engine = CivicGuardInference(model_path)
    return _engine
