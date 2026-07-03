"""
CivicGuard AI — Post-Processing Rules Module
Lightweight rules to improve precision without overriding strong model predictions.
"""

import re
from typing import Dict, Optional


# Hate/slur keyword sets (minimal, for precision boosting)
HATE_KEYWORDS = {
    "exterminate", "genocide", "ethnic cleansing", "subhuman", "inferior race",
    "gas chamber", "lynch", "rape them", "sterilize them",
    # Hindi
    "maar daalo", "jala do", "kaat daalo",
    # Telugu
    "champeyaali", "tagalabettaali",
    # Bengali
    "mere felo", "jwale dao",
}

ABUSIVE_KEYWORDS = {
    "idiot", "moron", "stupid", "dumb", "loser", "pathetic",
    "trash", "garbage", "worthless", "shut up", "disgusting",
    # Hindi
    "bewkoof", "gadha", "bakwas", "wahiyat", "chutiya",
    # Telugu
    "gadidha", "buddhileni", "pichi",
    # Bengali
    "boka", "gadha", "pagol",
}

POSITIVE_INDICATORS = {
    "love", "great", "happy", "beautiful", "wonderful", "amazing",
    "grateful", "blessed", "inspiring", "peaceful", "kindness",
    "celebrate", "congratulations", "proud", "excellent",
}

# Context patterns for "kill/die" disambiguation
FIGURATIVE_KILL_PATTERNS = [
    r"kill(ed|ing|s)?\s+(it|the game|the competition|the audience|the crowd)",
    r"(killing|kills)\s+(me|my)\s+(budget|sleep|schedule|mood|vibe)",
    r"dying\s+(to|of)\s+(try|laughter|see|eat|know)",
    r"(exam|test|deadline|homework|assignment|traffic)\s+.*(killing|kill)",
    r"(killer|die-hard|to die for)",
    r"kill\s+(your|my)\s+(ego|doubts|fears|excuses|limits)",
    r"mar\s+(jaunga|jaungi|jayega|daalegi)",  # Hindi figurative
    r"tini\s+chachchipotha",  # Telugu figurative
]

THREAT_KILL_PATTERNS = [
    r"(want|going)\s+to\s+kill\s+(you|them|him|her|everyone|all)",
    r"(should|must|need to)\s+(die|be killed|be eliminated)",
    r"(deserve|deserves)\s+to\s+die",
    r"death\s+to\s+(all|them|you|every)",
    r"(kill|murder|eliminate)\s+(these|those|all)\s+(people|immigrants|refugees)",
]


def apply_post_processing(
    predicted_label: str,
    confidence: float,
    text: str,
    sentiment_score: float = 0.0,
) -> Dict:
    """
    Apply lightweight post-processing rules.

    IMPORTANT: Rules do NOT override strong model predictions (confidence > 0.85).
    They only adjust borderline cases (confidence < 0.7).

    Returns:
        dict with adjusted label, confidence, and explanation.
    """
    text_lower = text.lower()
    adjusted_label = predicted_label
    explanation_parts = []
    triggered_keywords = []

    # === RULE 0: Never override high-confidence model predictions ===
    if confidence > 0.85:
        return {
            "adjusted_label": predicted_label,
            "confidence": confidence,
            "explanation": "High-confidence model prediction",
            "keywords": [],
            "rule_applied": False,
        }

    # === RULE 1: Kill/Die context disambiguation ===
    has_kill_die = bool(re.search(r"\b(kill|die|death|murder|dying|killed|killing)\b", text_lower))
    has_hindi_kill = bool(re.search(r"(maar|marna|champ|kaat)", text_lower))
    has_telugu_kill = bool(re.search(r"(champu|champey|tagala)", text_lower))

    if has_kill_die or has_hindi_kill or has_telugu_kill:
        # Check for figurative usage
        is_figurative = any(
            re.search(pattern, text_lower) for pattern in FIGURATIVE_KILL_PATTERNS
        )

        # Check for threatening usage
        is_threat = any(
            re.search(pattern, text_lower) for pattern in THREAT_KILL_PATTERNS
        )

        if is_figurative and predicted_label == "hate":
            if confidence < 0.7:
                adjusted_label = "neutral"
                explanation_parts.append("Figurative use of violent language detected")

        elif is_threat and predicted_label != "hate":
            if confidence < 0.7:
                adjusted_label = "hate"
                explanation_parts.append("Direct threat detected")

        elif has_kill_die and sentiment_score < -0.6 and predicted_label != "hate":
            if confidence < 0.6:
                adjusted_label = "hate"
                explanation_parts.append("Negative sentiment with violent language")

    # === RULE 2: Positive word check ===
    positive_found = [w for w in POSITIVE_INDICATORS if w in text_lower]
    if positive_found and predicted_label == "hate" and confidence < 0.6:
        adjusted_label = "neutral"
        explanation_parts.append(f"Positive indicators found: {', '.join(positive_found[:3])}")
        triggered_keywords.extend(positive_found[:3])

    # === RULE 3: Abusive keyword check ===
    abusive_found = [w for w in ABUSIVE_KEYWORDS if w in text_lower]
    if abusive_found and predicted_label == "neutral" and confidence < 0.65:
        adjusted_label = "offensive"
        explanation_parts.append(f"Abusive keywords detected: {', '.join(abusive_found[:3])}")
        triggered_keywords.extend(abusive_found[:3])

    # === RULE 4: Extreme hate keyword check ===
    hate_found = [kw for kw in HATE_KEYWORDS if kw in text_lower]
    if hate_found and predicted_label != "hate" and confidence < 0.7:
        adjusted_label = "hate"
        explanation_parts.append(f"Extreme hate keywords detected: {', '.join(hate_found[:3])}")
        triggered_keywords.extend(hate_found[:3])

    rule_applied = adjusted_label != predicted_label
    explanation = "; ".join(explanation_parts) if explanation_parts else "Model prediction used"

    return {
        "adjusted_label": adjusted_label,
        "confidence": confidence if not rule_applied else max(confidence * 0.9, 0.5),
        "explanation": explanation,
        "keywords": triggered_keywords,
        "rule_applied": rule_applied,
    }


def get_explanation(label: str, confidence: float, text: str) -> str:
    """Generate a human-readable explanation for the prediction."""
    explanations = {
        "hate": [
            "This text contains language that targets a specific group or individual with dehumanizing, threatening, or discriminatory intent.",
            "Content classified as hate speech due to discriminatory language or calls for violence against identifiable groups.",
            "The language in this text promotes hostility or violence against people based on protected characteristics.",
        ],
        "offensive": [
            "This text contains rude, vulgar, or disrespectful language but does not target specific protected groups.",
            "Content uses offensive or inappropriate language without rising to the level of hate speech.",
            "The text includes insults or crude language directed at individuals or situations.",
        ],
        "neutral": [
            "This text does not contain hateful or offensive content.",
            "Content is classified as neutral — no harmful language detected.",
            "The text uses normal, non-harmful language.",
        ],
    }

    import random
    random.seed(hash(text) % 1000)
    base = random.choice(explanations.get(label, explanations["neutral"]))

    if confidence > 0.9:
        return f"{base} (High confidence: {confidence:.1%})"
    elif confidence > 0.7:
        return f"{base} (Moderate confidence: {confidence:.1%})"
    else:
        return f"{base} (Low confidence: {confidence:.1%} — borderline case)"
