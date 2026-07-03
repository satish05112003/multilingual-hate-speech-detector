"""
CivicGuard AI — FastAPI Backend
Real-time hate speech detection API.
"""

import sys
from pathlib import Path
from typing import Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from backend.inference import get_engine


# ─── Pydantic Models ───────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")


class BatchAnalyzeRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50, description="List of texts to analyze")


class AnalyzeResponse(BaseModel):
    label: str = Field(..., description="Classification: hate, offensive, or neutral")
    confidence: float = Field(..., description="Model confidence score (0-1)")
    sentiment_score: Optional[float] = Field(default=0.0, description="Sentiment score (-1 to 1)")
    explanation: Optional[str] = Field(default="", description="Human-readable explanation")
    keywords: List[str] = Field(default=[], description="Triggered keywords")
    raw_probabilities: Optional[dict] = Field(default=None, description="Per-class probabilities")


class BatchAnalyzeResponse(BaseModel):
    results: List[AnalyzeResponse]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# ─── FastAPI App ───────────────────────────────────────────────────

app = FastAPI(
    title="CivicGuard AI",
    description="Multilingual Context-Aware Hate Speech Detection API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Chrome extension and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Startup Event ─────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load model on startup — cached in memory."""
    print("\n" + "=" * 60)
    print("  CivicGuard AI — Backend Starting")
    print("=" * 60)
    get_engine()
    print("\n✓ Backend ready!")


# ─── Endpoints ──────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    engine = get_engine()
    return HealthResponse(
        status="healthy",
        model_loaded=engine.is_loaded,
        version="1.0.0",
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze a single text for hate speech.

    Flow: Input → Tokenizer → XLM-R → Logits → Label → Rules → Output
    """
    try:
        engine = get_engine()
        result = engine.predict(request.text)
        return AnalyzeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/analyze/batch", response_model=BatchAnalyzeResponse, tags=["Analysis"])
async def analyze_batch(request: BatchAnalyzeRequest):
    """Analyze multiple texts in a single request."""
    try:
        engine = get_engine()
        results = engine.predict_batch(request.texts)
        return BatchAnalyzeResponse(
            results=[AnalyzeResponse(**r) for r in results],
            total=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "CivicGuard AI",
        "version": "1.0.0",
        "description": "Multilingual Context-Aware Hate Speech Detection",
        "endpoints": {
            "POST /analyze": "Analyze single text",
            "POST /analyze/batch": "Analyze multiple texts",
            "GET /health": "Health check",
            "GET /docs": "API documentation (Swagger)",
        },
        "supported_languages": ["English", "Hindi", "Telugu", "Bengali", "Code-mixed"],
    }


# ─── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  # Disable reload to prevent double model loading
        log_level="info",
    )
