import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.classifier import HierarchicalClassifier
from app.ocr import extract_text_from_image, ocr_available
from app.preprocessing import is_low_information
from app.retrieval import SolutionRetriever

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"

_classifier: Optional[HierarchicalClassifier] = None
_retriever: Optional[SolutionRetriever] = None
_training_metrics: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier, _retriever, _training_metrics
    try:
        _classifier = HierarchicalClassifier.load(ARTIFACT_DIR)
        _retriever = SolutionRetriever.load(ARTIFACT_DIR)
        metrics_path = ARTIFACT_DIR / "training_metrics.json"
        if metrics_path.exists():
            _training_metrics = json.loads(metrics_path.read_text())
    except FileNotFoundError:
        # Artifacts not built yet -- API will report unhealthy until `python train.py` is run.
        _classifier = None
        _retriever = None
    yield


app = FastAPI(
    title="IT Ticket Auto-Resolution API",
    description="Classifies IT support tickets and suggests instant solutions.",
    version="1.0.0",
    lifespan=lifespan,
)


class TicketRequest(BaseModel):
    text: str
    ticket_id: Optional[str] = None


class ClassificationResponse(BaseModel):
    ticket_id: Optional[str]
    coarse_category: str
    issue_type: str
    confidence: float
    should_escalate_to_human: bool
    suggested_resolution: Optional[str]
    similar_past_tickets: list
    response_time_ms: float


def _ensure_ready():
    if _classifier is None or _retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Run `python train.py` to build them.",
        )


@app.get("/health")
def health():
    return {
        "status": "ok" if _classifier is not None else "artifacts_missing",
        "ocr_available": ocr_available(),
        "training_metrics": _training_metrics,
    }


@app.get("/metrics")
def metrics():
    return _training_metrics


@app.post("/classify", response_model=ClassificationResponse)
def classify_ticket(request: TicketRequest):
    _ensure_ready()
    start = time.time()

    if is_low_information(request.text):
        raise HTTPException(
            status_code=422,
            detail="Ticket text too short/uninformative to classify reliably.",
        )

    result = _classifier.predict(request.text)
    resolution = _retriever.top_resolution_for_issue_type(result.issue_type)
    similar = _retriever.nearest_similar_tickets(request.text, result.issue_type, k=3)

    elapsed_ms = (time.time() - start) * 1000
    return ClassificationResponse(
        ticket_id=request.ticket_id,
        coarse_category=result.coarse_category,
        issue_type=result.issue_type,
        confidence=result.confidence,
        should_escalate_to_human=result.should_escalate,
        suggested_resolution=None if result.should_escalate else resolution,
        similar_past_tickets=similar,
        response_time_ms=round(elapsed_ms, 2),
    )


@app.post("/classify_image", response_model=ClassificationResponse)
async def classify_image_ticket(file: UploadFile = File(...), ticket_id: Optional[str] = None):
    _ensure_ready()
    start = time.time()

    image_bytes = await file.read()
    text, ocr_confidence = extract_text_from_image(image_bytes)

    if ocr_confidence < 0.3 or is_low_information(text):
        elapsed_ms = (time.time() - start) * 1000
        return ClassificationResponse(
            ticket_id=ticket_id,
            coarse_category="unknown",
            issue_type="unknown",
            confidence=0.0,
            should_escalate_to_human=True,
            suggested_resolution=None,
            similar_past_tickets=[],
            response_time_ms=round(elapsed_ms, 2),
        )

    result = _classifier.predict(text)
    resolution = _retriever.top_resolution_for_issue_type(result.issue_type)
    similar = _retriever.nearest_similar_tickets(text, result.issue_type, k=3)

    elapsed_ms = (time.time() - start) * 1000
    return ClassificationResponse(
        ticket_id=ticket_id,
        coarse_category=result.coarse_category,
        issue_type=result.issue_type,
        confidence=result.confidence,
        should_escalate_to_human=result.should_escalate,
        suggested_resolution=None if result.should_escalate else resolution,
        similar_past_tickets=similar,
        response_time_ms=round(elapsed_ms, 2),
    )
