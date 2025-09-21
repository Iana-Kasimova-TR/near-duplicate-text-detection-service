import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from duplicate_detection.embedder import Embedder
from duplicate_detection.loader import load_text_directory
from duplicate_detection.models import DetectionConfig, Document
from duplicate_detection.service import DuplicateDetectionService

DEFAULT_DATASET_DIR = os.environ.get("DUPLICATE_DATASET_DIR")
DEFAULT_MODEL_NAME = os.environ.get(
    "DUPLICATE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_BM25_TOP_K = int(os.environ.get("DUPLICATE_BM25_TOP_K", "200"))
DEFAULT_MAX_CANDIDATES = int(os.environ.get("DUPLICATE_MAX_CANDIDATES", "50"))
DEFAULT_FEEDBACK_DB = os.environ.get("DUPLICATE_FEEDBACK_DB", "feedback.db")

app = FastAPI(title="Duplicate Detection Service")
service: Optional[DuplicateDetectionService] = None


class EvaluateRequest(BaseModel):
    doc_id: str
    text: str


class UploadRequest(BaseModel):
    doc_id: str
    text: str
    title: Optional[str] = None
    include_self: bool = False


class FeedbackRequest(BaseModel):
    document_id: str
    duplicate_id: str
    verdict: str
    notes: Optional[str] = None


class CandidateResponse(BaseModel):
    doc_id: str
    title: Optional[str]
    bm25_score: float
    simhash_similarity: float
    embedding_similarity: float


class EvaluationResponse(BaseModel):
    evaluated_document_id: str
    candidates: List[CandidateResponse]


@app.on_event("startup")
async def startup_event() -> None:
    global service
    if service is not None:
        return

    dataset_dir = DEFAULT_DATASET_DIR
    if not dataset_dir:
        service = DuplicateDetectionService([], config=DetectionConfig(), embedder=Embedder(DEFAULT_MODEL_NAME))
        return

    documents = load_text_directory(
        Path(dataset_dir),
        limit=None,
    )
    config = DetectionConfig(
        bm25_top_k=DEFAULT_BM25_TOP_K,
        max_candidates=DEFAULT_MAX_CANDIDATES,
    )
    service = DuplicateDetectionService(documents, config=config, embedder=Embedder(DEFAULT_MODEL_NAME))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/duplicates/evaluate", response_model=EvaluationResponse)
async def evaluate(req: EvaluateRequest) -> EvaluationResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    document = Document(doc_id=req.doc_id, title=None, text=req.text)
    result = service.evaluate_document(document, include_self=False)
    candidates = [
        CandidateResponse(
            doc_id=candidate.doc_b.doc_id,
            title=candidate.doc_b.title,
            bm25_score=candidate.bm25_score,
            simhash_similarity=candidate.simhash_similarity,
            embedding_similarity=candidate.embedding_similarity,
        )
        for candidate in result.candidates
    ]
    return EvaluationResponse(
        evaluated_document_id=result.evaluated_document_id,
        candidates=candidates,
    )


@app.post("/duplicates/upload", response_model=EvaluationResponse)
async def upload(req: UploadRequest) -> EvaluationResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    document = Document(doc_id=req.doc_id, title=req.title, text=req.text)
    result, _ = service.upload_document(document, include_self=req.include_self)
    candidates = [
        CandidateResponse(
            doc_id=candidate.doc_b.doc_id,
            title=candidate.doc_b.title,
            bm25_score=candidate.bm25_score,
            simhash_similarity=candidate.simhash_similarity,
            embedding_similarity=candidate.embedding_similarity,
        )
        for candidate in result.candidates
    ]
    return EvaluationResponse(
        evaluated_document_id=result.evaluated_document_id,
        candidates=candidates,
    )


@app.post("/duplicates/feedback")
async def feedback(req: FeedbackRequest) -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    db_url = DEFAULT_FEEDBACK_DB
    service.record_feedback(db_url, req.document_id, req.duplicate_id, req.verdict, req.notes)
    return {"status": "saved"}
