from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class Document:
    doc_id: str
    title: Optional[str]
    text: str
    metadata: Optional[dict] = None


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    tokens: Sequence[str]


@dataclass
class ChunkEmbedding:
    chunk: Chunk
    vector: List[float]


@dataclass
class DocumentEmbedding:
    document: Document
    vector: List[float]


@dataclass
class ChunkMatch:
    chunk_a: Chunk
    chunk_b: Chunk
    similarity: float


@dataclass
class SimilarityScore:
    weighted_jaccard: float
    max_window_score: float
    window_coverage: float


@dataclass
class ComparisonPreview:
    doc_a_id: str
    doc_b_id: str
    ready: bool = False


@dataclass
class CandidateMatch:
    doc_a: Document
    doc_b: Document
    bm25_score: float
    simhash_similarity: float
    embedding_similarity: float
    preview: Optional[ComparisonPreview] = None


@dataclass
class DetectionResult:
    candidates: List[CandidateMatch]
    evaluated_document_id: str


@dataclass
class DetectionConfig:
    max_candidates: int = 10
    bm25_top_k: int = 25
    simhash_bits: int = 64
    simhash_threshold: float = 0.8
    embedding_threshold: float = 0.8
    chunk_size: int = 250
    chunk_overlap: int = 50
    lowercase: bool = True
    strip_accents: bool = False
    remove_punctuation: bool = False
