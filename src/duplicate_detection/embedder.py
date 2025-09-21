from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .models import Chunk, ChunkEmbedding, Document, DocumentEmbedding


@dataclass
class EmbeddingStore:
    document_embeddings: List[DocumentEmbedding]
    chunk_embeddings: List[ChunkEmbedding]


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, documents: Iterable[Document]) -> List[DocumentEmbedding]:
        texts = [doc.text for doc in documents]
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [DocumentEmbedding(document=doc, vector=vec.tolist()) for doc, vec in zip(documents, vectors)]

    def encode_chunks(self, chunks: Sequence[Chunk]) -> List[ChunkEmbedding]:
        if not chunks:
            return []
        texts = [chunk.text for chunk in chunks]
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [ChunkEmbedding(chunk=chunk, vector=vec.tolist()) for chunk, vec in zip(chunks, vectors)]

    @staticmethod
    def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        a = np.asarray(vec_a)
        b = np.asarray(vec_b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

