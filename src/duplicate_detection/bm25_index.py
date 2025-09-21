from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from rank_bm25 import BM25Okapi

from .models import DetectionConfig, Document
from .preprocess import Preprocessor


@dataclass
class BM25Candidate:
    doc: Document
    score: float


class BM25Repository:
    def __init__(
        self, documents: Sequence[Document], preprocessor: Preprocessor
    ) -> None:
        self.documents = list(documents)
        self.preprocessor = preprocessor
        self._corpus_tokens = [self._tokenize(doc) for doc in self.documents]
        self.index = BM25Okapi(self._corpus_tokens)

    def _tokenize(self, document: Document) -> List[str]:
        normalized = self.preprocessor.normalize(document.text)
        return self.preprocessor.tokenize(normalized).tokens

    def query(self, text: str, top_k: int) -> List[BM25Candidate]:
        normalized = self.preprocessor.normalize(text)
        tokens = self.preprocessor.tokenize(normalized).tokens
        scores = self.index.get_scores(tokens)
        ranked: List[Tuple[int, float]] = sorted(
            enumerate(scores), key=lambda t: t[1], reverse=True
        )[:top_k]
        return [
            BM25Candidate(doc=self.documents[idx], score=float(score))
            for idx, score in ranked
            if score > 0
        ]
