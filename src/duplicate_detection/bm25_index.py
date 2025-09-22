from dataclasses import dataclass
from typing import List, Optional, Sequence

import bm25s

from .models import Chunk


@dataclass
class BM25Candidate:
    chunk: Chunk
    score: float


class BM25Repository:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self._tokenized_corpus = [list(chunk.tokens) for chunk in self.chunks]
        self._retriever: Optional[bm25s.BM25] = None
        if self._tokenized_corpus:
            self._retriever = bm25s.BM25()
            self._retriever.index(self._tokenized_corpus, show_progress=False)

    def query(self, tokens: Sequence[str], top_k: int) -> List[BM25Candidate]:
        if not tokens or self._retriever is None:
            return []
        corpus_size = len(self.chunks)
        if corpus_size == 0:
            return []
        k = min(max(top_k, 0), corpus_size)
        if k == 0:
            return []
        results = self._retriever.retrieve(
            [list(tokens)],
            k=k,
            return_as="tuple",
            show_progress=False,
            leave_progress=False,
            n_threads=0,
            backend_selection="auto",
        )
        indices = results.documents[0].tolist()
        scores = results.scores[0].tolist()
        output: List[BM25Candidate] = []
        for idx, score in zip(indices, scores):
            if score <= 0:
                continue
            output.append(BM25Candidate(chunk=self.chunks[idx], score=float(score)))
        return output
