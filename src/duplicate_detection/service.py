import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from simhash import Simhash

from .bm25_index import BM25Repository
from .embedder import Embedder
from .feedback import save_feedback
from .models import CandidateMatch, DetectionConfig, DetectionResult, Document
from .preprocess import Preprocessor


@dataclass
class IndexedDocument:
    document: Document
    normalized_text: str
    tokens: Sequence[str]
    simhash: int
    embedding_vector: Sequence[float]


@dataclass
class _PreparedDocument:
    document: Document
    normalized_text: str
    tokens: Sequence[str]
    simhash: int
    embedding_vector: Sequence[float]


class DuplicateDetectionService:
    """Simple duplicate detection using BM25 → SimHash → embeddings."""

    def __init__(
        self,
        documents: Iterable[Document],
        config: Optional[DetectionConfig] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.config = config or DetectionConfig()
        self.embedder = embedder or Embedder()
        self.preprocessor = Preprocessor(self.config)

        self._documents: Dict[str, IndexedDocument] = {}
        self._bm25_repo: Optional[BM25Repository] = None
        self._initialize(list(documents))

    def _initialize(self, docs: List[Document]) -> None:
        if not docs:
            self._bm25_repo = BM25Repository([], self.preprocessor)
            return

        normalized: Dict[str, str] = {}
        tokens_map: Dict[str, Sequence[str]] = {}
        simhash_map: Dict[str, int] = {}

        for doc in docs:
            norm_text, tokenized = self.preprocessor.prepare(doc)
            tokens = tokenized.tokens
            normalized[doc.doc_id] = norm_text
            tokens_map[doc.doc_id] = tokens
            simhash_map[doc.doc_id] = self._compute_simhash(tokens)

        embeddings = {
            embedding.document.doc_id: embedding.vector
            for embedding in self.embedder.encode_documents(docs)
        }

        for doc in docs:
            self._documents[doc.doc_id] = IndexedDocument(
                document=doc,
                normalized_text=normalized[doc.doc_id],
                tokens=tokens_map[doc.doc_id],
                simhash=simhash_map[doc.doc_id],
                embedding_vector=embeddings[doc.doc_id],
            )

        self._bm25_repo = BM25Repository(docs, self.preprocessor)

    @property
    def documents(self) -> Sequence[Document]:
        return [indexed.document for indexed in self._documents.values()]

    def ingest_document(
        self,
        document: Document,
        recompute_index: bool = True,
        prepared: Optional[_PreparedDocument] = None,
    ) -> None:
        prepared_doc = prepared or self._prepare_document(document)
        self._documents[document.doc_id] = IndexedDocument(
            document=document,
            normalized_text=prepared_doc.normalized_text,
            tokens=prepared_doc.tokens,
            simhash=prepared_doc.simhash,
            embedding_vector=prepared_doc.embedding_vector,
        )
        if recompute_index:
            self._bm25_repo = BM25Repository(self.documents, self.preprocessor)

    def evaluate_document(
        self,
        document: Union[Document, str],
        include_self: bool = False,
    ) -> DetectionResult:
        if self._bm25_repo is None:
            raise RuntimeError("BM25 repository not initialised")

        prepared, base_document = self._prepare_input(document)
        return self._evaluate_prepared(base_document, prepared, include_self)

    def upload_document(
        self,
        document: Document,
        include_self: bool = False,
        recompute_index: bool = True,
    ) -> Tuple[DetectionResult, bool]:
        prepared = self._prepare_document(document)
        result = self._evaluate_prepared(document, prepared, include_self=include_self)
        ingested = False
        if not result.candidates:
            self.ingest_document(
                document, recompute_index=recompute_index, prepared=prepared
            )
            ingested = True
        return result, ingested

    def record_feedback(
        self,
        database_url: str,
        document_id: str,
        duplicate_id: str,
        verdict: str,
        notes: Optional[str] = None,
    ) -> None:
        save_feedback(
            database_url=database_url,
            document_id=document_id,
            duplicate_id=duplicate_id,
            verdict=verdict,
            notes=notes,
        )

    def _evaluate_prepared(
        self,
        base_document: Document,
        prepared: _PreparedDocument,
        include_self: bool,
    ) -> DetectionResult:
        bm25_candidates = self._bm25_repo.query(
            base_document.text, top_k=self.config.bm25_top_k
        )
        if not include_self:
            bm25_candidates = [
                candidate
                for candidate in bm25_candidates
                if candidate.doc.doc_id != base_document.doc_id
            ]

        matches: List[CandidateMatch] = []
        for candidate in bm25_candidates:
            indexed_candidate = self._documents.get(candidate.doc.doc_id)
            if indexed_candidate is None:
                continue

            simhash_similarity = self._simhash_similarity(
                prepared.simhash, indexed_candidate.simhash
            )
            if simhash_similarity < self.config.simhash_threshold:
                logging.debug(
                    "Document %s -> %s skipped: simhash %.3f < %.3f",
                    base_document.doc_id,
                    indexed_candidate.document.doc_id,
                    simhash_similarity,
                    self.config.simhash_threshold,
                )
                continue

            embedding_similarity = self.embedder.cosine_similarity(
                prepared.embedding_vector, indexed_candidate.embedding_vector
            )
            if embedding_similarity < self.config.embedding_threshold:
                logging.debug(
                    "Document %s -> %s skipped: embedding %.3f < %.3f",
                    base_document.doc_id,
                    indexed_candidate.document.doc_id,
                    embedding_similarity,
                    self.config.embedding_threshold,
                )
                continue

            matches.append(
                CandidateMatch(
                    doc_a=prepared.document,
                    doc_b=indexed_candidate.document,
                    bm25_score=candidate.score,
                    simhash_similarity=simhash_similarity,
                    embedding_similarity=embedding_similarity,
                )
            )

        matches.sort(
            key=lambda m: (m.embedding_similarity, m.simhash_similarity, m.bm25_score),
            reverse=True,
        )
        top_matches = matches[: self.config.max_candidates]
        return DetectionResult(
            candidates=top_matches, evaluated_document_id=base_document.doc_id
        )

    def _prepare_input(
        self,
        document: Union[Document, str],
    ) -> Tuple[_PreparedDocument, Document]:
        if isinstance(document, Document):
            doc_id = document.doc_id
            indexed = self._documents.get(doc_id)
            if indexed and indexed.document.text == document.text:
                return self._prepare_from_index(doc_id), indexed.document
            return self._prepare_document(document), document

        doc_id = document
        if doc_id not in self._documents:
            raise KeyError(f"Document ID {doc_id} not found in index")
        prepared = self._prepare_from_index(doc_id)
        return prepared, self._documents[doc_id].document

    def _prepare_from_index(self, doc_id: str) -> _PreparedDocument:
        indexed = self._documents[doc_id]
        return _PreparedDocument(
            document=indexed.document,
            normalized_text=indexed.normalized_text,
            tokens=indexed.tokens,
            simhash=indexed.simhash,
            embedding_vector=indexed.embedding_vector,
        )

    def _prepare_document(self, document: Document) -> _PreparedDocument:
        normalized_text, tokenized = self.preprocessor.prepare(document)
        tokens = tokenized.tokens
        simhash_value = self._compute_simhash(tokens)
        embedding_vector = self.embedder.encode_documents([document])[0].vector
        return _PreparedDocument(
            document=document,
            normalized_text=normalized_text,
            tokens=tokens,
            simhash=simhash_value,
            embedding_vector=embedding_vector,
        )

    def _compute_simhash(self, tokens: Sequence[str]) -> int:
        if not tokens:
            return 0
        return Simhash(tokens, f=self.config.simhash_bits).value

    def _simhash_similarity(self, hash_a: int, hash_b: int) -> float:
        if hash_a == hash_b:
            return 1.0
        bits = max(1, self.config.simhash_bits)
        xor = hash_a ^ hash_b
        distance = xor.bit_count()
        return 1.0 - (distance / bits)
