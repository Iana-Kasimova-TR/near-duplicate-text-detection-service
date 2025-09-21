import re
import string
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import nltk

from .models import Chunk, DetectionConfig, Document


_WORD_RE = re.compile(r"\w+")


@dataclass
class TokenizerResult:
    tokens: List[str]
    offsets: List[int]


class Preprocessor:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._ensure_nltk_resources()

    @staticmethod
    def _ensure_nltk_resources() -> None:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

    def normalize(self, text: str) -> str:
        normalized = text
        if self.config.lowercase:
            normalized = normalized.lower()
        normalized = self._normalize_whitespace(normalized)
        if self.config.strip_accents:
            normalized = self._strip_accents(normalized)
        if self.config.remove_punctuation:
            normalized = normalized.translate(str.maketrans("", "", string.punctuation))
        return normalized

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _strip_accents(text: str) -> str:
        import unicodedata

        return "".join(
            c
            for c in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(c)
        )

    def tokenize(self, text: str) -> TokenizerResult:
        tokens: List[str] = []
        offsets: List[int] = []
        for match in _WORD_RE.finditer(text):
            tokens.append(match.group(0))
            offsets.append(match.start())
        return TokenizerResult(tokens=tokens, offsets=offsets)

    def prepare(self, document: Document) -> Tuple[str, TokenizerResult]:
        normalized = self.normalize(document.text)
        tokenized = self.tokenize(normalized)
        return normalized, tokenized

    def chunk(
        self,
        document: Document,
        normalized_text: Optional[str] = None,
        tokenized: Optional[TokenizerResult] = None,
    ) -> Sequence[Chunk]:
        if normalized_text is None or tokenized is None:
            normalized_text, tokenized = self.prepare(document)

        tokens = tokenized.tokens
        offsets = tokenized.offsets
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        if not tokens:
            return []

        chunks: List[Chunk] = []
        step = max(chunk_size - overlap, 1)
        for start_idx in range(0, len(tokens), step):
            end_idx = min(start_idx + chunk_size, len(tokens))
            if start_idx >= end_idx:
                break
            start_offset = offsets[start_idx]
            end_offset = offsets[end_idx - 1] + len(tokens[end_idx - 1])
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = normalized_text[start_offset:end_offset]
            chunk_id = f"{document.doc_id}_chunk_{len(chunks)}"
            chunks.append(
                Chunk(
                    doc_id=document.doc_id,
                    chunk_id=chunk_id,
                    text=chunk_text,
                    tokens=chunk_tokens,
                )
            )
            if end_idx == len(tokens):
                break

        return chunks


def build_chunks(
    documents: Iterable[Document], preprocessor: Preprocessor
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in documents:
        chunks.extend(preprocessor.chunk(doc))
    return chunks
