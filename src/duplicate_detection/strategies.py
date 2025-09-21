import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from simhash import Simhash

from .models import SimilarityScore
from .monoactive import MonoActiveIndex, MonoActiveProfile


@dataclass
class WindowSimhash:
    start: int
    end: int
    fingerprint: int


@dataclass
class SimhashProfile:
    tokens: Sequence[str]
    fingerprint: int
    windows: List[WindowSimhash]


class DuplicateStrategy(ABC):
    @abstractmethod
    def ingest_document(self, doc_id: str, tokens: Sequence[str]) -> None: ...

    @abstractmethod
    def remove_document(self, doc_id: str) -> None: ...

    @abstractmethod
    def get_profile(self, doc_id: str): ...

    @abstractmethod
    def build_profile(self, tokens: Sequence[str]): ...

    @abstractmethod
    def score_profiles(self, profile_a, profile_b) -> SimilarityScore: ...


class MonoActiveStrategy(DuplicateStrategy):
    def __init__(
        self,
        *,
        token_lookup: Optional[Dict[str, Sequence[str]]] = None,
        signature_size: int,
        window_size: int,
        stride: int,
        max_windows: int,
        window_match_threshold: float,
    ) -> None:
        tokens = token_lookup or {}
        self.index = MonoActiveIndex(
            token_lookup=tokens,
            signature_size=signature_size,
            window_size=window_size,
            stride=stride,
            max_windows=max_windows,
            window_match_threshold=window_match_threshold,
        )

    def ingest_document(self, doc_id: str, tokens: Sequence[str]) -> None:
        self.index.ingest_document(doc_id, tokens)

    def remove_document(self, doc_id: str) -> None:
        self.index.remove_document(doc_id)

    def get_profile(self, doc_id: str) -> Optional[MonoActiveProfile]:
        return self.index.get_profile(doc_id)

    def build_profile(self, tokens: Sequence[str]) -> MonoActiveProfile:
        return self.index.build_profile(tokens)

    def score_profiles(
        self, profile_a: MonoActiveProfile, profile_b: MonoActiveProfile
    ) -> SimilarityScore:
        return self.index.score_profiles(profile_a, profile_b)


class SimhashStrategy(DuplicateStrategy):
    def __init__(
        self,
        *,
        token_lookup: Optional[Dict[str, Sequence[str]]] = None,
        bits: int,
        window_size: int,
        stride: int,
        window_match_threshold: float,
    ) -> None:
        self.bits = bits
        self.window_size = max(1, window_size)
        self.stride = max(1, stride)
        self.window_match_threshold = window_match_threshold
        self._profiles: Dict[str, SimhashProfile] = {}

        if token_lookup:
            for doc_id, tokens in token_lookup.items():
                self._profiles[doc_id] = self.build_profile(tokens)

    def ingest_document(self, doc_id: str, tokens: Sequence[str]) -> None:
        self._profiles[doc_id] = self.build_profile(tokens)

    def remove_document(self, doc_id: str) -> None:
        self._profiles.pop(doc_id, None)

    def get_profile(self, doc_id: str) -> Optional[SimhashProfile]:
        return self._profiles.get(doc_id)

    def build_profile(self, tokens: Sequence[str]) -> SimhashProfile:
        fingerprint = self._simhash(tokens)
        windows = self._window_fingerprints(tokens)
        return SimhashProfile(tokens=tokens, fingerprint=fingerprint, windows=windows)

    def score_profiles(
        self, profile_a: SimhashProfile, profile_b: SimhashProfile
    ) -> SimilarityScore:
        full_similarity = self._similarity(profile_a.fingerprint, profile_b.fingerprint)

        window_scores: List[float] = []
        for window_a in profile_a.windows:
            best = 0.0
            for window_b in profile_b.windows:
                score = self._similarity(window_a.fingerprint, window_b.fingerprint)
                if score > best:
                    best = score
                    if math.isclose(best, 1.0, rel_tol=1e-9):
                        break
            window_scores.append(best)

        max_window = max(window_scores) if window_scores else 0.0
        coverage = (
            sum(1 for score in window_scores if score >= self.window_match_threshold)
            / len(window_scores)
            if window_scores
            else 0.0
        )

        return SimilarityScore(
            weighted_jaccard=full_similarity,
            max_window_score=max_window,
            window_coverage=coverage,
        )

    def _simhash(self, tokens: Sequence[str]) -> int:
        if not tokens:
            return 0
        features = [(token, freq) for token, freq in Counter(tokens).items()]
        return Simhash(features, f=self.bits).value

    def _window_fingerprints(self, tokens: Sequence[str]) -> List[WindowSimhash]:
        if not tokens:
            return []
        length = len(tokens)
        window_size = min(self.window_size, length)
        fingerprints: List[WindowSimhash] = []
        start = 0
        while start < length:
            end = min(length, start + window_size)
            window_tokens = tokens[start:end]
            fingerprint = self._simhash(window_tokens)
            fingerprints.append(
                WindowSimhash(start=start, end=end, fingerprint=fingerprint)
            )
            if end >= length:
                break
            start += self.stride
        return fingerprints

    def _similarity(self, fp_a: int, fp_b: int) -> float:
        if self.bits == 0:
            return 0.0
        xor = fp_a ^ fp_b
        distance = xor.bit_count()
        return 1.0 - (distance / self.bits)
