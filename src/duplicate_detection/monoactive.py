import hashlib
import math
import random
import struct
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .models import SimilarityScore


def _float_to_bits(value: float) -> int:
    return struct.unpack("!Q", struct.pack("!d", value))[0]


def _bits_to_float(bits: int) -> float:
    return struct.unpack("!d", struct.pack("!Q", bits))[0]


@dataclass(frozen=True)
class HashValue:
    token: str
    y_bits: int
    a: float

    @property
    def y(self) -> float:
        return _bits_to_float(self.y_bits)

    def matches(self, other: "HashValue") -> bool:
        return self.token == other.token and self.y_bits == other.y_bits


@dataclass
class WindowSignature:
    start: int
    end: int
    signature: List[Optional[HashValue]]


@dataclass
class MonoActiveProfile:
    tokens: Sequence[str]
    counts: Counter
    signature: List[Optional[HashValue]]
    window_signatures: List[WindowSignature]


class ICWSHashFunction:
    """Improved Consistent Weighted Sampling hash function."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._gamma_r: Dict[str, float] = {}
        self._gamma_c: Dict[str, float] = {}
        self._beta: Dict[str, float] = {}

    def hash(self, token: str, weight: float) -> HashValue:
        if weight <= 0:
            raise ValueError("Weight must be positive for ICWS hashing.")

        r = self._get_gamma(self._gamma_r, token, "r")
        c = self._get_gamma(self._gamma_c, token, "c")
        beta = self._get_beta(token)

        log_w = math.log(weight)
        t = math.floor(log_w / r + beta)
        y = math.exp(r * (t - beta))
        a = c / (y * math.exp(r))
        return HashValue(token=token, y_bits=_float_to_bits(y), a=a)

    def _get_gamma(self, cache: Dict[str, float], token: str, label: str) -> float:
        if token not in cache:
            rng = random.Random(self._seed(token, label))
            cache[token] = rng.gammavariate(2.0, 1.0)
        return cache[token]

    def _get_beta(self, token: str) -> float:
        if token not in self._beta:
            rng = random.Random(self._seed(token, "beta"))
            self._beta[token] = rng.random()
        return self._beta[token]

    def _seed(self, token: str, label: str) -> int:
        payload = f"{self.seed}|{label}|{token}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big") & 0xFFFFFFFF


class MonoActiveIndex:
    """MonoActive-inspired index using consistent weighted sampling."""

    def __init__(
        self,
        token_lookup: Optional[Dict[str, Sequence[str]]] = None,
        *,
        signature_size: int,
        window_size: int,
        stride: int,
        max_windows: int,
        window_match_threshold: float,
        seed: int = 1315423911,
    ) -> None:
        self.signature_size = max(1, signature_size)
        self.window_size = max(1, window_size)
        self.stride = max(1, stride)
        self.max_windows = max(1, max_windows)
        self.window_match_threshold = window_match_threshold
        self.hash_functions = [
            ICWSHashFunction(seed + i) for i in range(self.signature_size)
        ]
        self._profiles: Dict[str, MonoActiveProfile] = {}

        if token_lookup:
            for doc_id, tokens in token_lookup.items():
                self._profiles[doc_id] = self.build_profile(tokens)

    def ingest_document(self, doc_id: str, tokens: Sequence[str]) -> None:
        self._profiles[doc_id] = self.build_profile(tokens)

    def remove_document(self, doc_id: str) -> None:
        self._profiles.pop(doc_id, None)

    def get_profile(self, doc_id: str) -> Optional[MonoActiveProfile]:
        return self._profiles.get(doc_id)

    def build_profile(self, tokens: Sequence[str]) -> MonoActiveProfile:
        counts = Counter(tokens)
        signature = self._compute_signature(counts)
        window_signatures = self._build_windows(tokens)
        return MonoActiveProfile(
            tokens=tokens,
            counts=counts,
            signature=signature,
            window_signatures=window_signatures,
        )

    def score_profiles(
        self,
        profile_a: MonoActiveProfile,
        profile_b: MonoActiveProfile,
    ) -> SimilarityScore:
        weighted_jaccard = self._match_fraction(
            profile_a.signature, profile_b.signature
        )

        window_scores: List[float] = []
        for window_a in profile_a.window_signatures:
            best = 0.0
            for window_b in profile_b.window_signatures:
                candidate = self._match_fraction(window_a.signature, window_b.signature)
                if candidate > best:
                    best = candidate
                    if math.isclose(best, 1.0, rel_tol=1e-9):
                        break
            window_scores.append(best)

        max_window_score = max(window_scores) if window_scores else 0.0
        coverage = (
            sum(1 for score in window_scores if score >= self.window_match_threshold)
            / len(window_scores)
            if window_scores
            else 0.0
        )

        return SimilarityScore(
            weighted_jaccard=weighted_jaccard,
            max_window_score=max_window_score,
            window_coverage=coverage,
        )

    def _compute_signature(self, counts: Counter) -> List[Optional[HashValue]]:
        if not counts:
            return [None for _ in range(self.signature_size)]

        signature: List[Optional[HashValue]] = []
        for hash_fn in self.hash_functions:
            best_value: Optional[HashValue] = None
            for token, freq in counts.items():
                weight = self._weight(freq)
                if weight <= 0:
                    continue
                hv = hash_fn.hash(token, weight)
                if (
                    best_value is None
                    or hv.a < best_value.a
                    or (
                        math.isclose(hv.a, best_value.a)
                        and hv.y_bits < best_value.y_bits
                    )
                ):
                    best_value = hv
            signature.append(best_value)
        return signature

    def _build_windows(self, tokens: Sequence[str]) -> List[WindowSignature]:
        if not tokens:
            return []

        tokens_len = len(tokens)
        window_size = min(self.window_size, tokens_len)
        windows: List[WindowSignature] = []
        start = 0
        while start < tokens_len and len(windows) < self.max_windows:
            end = min(tokens_len, start + window_size)
            window_tokens = tokens[start:end]
            counts = Counter(window_tokens)
            signature = self._compute_signature(counts)
            windows.append(WindowSignature(start=start, end=end, signature=signature))
            if end >= tokens_len:
                break
            start += self.stride
        return windows

    def _match_fraction(
        self,
        signature_a: Sequence[Optional[HashValue]],
        signature_b: Sequence[Optional[HashValue]],
    ) -> float:
        matches = 0
        total = min(len(signature_a), len(signature_b), self.signature_size)
        if total == 0:
            return 0.0
        for idx in range(total):
            left = signature_a[idx]
            right = signature_b[idx]
            if left is not None and right is not None and left.matches(right):
                matches += 1
        return matches / total

    @staticmethod
    def _weight(frequency: int) -> float:
        return float(frequency)
