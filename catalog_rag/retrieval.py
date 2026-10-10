"""Two interchangeable retrievers; TF-IDF has no network or model download."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import Chunk, Hit


class Retriever(Protocol):
    def search(self, question: str, k: int = 5) -> list[Hit]: ...


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Cannot create a retriever with no chunks")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            sublinear_tf=True, norm="l2"
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, question: str, k: int = 5) -> list[Hit]:
        if not question.strip() or k < 1:
            return []
        query = self.vectorizer.transform([question])
        scores = (self.matrix @ query.T).toarray().ravel()
        # Zero-overlap passages are not evidence and must not inflate Recall@5.
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[int(i)], float(scores[i])) for i in order if scores[i] > 0]


class HybridTfidfRetriever:
    """Blend word n-grams and character n-grams for a stronger offline model."""

    def __init__(self, chunks: list[Chunk], word_weight: float = 0.7):
        if not chunks:
            raise ValueError("Cannot create a retriever with no chunks")
        if not 0 <= word_weight <= 1:
            raise ValueError("word_weight must be between 0 and 1")
        self.chunks = chunks
        self.word_weight = word_weight
        self.words = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.characters = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        texts = [c.text for c in chunks]
        self.word_matrix = self.words.fit_transform(texts)
        self.char_matrix = self.characters.fit_transform(texts)

    def search(self, question: str, k: int = 5) -> list[Hit]:
        if not question.strip() or k < 1:
            return []
        q_word = self.words.transform([question])
        q_char = self.characters.transform([question])
        ws = (self.word_matrix @ q_word.T).toarray().ravel()
        cs = (self.char_matrix @ q_char.T).toarray().ravel()
        scores = self.word_weight * ws + (1 - self.word_weight) * cs
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[int(i)], float(scores[i])) for i in order if scores[i] > 0]


class EmbeddingRetriever:
    """Optional local/cached SentenceTransformer; may download on first setup."""

    def __init__(
        self, chunks: list[Chunk], model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *, local_only: bool = True,
    ):
        if not chunks:
            raise ValueError("Cannot create a retriever with no chunks")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install requirements-optional.txt for embeddings") from exc
        self.chunks = chunks
        self.model = SentenceTransformer(model_name, local_files_only=local_only)
        self.matrix = np.asarray(
            self.model.encode([c.text for c in chunks], normalize_embeddings=True),
            dtype=np.float32,
        )

    def search(self, question: str, k: int = 5) -> list[Hit]:
        if not question.strip() or k < 1:
            return []
        query = np.asarray(self.model.encode([question], normalize_embeddings=True)[0])
        scores = self.matrix @ query
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[int(i)], float(scores[i])) for i in order]


def create_retriever(chunks: list[Chunk], kind: str = "tfidf", *, model: str = "", local_only=True):
    if kind == "tfidf":
        return TfidfRetriever(chunks)
    if kind == "hybrid":
        return HybridTfidfRetriever(chunks)
    if kind == "embeddings":
        return EmbeddingRetriever(
            chunks, model_name=model or "sentence-transformers/all-MiniLM-L6-v2",
            local_only=local_only,
        )
    raise ValueError(f"Unknown retriever: {kind}")

