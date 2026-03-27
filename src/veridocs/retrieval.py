from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import DocumentChunk


class Retriever:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = None
        self.chunks: list[DocumentChunk] = []

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        corpus = [chunk.text for chunk in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    def search(self, query: str, top_k: int = 4) -> list[tuple[DocumentChunk, float]]:
        if self.matrix is None or not self.chunks:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        if scores.size == 0:
            return []

        k = min(top_k, len(self.chunks))
        best_idx = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in best_idx]
