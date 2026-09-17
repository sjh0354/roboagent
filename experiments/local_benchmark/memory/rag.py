"""Auditable turn/chunk TF-IDF RAG baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer


class TurnChunkRAG:
    def __init__(self, log_path: Path, token_budget: int = 600):
        self.chunks: List[str] = []
        self.log_path = Path(log_path)
        self.token_budget = int(token_budget)

    def add(self, text: str) -> None:
        cleaned = str(text).strip()
        if cleaned:
            self.chunks.append(cleaned)

    def search(self, query: str, k: int = 3) -> str:
        if not self.chunks:
            return ""
        documents = self.chunks + [query]
        matrix = TfidfVectorizer(lowercase=True, ngram_range=(1, 2)).fit_transform(documents)
        scores = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
        order = sorted(range(len(self.chunks)), key=lambda index: (-scores[index], index))[:k]
        selected = [self.chunks[index] for index in order]
        text = "\n\n".join(selected)
        # Same explicit approximate-token policy as the rest of this repo.
        clipped = text[-self.token_budget * 4:]
        record = {
            "query": query, "selected_indices": order,
            "scores": [float(scores[index]) for index in order],
            "retrieval_chars": len(clipped), "token_budget": self.token_budget,
            "embedding": "scikit-learn TF-IDF word 1-2 grams",
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return "[TURN/CHUNK RAG]\n" + clipped
