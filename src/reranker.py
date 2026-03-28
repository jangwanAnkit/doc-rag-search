#!/usr/bin/env python3
"""
Cross-Encoder Reranker for improving retrieval precision.
"""

from dataclasses import dataclass
from typing import List, Optional

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


@dataclass
class RankedResult:
    text: str
    score: float
    rerank_score: float
    metadata: dict


class Reranker:
    """Cross-encoder reranker for improving retrieval precision."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: List[object], top_k: int = 5
    ) -> List[RankedResult]:
        """
        Rerank results using cross-encoder.

        Args:
            query: User query
            results: List of retrieval results with 'text', 'score', 'metadata'
            top_k: Number of top results to return after reranking

        Returns:
            Top-k reranked results
        """
        if not results:
            return []

        pairs = [(query, r.text) for r in results]

        ce_scores = self.model.predict(pairs)

        ranked = []
        for result, ce_score in zip(results, ce_scores):
            ranked.append(
                RankedResult(
                    text=result.text,
                    score=result.score,
                    rerank_score=float(ce_score),
                    metadata=result.metadata,
                )
            )

        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        return ranked[:top_k]


if __name__ == "__main__":
    from rag_pipeline import RetrievalResult

    reranker = Reranker()

    test_results = [
        RetrievalResult(
            text="The plaintiff claimed breach of contract for failing to deliver goods.",
            score=0.85,
            metadata={"source": "smith_v_jones.pdf"},
        ),
        RetrievalResult(
            text="The defendant denied any wrongdoing in the employment matter.",
            score=0.82,
            metadata={"source": "doe_v_acme.pdf"},
        ),
        RetrievalResult(
            text="The court awarded damages of $175,000 to the plaintiff.",
            score=0.78,
            metadata={"source": "smith_v_jones.pdf"},
        ),
    ]

    query = "breach of contract damages"
    reranked = reranker.rerank(query, test_results, top_k=3)

    print(f"Query: {query}")
    print(f"\nOriginal results:")
    for i, r in enumerate(test_results, 1):
        print(f"  {i}. [{r.score:.3f}] {r.metadata['source']}: {r.text[:50]}...")

    print(f"\nReranked results:")
    for i, r in enumerate(reranked, 1):
        print(f"  {i}. [{r.rerank_score:.3f}] {r.metadata['source']}: {r.text[:50]}...")
