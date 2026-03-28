"""Local ONNX embedding client using sentence-transformers."""

import os
from pathlib import Path
from typing import List

from config import settings


class EmbeddingClient:
    """Generate embeddings using local ONNX model via sentence-transformers."""

    def __init__(
        self,
        model_name: str = None,
        dimensions: int = None,
        cache_dir: str = None,
    ):
        self.model_name = model_name or settings.vector.embedding_model
        self.dimensions = dimensions or settings.vector.dimensions
        hf_home = os.environ.get("HF_HOME") or str(
            Path.home() / ".cache" / "huggingface"
        )
        self.cache_dir = cache_dir or os.path.join(hf_home, "hub")

        os.makedirs(self.cache_dir, exist_ok=True)

        self._model = None

    def _load_model(self):
        """Lazy load ONNX model on first use."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self.model_name,
            cache_folder=self.cache_dir,
            device="cpu",
        )

    def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        self._load_model()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        self._load_model()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return embeddings.tolist()
