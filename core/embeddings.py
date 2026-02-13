"""Embedding generation using sentence-transformers."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_name: str
    batch_size: int = 32


class EmbeddingGenerator:
    """Generates embeddings for text using a SentenceTransformer model."""

    def __init__(self, config: EmbeddingConfig) -> None:
        """Load the embedding model."""

        self.config = config
        try:
            self._model = SentenceTransformer(config.model_name)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load embedding model")
            raise RuntimeError("Failed to load embedding model") from exc

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""

        if not texts:
            return []
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Embedding generation failed")
            raise RuntimeError("Embedding generation failed") from exc

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a single query text."""

        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []
