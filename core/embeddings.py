"""Embedding generation with automatic backend selection.

Prefers sentence-transformers (PyTorch) when available for local development.
Falls back to ChromaDB's built-in ONNX embeddings (lightweight) for cloud
deployments like Streamlit Cloud where PyTorch is too heavy.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_name: str
    batch_size: int = 32


class EmbeddingGenerator:
    """Generates embeddings for text, auto-selecting the best available backend.

    Priority order:
        1. sentence-transformers (requires PyTorch - best for local development)
        2. ChromaDB ONNX all-MiniLM-L6-v2 (lightweight - ideal for cloud)
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Load the embedding model using the best available backend."""

        self.config = config
        self._backend: str | None = None
        self._st_model = None
        self._onnx_ef = None

        # Try sentence-transformers first (best quality for local runtime).
        if self._initialize_sentence_transformers():
            return

        # Fall back to ChromaDB ONNX embeddings (cloud-friendly and lightweight).
        if self._initialize_chromadb_onnx():
            return

        raise RuntimeError(
            "No embedding backend available. "
            "Install sentence-transformers (for local) or onnxruntime + tokenizers (for cloud)."
        )

    def _initialize_sentence_transformers(self) -> bool:
        """Initialize sentence-transformers backend when available."""

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            return False

        try:
            self._st_model = SentenceTransformer(self.config.model_name)
            self._backend = "sentence-transformers"
            logger.info("Embedding backend: sentence-transformers (%s)", self.config.model_name)
            return True
        except Exception:  # noqa: BLE001
            logger.warning(
                "sentence-transformers import succeeded but model '%s' failed to load; "
                "falling back to ONNX backend",
                self.config.model_name,
            )
            return False

    def _initialize_chromadb_onnx(self) -> bool:
        """Initialize ChromaDB ONNX embedding backend when available."""

        embedding_cls = None
        try:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2  # type: ignore

            embedding_cls = ONNXMiniLM_L6_V2
        except ImportError:
            try:
                from chromadb.utils.embedding_functions import (  # type: ignore
                    DefaultEmbeddingFunction,
                )

                embedding_cls = DefaultEmbeddingFunction
            except ImportError:
                return False

        try:
            self._onnx_ef = embedding_cls()
            self._backend = "chromadb-onnx"
            logger.info("Embedding backend: ChromaDB ONNX (all-MiniLM-L6-v2)")
            return True
        except Exception:  # noqa: BLE001
            logger.warning("ChromaDB ONNX embedding function failed to initialize")
            return False

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""

        if not texts:
            return []

        try:
            if self._backend == "sentence-transformers":
                embeddings = self._st_model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                return embeddings.tolist()

            if self._backend == "chromadb-onnx":
                raw = self._onnx_ef(texts)
                return [list(e) for e in raw]

        except Exception as exc:  # noqa: BLE001
            logger.exception("Embedding generation failed")
            raise RuntimeError("Embedding generation failed") from exc

        raise RuntimeError("No embedding backend configured")

    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a single query text."""

        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []
