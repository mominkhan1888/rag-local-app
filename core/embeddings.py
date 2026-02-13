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

# ---------------------------------------------------------------------------
# Backend availability detection
# ---------------------------------------------------------------------------
_SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

_CHROMADB_ONNX_AVAILABLE = False
_OnnxEmbeddingFunction = None
try:
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2  # type: ignore

    _OnnxEmbeddingFunction = ONNXMiniLM_L6_V2
    _CHROMADB_ONNX_AVAILABLE = True
except ImportError:
    try:
        from chromadb.utils.embedding_functions import (  # type: ignore
            DefaultEmbeddingFunction,
        )

        _OnnxEmbeddingFunction = DefaultEmbeddingFunction
        _CHROMADB_ONNX_AVAILABLE = True
    except ImportError:
        pass


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_name: str
    batch_size: int = 32


class EmbeddingGenerator:
    """Generates embeddings for text, auto-selecting the best available backend.

    Priority order:
        1. sentence-transformers (requires PyTorch — best for local development)
        2. ChromaDB ONNX all-MiniLM-L6-v2 (lightweight — ideal for cloud)
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Load the embedding model using the best available backend."""

        self.config = config
        self._backend: str | None = None

        # --- Try sentence-transformers first (best quality) ----------------
        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._st_model = SentenceTransformer(config.model_name)
                self._backend = "sentence-transformers"
                logger.info(
                    "Embedding backend: sentence-transformers (%s)",
                    config.model_name,
                )
                return
            except Exception:  # noqa: BLE001
                logger.warning(
                    "sentence-transformers installed but failed to load model '%s'; "
                    "falling back to ONNX backend",
                    config.model_name,
                )

        # --- Try ChromaDB ONNX embeddings (lightweight cloud fallback) -----
        if _CHROMADB_ONNX_AVAILABLE and _OnnxEmbeddingFunction is not None:
            try:
                self._onnx_ef = _OnnxEmbeddingFunction()
                self._backend = "chromadb-onnx"
                logger.info("Embedding backend: ChromaDB ONNX (all-MiniLM-L6-v2)")
                return
            except Exception:  # noqa: BLE001
                logger.warning("ChromaDB ONNX embedding function failed to initialize")

        raise RuntimeError(
            "No embedding backend available. "
            "Install sentence-transformers (for local) or onnxruntime + tokenizers (for cloud)."
        )

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
