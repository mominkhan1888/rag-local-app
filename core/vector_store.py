"""ChromaDB vector store wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Any, Dict, List

import chromadb

from core.pdf_processor import Chunk

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for the ChromaDB vector store."""

    persist_path: Path
    collection_name: str = "rag_chunks"


class VectorStore:
    """Persisted vector store with CRUD utilities for PDF chunks."""

    def __init__(self, config: VectorStoreConfig) -> None:
        """Initialize the ChromaDB client and collection."""

        self.config = config
        try:
            self._client = chromadb.PersistentClient(path=str(config.persist_path))
            self._collection = self._client.get_or_create_collection(
                name=config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to initialize ChromaDB")
            raise RuntimeError("Failed to initialize ChromaDB") from exc

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        """Add chunks and embeddings to the collection."""

        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch")

        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"{chunk.metadata['source']}:{chunk.metadata['chunk_id']}" for chunk in chunks]

        try:
            if hasattr(self._collection, "upsert"):
                self._collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
            else:
                self._collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to add chunks to ChromaDB")
            raise RuntimeError("Failed to add chunks to ChromaDB") from exc

        return len(chunks)

    def similarity_search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Return the most similar chunks to a query embedding."""

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to query ChromaDB")
            raise RuntimeError("Failed to query ChromaDB") from exc

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            matches.append({"text": doc, "metadata": meta, "distance": dist})

        return matches

    def list_pdfs(self) -> List[str]:
        """List unique PDF names stored in the collection."""

        try:
            data = self._collection.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to list PDFs")
            raise RuntimeError("Failed to list PDFs") from exc

        metadatas = data.get("metadatas") or []
        if metadatas and isinstance(metadatas[0], list):
            metadatas = [item for sublist in metadatas for item in sublist]
        pdfs = {meta.get("source", "") for meta in metadatas if isinstance(meta, dict)}
        return sorted(name for name in pdfs if name)

    def delete_pdf(self, pdf_name: str) -> int:
        """Delete all chunks for a given PDF name."""

        try:
            # Chroma's `ids` are always returned. Passing "ids" in include raises
            # a ValueError in 0.4.x, so request only supported include fields.
            data = self._collection.get(include=["metadatas"])
            ids = data.get("ids") or []
            metadatas = data.get("metadatas") or []

            if ids and isinstance(ids[0], list):
                ids = [item for sublist in ids for item in sublist]
            if metadatas and isinstance(metadatas[0], list):
                metadatas = [item for sublist in metadatas for item in sublist]

            if not ids or not metadatas:
                return 0

            ids_to_delete: List[str] = []
            for doc_id, meta in zip(ids, metadatas):
                if isinstance(meta, dict) and meta.get("source") == pdf_name:
                    ids_to_delete.append(doc_id)

            if not ids_to_delete:
                return 0

            self._collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to delete PDF %s", pdf_name)
            raise RuntimeError("Failed to delete PDF") from exc
