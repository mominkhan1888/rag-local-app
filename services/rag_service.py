"""RAG orchestration service for indexing and answering questions."""
from __future__ import annotations

from pathlib import Path
import logging
import threading
import time
from typing import Callable, Iterator, List

from config.settings import Settings
from core.embeddings import EmbeddingConfig, EmbeddingGenerator
from core.llm_client import OllamaClient, OpenAICompatClient
from core.pdf_processor import Chunk, PDFProcessor, ProgressCallback
from core.vector_store import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)


class RAGService:
    """High-level service that manages PDF indexing and query answering."""

    def __init__(self, settings: Settings) -> None:
        """Initialize dependencies using application settings."""

        self.settings = settings
        self._pdf_processor = PDFProcessor(settings.chunk_size, settings.chunk_overlap)
        self._embedder: EmbeddingGenerator | None = None
        self._vector_store: VectorStore | None = None
        self._dependency_lock = threading.Lock()
        provider = settings.llm_provider.lower()
        if provider == "ollama":
            self._llm_client = OllamaClient(
                base_url=settings.ollama_base_url,
                model=settings.model_name,
                request_timeout_seconds=settings.request_timeout_seconds,
            )
        else:
            model = settings.openai_model
            if not model:
                if provider == "openrouter":
                    model = "openrouter/free"
                elif provider == "groq":
                    model = "llama-3.1-8b-instant"
                else:
                    model = "openrouter/free"
            self._llm_client = OpenAICompatClient(
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
                model=model,
                request_timeout_seconds=settings.request_timeout_seconds,
            )

    def _get_embedder(self) -> EmbeddingGenerator:
        """Return a lazily initialized embedding generator."""

        if self._embedder is not None:
            return self._embedder

        with self._dependency_lock:
            if self._embedder is None:
                self._embedder = EmbeddingGenerator(
                    EmbeddingConfig(
                        model_name=self.settings.embedding_model,
                        batch_size=self.settings.embedding_batch_size,
                    )
                )
        return self._embedder

    def _get_vector_store(self) -> VectorStore:
        """Return a lazily initialized vector store."""

        if self._vector_store is not None:
            return self._vector_store

        with self._dependency_lock:
            if self._vector_store is None:
                self._vector_store = VectorStore(
                    VectorStoreConfig(persist_path=self.settings.chroma_db_path)
                )
        return self._vector_store

    def index_pdf(
        self,
        pdf_path: Path,
        pdf_name: str,
        progress_callback: ProgressCallback | None = None,
    ) -> int:
        """Process a PDF and store its chunks in the vector database."""

        start_time = time.perf_counter()
        try:
            chunks = list(self._pdf_processor.iter_chunks(pdf_path, pdf_name, progress_callback))
        except Exception as exc:  # noqa: BLE001
            logger.exception("PDF extraction failed")
            raise RuntimeError(f"Failed to extract PDF text: {exc}") from exc

        chunks = [chunk for chunk in chunks if chunk.text.strip()]
        if not chunks:
            return 0

        try:
            deleted = self._get_vector_store().delete_pdf(pdf_name)
            if deleted:
                logger.info("Cleared %s existing chunks for %s", deleted, pdf_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not clear existing PDF data for %s: %s", pdf_name, exc)

        try:
            embeddings = self._get_embedder().embed_texts([chunk.text for chunk in chunks])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Embedding generation failed")
            raise RuntimeError(f"Failed to generate embeddings: {exc}") from exc

        try:
            added = self._get_vector_store().add_chunks(chunks, embeddings)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Vector store write failed")
            raise RuntimeError(f"Failed to store embeddings: {exc}") from exc

        elapsed = time.perf_counter() - start_time
        logger.info("Indexed PDF '%s' with %s chunks in %.2f seconds", pdf_name, added, elapsed)
        return added

    def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
        chat_history: List[dict] | None = None,
    ) -> Iterator[str]:
        """Stream a response to a user question using RAG."""

        retrieval_start = time.perf_counter()
        try:
            query_embedding = self._get_embedder().embed_query(question)
            matches = self._get_vector_store().similarity_search(
                query_embedding,
                top_k or self.settings.top_k,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed during retrieval")
            raise RuntimeError("Failed to retrieve context") from exc

        retrieval_elapsed = time.perf_counter() - retrieval_start
        logger.info("Retrieved context in %.2f seconds", retrieval_elapsed)

        if not matches:
            def _no_context_stream() -> Iterator[str]:
                yield (
                    "I don't have any indexed PDF content yet. "
                    "Please upload a PDF and click Index PDF, then ask your question again."
                )

            return _no_context_stream()

        history = self._prepare_history(chat_history or [], question)
        prompt = self._llm_client.build_prompt(question, matches, history=history)

        def _stream() -> Iterator[str]:
            start_time = time.perf_counter()
            try:
                for token in self._llm_client.stream_answer(prompt):
                    yield token
            finally:
                elapsed = time.perf_counter() - start_time
                logger.info("Streamed response in %.2f seconds", elapsed)

        return _stream()

    def _prepare_history(self, history: List[dict], question: str) -> List[dict]:
        """Trim and sanitize chat history for prompt injection."""

        cleaned: List[dict] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if not role or not content:
                continue
            cleaned.append({"role": role, "content": content})

        if cleaned and cleaned[-1]["role"].lower() == "assistant" and not cleaned[-1]["content"].strip():
            cleaned = cleaned[:-1]

        if cleaned and cleaned[-1]["role"].lower() == "user" and cleaned[-1]["content"].strip() == question.strip():
            cleaned = cleaned[:-1]

        max_messages = self.settings.chat_history_max_messages
        if max_messages > 0:
            cleaned = cleaned[-max_messages:]

        return cleaned

    def list_pdfs(self) -> List[str]:
        """Return a sorted list of indexed PDF names."""

        return self._get_vector_store().list_pdfs()

    def delete_pdf(self, pdf_name: str) -> dict:
        """Delete a PDF and all associated chunks from the vector database."""

        deleted_chunks = self._get_vector_store().delete_pdf(pdf_name)
        file_deleted = False
        file_missing = False
        try:
            file_path = self.settings.upload_dir / pdf_name
            if file_path.exists():
                file_path.unlink()
                file_deleted = True
            else:
                file_missing = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to delete uploaded file %s: %s", pdf_name, exc)

        return {
            "deleted_chunks": deleted_chunks,
            "file_deleted": file_deleted,
            "file_missing": file_missing,
        }
