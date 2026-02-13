"""PDF text extraction and chunking utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re
from typing import Any, Callable, Dict, Iterator, List, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class Chunk:
    """A text chunk with associated metadata."""

    text: str
    metadata: Dict[str, Any]


class PDFProcessor:
    """Extracts text from PDFs and splits it into overlapping chunks."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """Initialize the processor with chunking parameters."""

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def iter_chunks(
        self,
        pdf_path: Path,
        source_name: str,
        progress_callback: ProgressCallback | None = None,
    ) -> Iterator[Chunk]:
        """Yield chunks from a PDF without loading the entire document into memory."""

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to open PDF: %s", pdf_path)
            raise RuntimeError(f"Failed to open PDF: {pdf_path}") from exc

        total_pages = doc.page_count
        token_buffer: List[str] = []
        page_buffer: List[int] = []
        chunk_id = 0

        try:
            for page_index in range(total_pages):
                try:
                    page = doc.load_page(page_index)
                    text = page.get_text("text")
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to extract page %s", page_index + 1)
                    continue

                tokens = self._tokenize(text)
                for token in tokens:
                    token_buffer.append(token)
                    page_buffer.append(page_index + 1)

                    if len(token_buffer) >= self.chunk_size:
                        chunk_text, page_start, page_end = self._build_chunk(
                            token_buffer[: self.chunk_size],
                            page_buffer[: self.chunk_size],
                        )
                        metadata = {
                            "source": source_name,
                            "chunk_id": chunk_id,
                            "page_start": page_start,
                            "page_end": page_end,
                        }
                        yield Chunk(text=chunk_text, metadata=metadata)
                        chunk_id += 1

                        token_buffer = token_buffer[self.chunk_size - self.chunk_overlap :]
                        page_buffer = page_buffer[self.chunk_size - self.chunk_overlap :]

                if progress_callback:
                    progress = (page_index + 1) / max(total_pages, 1)
                    progress_callback(progress, f"Processed page {page_index + 1}/{total_pages}")

            if token_buffer:
                chunk_text, page_start, page_end = self._build_chunk(token_buffer, page_buffer)
                metadata = {
                    "source": source_name,
                    "chunk_id": chunk_id,
                    "page_start": page_start,
                    "page_end": page_end,
                }
                yield Chunk(text=chunk_text, metadata=metadata)
        finally:
            doc.close()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into a simple word/punctuation list."""

        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    @staticmethod
    def _build_chunk(tokens: List[str], pages: List[int]) -> Tuple[str, int, int]:
        """Build chunk text and page range from token buffers."""

        chunk_text = " ".join(tokens).strip()
        page_start = min(pages) if pages else 1
        page_end = max(pages) if pages else 1
        return chunk_text, page_start, page_end
