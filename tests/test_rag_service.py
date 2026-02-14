"""Tests for RAG service orchestration helpers."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from services.rag_service import RAGService


class RAGServiceTests(unittest.TestCase):
    """Validate lightweight logic in RAGService without heavy model initialization."""

    def test_index_uploaded_files_partial_success(self) -> None:
        """Batch indexing should continue when one file fails."""

        with TemporaryDirectory() as temp_dir:
            service = RAGService.__new__(RAGService)
            service.settings = SimpleNamespace(upload_dir=Path(temp_dir))

            indexed_names: list[str] = []

            def fake_index(file_path: Path, pdf_name: str, progress_callback=None) -> int:
                indexed_names.append(pdf_name)
                if progress_callback:
                    progress_callback(1.0, f"done {pdf_name}")
                if pdf_name == "bad.pdf":
                    raise RuntimeError("simulated failure")
                return 3

            service.index_pdf = fake_index  # type: ignore[method-assign]
            progress_events: list[tuple[str, float, str]] = []
            results = service.index_uploaded_files(
                [
                    ("good.pdf", b"good"),
                    ("bad.pdf", b"bad"),
                    ("good2.pdf", b"good2"),
                ],
                progress_callback=lambda name, progress, message: progress_events.append(
                    (name, progress, message)
                ),
            )

            self.assertEqual(indexed_names, ["good.pdf", "bad.pdf", "good2.pdf"])
            self.assertEqual(len(results), 3)
            self.assertTrue(results[0].success)
            self.assertFalse(results[1].success)
            self.assertTrue(results[2].success)
            self.assertIn("simulated failure", results[1].error)
            self.assertTrue((Path(temp_dir) / "good.pdf").exists())
            self.assertTrue((Path(temp_dir) / "bad.pdf").exists())
            self.assertTrue((Path(temp_dir) / "good2.pdf").exists())
            self.assertTrue(progress_events)
            self.assertGreaterEqual(progress_events[-1][1], 1.0)

    def test_index_uploaded_files_keeps_duplicate_names_for_replace_flow(self) -> None:
        """Duplicate file names should be passed through for replacement semantics."""

        with TemporaryDirectory() as temp_dir:
            service = RAGService.__new__(RAGService)
            service.settings = SimpleNamespace(upload_dir=Path(temp_dir))

            calls: list[str] = []

            def fake_index(file_path: Path, pdf_name: str, progress_callback=None) -> int:
                calls.append(pdf_name)
                return 1

            service.index_pdf = fake_index  # type: ignore[method-assign]

            results = service.index_uploaded_files(
                [
                    ("duplicate.pdf", b"first"),
                    ("duplicate.pdf", b"second"),
                ]
            )

            self.assertEqual(calls, ["duplicate.pdf", "duplicate.pdf"])
            self.assertTrue(all(result.success for result in results))

    def test_prepare_history_excludes_system_messages(self) -> None:
        """Prompt history should only contain user/assistant roles."""

        service = RAGService.__new__(RAGService)
        service.settings = SimpleNamespace(chat_history_max_messages=12)

        prepared = service._prepare_history(
            history=[
                {"role": "system", "content": "Indexed file"},
                {"role": "user", "content": "What is this about?"},
                {"role": "assistant", "content": "It is about testing."},
                {"role": "user", "content": "What is this about?"},
            ],
            question="What is this about?",
        )

        self.assertEqual(
            prepared,
            [
                {"role": "user", "content": "What is this about?"},
                {"role": "assistant", "content": "It is about testing."},
            ],
        )


if __name__ == "__main__":
    unittest.main()
