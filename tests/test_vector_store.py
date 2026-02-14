"""Tests for vector store deletion behavior."""
from __future__ import annotations

import unittest
from typing import Any

from core.vector_store import VectorStore


class _FakeCollection:
    """Minimal fake collection for delete_pdf tests."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.last_include: list[str] | None = None
        self.deleted_ids: list[str] | None = None

    def get(self, include: list[str]) -> dict[str, Any]:
        self.last_include = include
        return self.payload

    def delete(self, ids: list[str]) -> None:
        self.deleted_ids = ids


class VectorStoreTests(unittest.TestCase):
    """Validate vector-store logic that does not require a live DB."""

    @staticmethod
    def _build_store(fake_collection: _FakeCollection) -> VectorStore:
        store = VectorStore.__new__(VectorStore)
        store._collection = fake_collection  # type: ignore[attr-defined]
        return store

    def test_delete_pdf_uses_supported_include_fields(self) -> None:
        """delete_pdf should not request unsupported include=['ids', ...]."""

        fake_collection = _FakeCollection(
            payload={
                "ids": ["a", "b"],
                "metadatas": [{"source": "doc.pdf"}, {"source": "other.pdf"}],
            }
        )
        store = self._build_store(fake_collection)

        deleted = store.delete_pdf("doc.pdf")

        self.assertEqual(deleted, 1)
        self.assertEqual(fake_collection.last_include, ["metadatas"])
        self.assertEqual(fake_collection.deleted_ids, ["a"])

    def test_delete_pdf_handles_nested_results(self) -> None:
        """delete_pdf should flatten nested ids/metadatas payloads from Chroma."""

        fake_collection = _FakeCollection(
            payload={
                "ids": [["a", "b"]],
                "metadatas": [[{"source": "doc.pdf"}, {"source": "doc.pdf"}]],
            }
        )
        store = self._build_store(fake_collection)

        deleted = store.delete_pdf("doc.pdf")

        self.assertEqual(deleted, 2)
        self.assertEqual(fake_collection.deleted_ids, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
