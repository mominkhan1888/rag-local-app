"""Tests for cloud-safe settings behavior."""
from __future__ import annotations

import builtins
import os
import unittest
from unittest.mock import patch

from config import settings as settings_module


class SettingsTests(unittest.TestCase):
    """Validate settings defaults and runtime validation rules."""

    def setUp(self) -> None:
        """Snapshot environment variables before each test."""

        self._env_backup = os.environ.copy()

    def tearDown(self) -> None:
        """Restore environment variables after each test."""

        os.environ.clear()
        os.environ.update(self._env_backup)

    def _clear_app_env(self) -> None:
        """Remove app-specific environment variables for test isolation."""

        keys = [
            "CHUNK_SIZE",
            "CHUNK_OVERLAP",
            "MODEL_NAME",
            "LLM_PROVIDER",
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
            "EMBEDDING_MODEL",
            "CHROMA_DB_PATH",
            "UPLOAD_DIR",
            "OLLAMA_BASE_URL",
            "TOP_K",
            "REQUEST_TIMEOUT_SECONDS",
            "EMBEDDING_BATCH_SIZE",
            "CHAT_HISTORY_PATH",
            "CHAT_HISTORY_MAX_MESSAGES",
            "STREAMLIT_SHARING_MODE",
        ]
        for key in keys:
            os.environ.pop(key, None)

    def test_settings_load_without_dotenv(self) -> None:
        """Settings loader should not fail when python-dotenv is unavailable."""

        real_import = builtins.__import__

        def fake_import(name: str, globals_dict=None, locals_dict=None, fromlist=(), level: int = 0):
            if name == "dotenv":
                raise ImportError("dotenv unavailable")
            return real_import(name, globals_dict, locals_dict, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            settings_module._load_dotenv_if_available()

    def test_cloud_default_provider_openrouter(self) -> None:
        """Cloud mode should default to OpenRouter provider."""

        self._clear_app_env()
        os.environ["STREAMLIT_SHARING_MODE"] = "streamlit"

        settings = settings_module.get_settings()

        self.assertEqual(settings.llm_provider, "openrouter")
        self.assertEqual(settings.openai_base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(settings.openai_model, "openrouter/free")

    def test_validate_settings_openrouter_missing_key(self) -> None:
        """OpenRouter mode must require OPENAI_API_KEY."""

        self._clear_app_env()
        os.environ["LLM_PROVIDER"] = "openrouter"
        settings = settings_module.get_settings()

        with self.assertRaises(ValueError) as error:
            settings_module.validate_settings(settings)

        self.assertIn("OPENAI_API_KEY", str(error.exception))

    def test_validate_settings_ollama_mode(self) -> None:
        """Ollama mode should validate when required values exist."""

        self._clear_app_env()
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
        os.environ["MODEL_NAME"] = "llama3.2:3b"

        settings = settings_module.get_settings()
        settings_module.validate_settings(settings)


if __name__ == "__main__":
    unittest.main()
