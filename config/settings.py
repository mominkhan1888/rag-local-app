"""Centralized configuration settings for the RAG app."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment variables."""

    chunk_size: int
    chunk_overlap: int
    model_name: str
    llm_provider: str
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    embedding_model: str
    chroma_db_path: Path
    upload_dir: Path
    ollama_base_url: str
    top_k: int
    request_timeout_seconds: int
    embedding_batch_size: int
    chat_history_path: Path
    chat_history_max_messages: int


def _get_path(env_value: str | None, default_path: Path) -> Path:
    """Resolve a filesystem path from an environment value or default."""

    if env_value:
        return Path(env_value)
    return default_path


def get_settings() -> Settings:
    """Load settings from the environment with sane defaults."""

    return Settings(
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        model_name=os.getenv("MODEL_NAME", "llama3.2:3b"),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "").rstrip("/"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        chroma_db_path=_get_path(
            os.getenv("CHROMA_DB_PATH"),
            DATA_DIR / "chroma_db",
        ),
        upload_dir=_get_path(
            os.getenv("UPLOAD_DIR"),
            DATA_DIR / "uploads",
        ),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        top_k=int(os.getenv("TOP_K", "4")),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "180")),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        chat_history_path=_get_path(
            os.getenv("CHAT_HISTORY_PATH"),
            DATA_DIR / "chat_history.json",
        ),
        chat_history_max_messages=int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "12")),
    )


def ensure_directories(settings: Settings) -> None:
    """Create required data directories if they do not exist."""

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_db_path.mkdir(parents=True, exist_ok=True)


__all__ = ["Settings", "get_settings", "ensure_directories"]
