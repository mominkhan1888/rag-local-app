"""Centralized configuration settings for the RAG app."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

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


def _load_dotenv_if_available() -> None:
    """Load variables from .env when python-dotenv is installed."""

    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return
    load_dotenv()


_load_dotenv_if_available()


def _get_path(env_value: str | None, default_path: Path) -> Path:
    """Resolve a filesystem path from an environment value or default."""

    if env_value:
        return Path(env_value)
    return default_path


def _resolve_llm_provider(provider: str | None) -> str:
    """Resolve provider from env with cloud-aware defaults."""

    if provider and provider.strip():
        return provider.strip().lower()
    if os.getenv("STREAMLIT_SHARING_MODE", "").strip().lower() == "streamlit":
        return "openrouter"
    return "ollama"


def _default_openai_base_url(provider: str) -> str:
    """Return default base URL for known OpenAI-compatible providers."""

    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    if provider == "groq":
        return "https://api.groq.com/openai/v1"
    return ""


def _default_openai_model(provider: str) -> str:
    """Return default model for known OpenAI-compatible providers."""

    if provider == "openrouter":
        return "openrouter/free"
    if provider == "groq":
        return "llama-3.1-8b-instant"
    return ""


def get_settings() -> Settings:
    """Load settings from the environment with sane defaults."""

    provider = _resolve_llm_provider(os.getenv("LLM_PROVIDER"))
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
    if not openai_base_url:
        openai_base_url = _default_openai_base_url(provider)

    openai_model = os.getenv("OPENAI_MODEL", "").strip()
    if not openai_model:
        openai_model = _default_openai_model(provider)

    return Settings(
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        model_name=os.getenv("MODEL_NAME", "llama3.2:3b"),
        llm_provider=provider,
        openai_base_url=openai_base_url,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=openai_model,
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


def validate_settings(settings: Settings) -> None:
    """Validate required configuration for the selected provider."""

    provider = settings.llm_provider.strip().lower()
    if provider == "ollama":
        missing: list[str] = []
        if not settings.ollama_base_url.strip():
            missing.append("OLLAMA_BASE_URL")
        if not settings.model_name.strip():
            missing.append("MODEL_NAME")
        if missing:
            raise ValueError(
                f"Missing required settings for provider 'ollama': {', '.join(missing)}"
            )
        return

    missing = []
    if not settings.openai_base_url.strip():
        missing.append("OPENAI_BASE_URL")
    if not settings.openai_api_key.strip():
        missing.append("OPENAI_API_KEY")
    if not settings.openai_model.strip():
        missing.append("OPENAI_MODEL")

    if missing:
        raise ValueError(
            f"Missing required settings for provider '{provider}': {', '.join(missing)}"
        )


def ensure_directories(settings: Settings) -> None:
    """Create required data directories if they do not exist."""

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_db_path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "Settings",
    "get_settings",
    "validate_settings",
    "ensure_directories",
]
