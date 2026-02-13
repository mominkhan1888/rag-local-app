"""Centralized configuration settings for the RAG app."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import logging
import os
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
logger = logging.getLogger(__name__)

_SETTING_ALIASES: dict[str, tuple[str, ...]] = {
    "OPENAI_API_KEY": ("OPENROUTER_API_KEY", "GROQ_API_KEY"),
    "OPENAI_BASE_URL": ("OPENROUTER_BASE_URL", "GROQ_BASE_URL"),
    "OPENAI_MODEL": ("OPENROUTER_MODEL", "GROQ_MODEL"),
    "MODEL_NAME": ("OLLAMA_MODEL",),
}


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


def _normalize_key(key: str) -> str:
    """Normalize config keys to uppercase underscore form."""

    return key.strip().replace("-", "_").replace(".", "_").upper()


def _normalize_secrets(secrets: Mapping[str, Any] | None) -> dict[str, str]:
    """Flatten nested Streamlit secrets into uppercase underscore keys."""

    normalized: dict[str, str] = {}
    if not secrets:
        return normalized

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                key_part = _normalize_key(str(child_key))
                next_prefix = f"{prefix}_{key_part}" if prefix else key_part
                _walk(next_prefix, child_value)
            return

        if value is None:
            return
        normalized[prefix] = str(value)

    for root_key, root_value in secrets.items():
        _walk(_normalize_key(str(root_key)), root_value)

    return normalized


def _read_setting(
    key: str,
    default: str,
    secrets: Mapping[str, Any] | None = None,
) -> str:
    """Read a value from secrets first, then environment variables."""

    normalized_key = _normalize_key(key)
    alias_keys = _SETTING_ALIASES.get(normalized_key, ())
    candidates = (normalized_key, *alias_keys)

    for candidate in candidates:
        if secrets and candidate in secrets:
            value = secrets[candidate]
            if value is not None and str(value).strip():
                return str(value)

    for candidate in candidates:
        value = os.getenv(candidate)
        if value is not None and value.strip():
            return value

    return default


def _read_int_setting(
    key: str,
    default: int,
    secrets: Mapping[str, Any] | None = None,
) -> int:
    """Read an integer setting with safe fallback."""

    raw_value = _read_setting(key, str(default), secrets).strip()
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Invalid integer value for %s='%s'. Using default=%s.", key, raw_value, default)
        return default


def is_cloud_runtime(secrets: Mapping[str, Any] | None = None) -> bool:
    """Return True when running in a hosted Streamlit environment."""

    sharing_mode = _read_setting("STREAMLIT_SHARING_MODE", "", secrets).strip().lower()
    if sharing_mode == "streamlit":
        return True

    explicit_cloud_flag = _read_setting("IS_STREAMLIT_CLOUD", "", secrets).strip().lower()
    if explicit_cloud_flag in {"1", "true", "yes"}:
        return True

    home_dir = _read_setting("HOME", "", secrets).strip().lower()
    if home_dir.startswith("/home/adminuser"):
        return True

    hostname = _read_setting("HOSTNAME", "", secrets).strip().lower()
    if "streamlit" in hostname:
        return True

    return False


def _resolve_llm_provider(
    provider: str | None,
    openai_api_key: str,
    secrets: Mapping[str, Any] | None = None,
) -> str:
    """Resolve provider from env with cloud-aware defaults."""

    if provider and provider.strip():
        return provider.strip().lower()

    # If user provided an OpenAI-compatible key but not provider, default to openrouter.
    if openai_api_key.strip():
        return "openrouter"

    if is_cloud_runtime(secrets):
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


def get_settings(secrets: Mapping[str, Any] | None = None) -> Settings:
    """Load settings from the environment with sane defaults."""

    normalized_secrets = _normalize_secrets(secrets)

    openai_api_key = _read_setting("OPENAI_API_KEY", "", normalized_secrets).strip()
    provider = _resolve_llm_provider(
        _read_setting("LLM_PROVIDER", "", normalized_secrets),
        openai_api_key,
        normalized_secrets,
    )

    openai_base_url = _read_setting("OPENAI_BASE_URL", "", normalized_secrets).strip().rstrip("/")
    if not openai_base_url:
        openai_base_url = _default_openai_base_url(provider)

    openai_model = _read_setting("OPENAI_MODEL", "", normalized_secrets).strip()
    if not openai_model:
        openai_model = _default_openai_model(provider)

    return Settings(
        chunk_size=_read_int_setting("CHUNK_SIZE", 1000, normalized_secrets),
        chunk_overlap=_read_int_setting("CHUNK_OVERLAP", 200, normalized_secrets),
        model_name=_read_setting("MODEL_NAME", "llama3.2:3b", normalized_secrets).strip(),
        llm_provider=provider,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        embedding_model=_read_setting("EMBEDDING_MODEL", "all-MiniLM-L6-v2", normalized_secrets).strip(),
        chroma_db_path=_get_path(
            _read_setting("CHROMA_DB_PATH", "", normalized_secrets).strip() or None,
            DATA_DIR / "chroma_db",
        ),
        upload_dir=_get_path(
            _read_setting("UPLOAD_DIR", "", normalized_secrets).strip() or None,
            DATA_DIR / "uploads",
        ),
        ollama_base_url=_read_setting("OLLAMA_BASE_URL", "http://localhost:11434", normalized_secrets).strip(),
        top_k=_read_int_setting("TOP_K", 4, normalized_secrets),
        request_timeout_seconds=_read_int_setting("REQUEST_TIMEOUT_SECONDS", 180, normalized_secrets),
        embedding_batch_size=_read_int_setting("EMBEDDING_BATCH_SIZE", 32, normalized_secrets),
        chat_history_path=_get_path(
            _read_setting("CHAT_HISTORY_PATH", "", normalized_secrets).strip() or None,
            DATA_DIR / "chat_history.json",
        ),
        chat_history_max_messages=_read_int_setting("CHAT_HISTORY_MAX_MESSAGES", 12, normalized_secrets),
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
    "is_cloud_runtime",
    "validate_settings",
    "ensure_directories",
]
