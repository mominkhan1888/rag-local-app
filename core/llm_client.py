"""Ollama streaming client for local LLM inference."""
from __future__ import annotations

import json
import logging
from typing import Dict, Iterator, List

import requests

logger = logging.getLogger(__name__)


class BaseLLMClient:
    """Shared prompt builder for LLM clients."""

    def build_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, str]],
        history: List[Dict[str, str]] | None = None,
    ) -> str:
        """Build a prompt with context injection for RAG."""

        context_sections = []
        for chunk in context_chunks:
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "unknown")
            page_start = metadata.get("page_start", "?")
            page_end = metadata.get("page_end", "?")
            context_sections.append(
                f"Source: {source} (pages {page_start}-{page_end})\n{chunk.get('text', '')}"
            )

        context_text = "\n\n".join(context_sections) if context_sections else "No context available."

        history_text = ""
        if history:
            formatted = []
            for item in history:
                role = item.get("role", "").strip().lower()
                content = item.get("content", "").strip()
                if not role or not content:
                    continue
                label = "User" if role == "user" else "Assistant"
                formatted.append(f"{label}: {content}")
            if formatted:
                history_text = "Conversation history:\n" + "\n".join(formatted) + "\n\n"

        return (
            "You are a helpful assistant. Use the provided context to answer the question. "
            "If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{context_text}\n\n"
            f"{history_text}"
            f"Question: {question}\n"
            "Answer:"
        )


class OllamaClient(BaseLLMClient):
    """Client for streaming responses from a local Ollama server."""

    def __init__(self, base_url: str, model: str, request_timeout_seconds: int) -> None:
        """Initialize the client with endpoint configuration."""

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.request_timeout_seconds = request_timeout_seconds

    def stream_answer(
        self,
        prompt: str,
        temperature: float = 0.2,
        num_predict: int = 512,
    ) -> Iterator[str]:
        """Stream the model response as it is generated."""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "temperature": temperature,
            "num_predict": num_predict,
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=(5, self.request_timeout_seconds),
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to connect to Ollama")
            raise RuntimeError("Failed to connect to Ollama. Is it running?") from exc

        try:
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    yield data["response"]
                if data.get("done"):
                    break
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ollama streaming failed")
            raise RuntimeError("Ollama streaming failed") from exc


class OpenAICompatClient(BaseLLMClient):
    """Client for streaming responses from OpenAI-compatible APIs."""

    def __init__(self, base_url: str, api_key: str, model: str, request_timeout_seconds: int) -> None:
        """Initialize the client with endpoint configuration."""

        if not base_url:
            raise ValueError("OPENAI_BASE_URL is required for OpenAI-compatible providers.")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI-compatible providers.")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.request_timeout_seconds = request_timeout_seconds

    def stream_answer(
        self,
        prompt: str,
        temperature: float = 0.2,
        num_predict: int = 512,
    ) -> Iterator[str]:
        """Stream a response from an OpenAI-compatible chat completion endpoint."""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": True,
        }
        if num_predict > 0:
            payload["max_tokens"] = num_predict

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
                timeout=(5, self.request_timeout_seconds),
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to connect to OpenAI-compatible API")
            raise RuntimeError("Failed to connect to OpenAI-compatible API") from exc

        try:
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, (bytes, bytearray)) else str(raw_line)
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = payload.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content is None:
                    message = choices[0].get("message") or {}
                    content = message.get("content")
                if content:
                    yield content
        except Exception as exc:  # noqa: BLE001
            logger.exception("OpenAI-compatible streaming failed")
            raise RuntimeError("OpenAI-compatible streaming failed") from exc
