"""Persistent storage for chat history sessions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import threading
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HistoryStore:
    """Stores and retrieves chat history on disk with session support."""

    path: Path

    def __post_init__(self) -> None:
        """Ensure the history directory exists."""

        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> Dict[str, Any]:
        """Load chat history state from disk, migrating if needed."""

        with self._lock:
            if not self.path.exists():
                return self._default_state()
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                logger.exception("Failed to read chat history")
                return self._default_state()

        return self._migrate(raw)

    def save_state(self, state: Dict[str, Any]) -> None:
        """Persist chat history state to disk."""

        with self._lock:
            try:
                self.path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:  # noqa: BLE001
                logger.exception("Failed to write chat history")

    def list_sessions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return sessions sorted by updated time (desc)."""

        sessions = list(state.get("sessions") or [])
        return sorted(sessions, key=lambda item: item.get("updated_at", ""), reverse=True)

    def get_active_session_id(self, state: Dict[str, Any]) -> Optional[str]:
        """Return the active session id, fixing if missing."""

        active_id = state.get("active_session_id")
        if active_id and self.get_session(state, active_id):
            return active_id

        sessions = self.list_sessions(state)
        if sessions:
            state["active_session_id"] = sessions[0]["id"]
            return sessions[0]["id"]

        state["active_session_id"] = None
        return None

    def set_active_session(self, state: Dict[str, Any], session_id: str) -> None:
        """Set the active session id."""

        if self.get_session(state, session_id):
            state["active_session_id"] = session_id

    def get_session(self, state: Dict[str, Any], session_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a session by id."""

        for session in state.get("sessions") or []:
            if session.get("id") == session_id:
                return session
        return None

    def get_session_messages(self, state: Dict[str, Any], session_id: str) -> List[Dict[str, str]]:
        """Return messages for a session."""

        session = self.get_session(state, session_id)
        return list(session.get("messages") or []) if session else []

    def get_session_title(self, state: Dict[str, Any], session_id: str) -> str:
        """Return a session title or fallback."""

        session = self.get_session(state, session_id)
        if not session:
            return "Unknown"
        title = str(session.get("title", "")).strip()
        return title or "New Chat"

    def create_session(self, state: Dict[str, Any], title: Optional[str] = None) -> str:
        """Create a new session and set it active."""

        session_id = str(uuid.uuid4())
        now = self._now()
        session = {
            "id": session_id,
            "title": title or "New Chat",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        sessions = state.get("sessions") or []
        sessions.insert(0, session)
        state["sessions"] = sessions
        state["active_session_id"] = session_id
        return session_id

    def rename_session(self, state: Dict[str, Any], session_id: str, title: str) -> bool:
        """Rename a session."""

        session = self.get_session(state, session_id)
        if not session:
            return False
        cleaned = title.strip()
        if not cleaned:
            return False
        session["title"] = cleaned
        session["updated_at"] = self._now()
        return True

    def delete_session(self, state: Dict[str, Any], session_id: str) -> bool:
        """Delete a session and update active selection."""

        sessions = state.get("sessions") or []
        remaining = [session for session in sessions if session.get("id") != session_id]
        if len(remaining) == len(sessions):
            return False
        state["sessions"] = remaining
        if state.get("active_session_id") == session_id:
            state["active_session_id"] = remaining[0]["id"] if remaining else None
        return True

    def append_message(self, state: Dict[str, Any], session_id: str, role: str, content: str) -> int:
        """Append a message to a session and return its index."""

        session = self.get_session(state, session_id)
        if not session:
            session_id = self.create_session(state)
            session = self.get_session(state, session_id)

        message = {"role": role, "content": content}
        messages = session.get("messages") or []
        messages.append(message)
        session["messages"] = messages
        session["updated_at"] = self._now()

        if role == "user" and (session.get("title") in ("", "New Chat")):
            session["title"] = self._title_from_text(content)

        return len(messages) - 1

    def _default_state(self) -> Dict[str, Any]:
        """Return the default empty state."""

        return {"version": 1, "active_session_id": None, "sessions": []}

    def _migrate(self, data: Any) -> Dict[str, Any]:
        """Migrate old list-based history into session format."""

        if isinstance(data, list):
            state = self._default_state()
            session_id = str(uuid.uuid4())
            now = self._now()
            session = {
                "id": session_id,
                "title": self._title_from_messages(data),
                "created_at": now,
                "updated_at": now,
                "messages": data,
            }
            state["sessions"] = [session]
            state["active_session_id"] = session_id
            return state

        if isinstance(data, dict):
            if "version" not in data:
                data["version"] = 1
            if "sessions" not in data:
                data["sessions"] = []
            if "active_session_id" not in data:
                data["active_session_id"] = None
            return data

        return self._default_state()

    def _title_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Infer a title from the first user message."""

        for item in messages:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role == "user" and content:
                return self._title_from_text(content)
        return "New Chat"

    @staticmethod
    def _title_from_text(text: str) -> str:
        """Trim and shorten a title."""

        cleaned = " ".join(text.split())
        return cleaned[:40] + ("..." if len(cleaned) > 40 else "")

    @staticmethod
    def _now() -> str:
        """Return an ISO timestamp."""

        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = ["HistoryStore"]
