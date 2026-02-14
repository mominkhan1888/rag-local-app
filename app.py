"""Streamlit entry point for the local RAG application."""
from __future__ import annotations

from pathlib import Path
import logging
import queue
import time

import streamlit as st

from config.settings import (
    Settings,
    ensure_directories,
    get_settings,
    is_cloud_runtime,
    validate_settings,
)
from core.llm_client import check_ollama_health
from services.history_store import HistoryStore
from services.queue_manager import QueueManager, RequestHandle
from services.rag_service import RAGService
from ui.components import (
    create_upload_progress,
    render_chat_composer,
    render_chat_message,
    render_pdf_list,
    update_upload_progress,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


CACHE_VERSION = "v3"


@st.cache_resource
def init_services(cache_version: str, settings: Settings) -> tuple:
    """Initialize shared services for all Streamlit sessions."""

    ensure_directories(settings)
    rag_service = RAGService(settings)
    queue_manager = QueueManager()
    history_store = HistoryStore(settings.chat_history_path)
    _ = cache_version
    return settings, rag_service, queue_manager, history_store


def load_css() -> None:
    """Inject custom CSS into the Streamlit app."""

    css_path = Path(__file__).parent / "ui" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def load_streamlit_secrets() -> dict:
    """Load Streamlit secrets into a plain dictionary."""

    try:
        if hasattr(st.secrets, "to_dict"):
            return st.secrets.to_dict()  # type: ignore[no-any-return]
        return {key: st.secrets[key] for key in st.secrets.keys()}
    except Exception:  # noqa: BLE001
        return {}


def get_cached_ollama_health(base_url: str, api_key: str = "") -> tuple[bool, str]:
    """Check Ollama health with short-lived caching per session."""

    now = time.time()
    cached = st.session_state.get("ollama_health")
    if (
        isinstance(cached, dict)
        and cached.get("base_url") == base_url
        and cached.get("api_key_hash") == hash(api_key)
        and (now - float(cached.get("checked_at", 0))) < 15
    ):
        return bool(cached.get("ok")), str(cached.get("reason", ""))

    ok, reason = check_ollama_health(base_url, api_key=api_key)
    st.session_state["ollama_health"] = {
        "base_url": base_url,
        "api_key_hash": hash(api_key),
        "checked_at": now,
        "ok": ok,
        "reason": reason,
    }
    return ok, reason


def guard_ollama_runtime(settings: Settings, running_in_cloud: bool) -> None:
    """Show actionable guidance when Ollama is unreachable."""

    if settings.llm_provider.lower() != "ollama":
        return

    is_healthy, reason = get_cached_ollama_health(settings.ollama_base_url, settings.ollama_api_key)
    if is_healthy:
        return

    if running_in_cloud:
        st.error(
            "This deployment is configured for Ollama, but Streamlit Cloud cannot reach your local Ollama service."
        )
        st.info(
            "Set cloud secrets for an internet LLM provider:\n\n"
            "LLM_PROVIDER=openrouter\n"
            "OPENAI_BASE_URL=https://openrouter.ai/api/v1\n"
            "OPENAI_API_KEY=YOUR_KEY\n"
            "OPENAI_MODEL=openrouter/free"
        )
        st.stop()

    st.warning(
        f"Ollama is not reachable at {settings.ollama_base_url}. "
        "Start Ollama (`ollama serve`) and ensure the model is pulled."
    )
    if reason:
        st.caption(f"Connection detail: {reason}")


def ensure_session_state(history_store: HistoryStore) -> None:
    """Ensure required session state keys exist."""

    if not hasattr(history_store, "load_state"):
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

    if "history_state" not in st.session_state:
        st.session_state["history_state"] = history_store.load_state()
    if "pending_handle" not in st.session_state:
        st.session_state["pending_handle"] = None
    if "pending_assistant_index" not in st.session_state:
        st.session_state["pending_assistant_index"] = None
    if "pending_session_id" not in st.session_state:
        st.session_state["pending_session_id"] = None
    if "composer_busy" not in st.session_state:
        st.session_state["composer_busy"] = False
    if "composer_upload_nonce" not in st.session_state:
        st.session_state["composer_upload_nonce"] = 0


def get_history_state(history_store: HistoryStore) -> dict:
    """Return chat history state, ensuring session state exists."""

    ensure_session_state(history_store)
    return st.session_state.get("history_state", {})


def persist_history_state(history_store: HistoryStore, state: dict) -> None:
    """Persist and cache history state."""

    st.session_state["history_state"] = state
    history_store.save_state(state)


def stream_pending_response(
    handle: RequestHandle,
    queue_manager: QueueManager,
    assistant_index: int,
    session_id: str,
    history_store: HistoryStore,
) -> None:
    """Stream tokens from a queued request into the UI."""

    ensure_session_state(history_store)

    status_placeholder = st.empty()
    while True:
        position = queue_manager.get_position(handle.request_id)
        if position == 0:
            status_placeholder.empty()
            break
        if position == -1:
            if handle.done_event.is_set() or not handle.result_queue.empty():
                status_placeholder.empty()
                break
            status_placeholder.error("Request expired. Please retry.")
            st.session_state["pending_handle"] = None
            st.session_state["pending_assistant_index"] = None
            st.session_state["pending_session_id"] = None
            return
        status_placeholder.info(f"Queue position: {position}")
        time.sleep(0.5)

    with st.chat_message("assistant"):
        content_placeholder = st.empty()
        content = ""
        while True:
            try:
                token = handle.result_queue.get(timeout=0.5)
            except queue.Empty:
                if handle.done_event.is_set():
                    break
                continue
            if token is None:
                break
            content += token
            content_placeholder.markdown(content)

    if not handle.error_queue.empty():
        error_message = handle.error_queue.get()
        st.error(f"Error: {error_message}")

    state = get_history_state(history_store)
    session = history_store.get_session(state, session_id)
    if session:
        messages = session.get("messages") or []
        if 0 <= assistant_index < len(messages):
            messages[assistant_index]["content"] = content
            session["messages"] = messages
    persist_history_state(history_store, state)

    st.session_state["pending_handle"] = None
    st.session_state["pending_assistant_index"] = None
    st.session_state["pending_session_id"] = None


def render_chat_history_sidebar(history_store: HistoryStore) -> None:
    """Render chat history controls in the sidebar."""

    st.subheader("Chat History")

    state = get_history_state(history_store)
    sessions = history_store.list_sessions(state)

    if st.button("New Chat"):
        history_store.create_session(state)
        persist_history_state(history_store, state)
        st.rerun()

    if not sessions:
        st.caption("No chats yet. Click New Chat or ask a question.")
        return

    active_id = history_store.get_active_session_id(state)
    session_ids = [session["id"] for session in sessions]
    if active_id not in session_ids:
        active_id = session_ids[0]
        history_store.set_active_session(state, active_id)
        persist_history_state(history_store, state)

    selected_id = st.radio(
        "Saved Chats",
        session_ids,
        index=session_ids.index(active_id),
        format_func=lambda sid: history_store.get_session_title(state, sid),
    )

    if selected_id != active_id:
        history_store.set_active_session(state, selected_id)
        persist_history_state(history_store, state)
        st.rerun()

    active_session = history_store.get_session(state, selected_id)
    if not active_session:
        return

    st.caption(f"Last updated: {active_session.get('updated_at', '')}")
    rename_value = st.text_input(
        "Rename chat",
        value=active_session.get("title", ""),
        key=f"rename_{selected_id}",
    )
    rename_col, delete_col = st.columns([0.6, 0.4])
    if rename_col.button("Save name", key=f"save_name_{selected_id}"):
        if history_store.rename_session(state, selected_id, rename_value):
            persist_history_state(history_store, state)
            st.rerun()
    if delete_col.button("Delete chat", key=f"delete_chat_{selected_id}"):
        history_store.delete_session(state, selected_id)
        persist_history_state(history_store, state)
        st.rerun()


def main() -> None:
    """Render the Streamlit application."""

    st.set_page_config(page_title="Local RAG", layout="wide")
    load_css()

    secrets = load_streamlit_secrets()
    settings = get_settings(secrets=secrets)
    running_in_cloud = is_cloud_runtime(secrets=secrets)
    try:
        validate_settings(settings)
    except ValueError as exc:
        st.error("Configuration error")
        st.error(str(exc))
        st.info(
            "For Streamlit Cloud, set LLM_PROVIDER=groq/openrouter and provide "
            "OPENAI_BASE_URL, OPENAI_MODEL, and OPENAI_API_KEY "
            "(aliases: GROQ_API_KEY or OPENROUTER_API_KEY)."
        )
        st.stop()

    try:
        settings, rag_service, queue_manager, history_store = init_services(CACHE_VERSION, settings)
    except Exception as exc:  # noqa: BLE001
        st.error("Startup error")
        st.error(str(exc))
        st.info(
            "If this is Streamlit Cloud, use an internet provider (OpenRouter/Groq) and reboot after updating secrets."
        )
        st.stop()

    ensure_session_state(history_store)
    guard_ollama_runtime(settings, running_in_cloud)

    with st.sidebar:
        st.header("PDF Library")
        st.subheader("Indexed PDFs")
        try:
            pdfs = rag_service.list_pdfs()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            pdfs = []

        delete_target = render_pdf_list(pdfs)
        if delete_target:
            try:
                result = rag_service.delete_pdf(delete_target)
                deleted_chunks = result.get("deleted_chunks", 0)
                file_deleted = result.get("file_deleted", False)
                file_missing = result.get("file_missing", False)

                message = f"Deleted {deleted_chunks} chunks from {delete_target}."
                if file_deleted:
                    message += " Removed uploaded file."
                if file_missing:
                    message += " Uploaded file not found."
                if deleted_chunks == 0 and not file_deleted:
                    st.info(message)
                else:
                    st.success(message)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

        st.divider()
        render_chat_history_sidebar(history_store)

    st.title("Local RAG Assistant")
    st.caption("Attach PDFs in the composer, then ask questions in the same send flow.")

    chat_messages_container = st.container()
    composer_status_container = st.container()
    composer_container = st.container()

    pending_handle = st.session_state.get("pending_handle")
    composer_disabled = bool(st.session_state.get("composer_busy")) or pending_handle is not None

    with composer_container:
        submission = render_chat_composer(
            disabled=composer_disabled,
            upload_nonce=st.session_state.get("composer_upload_nonce", 0),
        )

    if submission.submitted:
        if not submission.question and not submission.files:
            composer_status_container.warning("Attach at least one PDF or enter a question.")
        else:
            should_rerun = False
            st.session_state["composer_busy"] = True
            try:
                state = get_history_state(history_store)
                active_id = history_store.get_active_session_id(state)
                if not active_id:
                    active_id = history_store.create_session(state)

                if submission.question:
                    history_store.append_message(state, active_id, "user", submission.question)

                if submission.files:
                    upload_payloads: list[tuple[str, bytes]] = []
                    for uploaded_file in submission.files:
                        try:
                            upload_payloads.append((uploaded_file.name, uploaded_file.getvalue()))
                        except Exception as exc:  # noqa: BLE001
                            history_store.append_message(
                                state,
                                active_id,
                                "system",
                                f'Failed to read "{uploaded_file.name}": {exc}',
                            )

                    if upload_payloads:
                        with composer_status_container:
                            progress_bar, status_text = create_upload_progress()
                            results = rag_service.index_uploaded_files(
                                upload_payloads,
                                progress_callback=lambda _name, progress, message: update_upload_progress(
                                    progress_bar,
                                    status_text,
                                    progress,
                                    message,
                                ),
                            )

                        indexed_count = 0
                        failed_count = 0
                        for result in results:
                            if result.success:
                                indexed_count += 1
                                history_store.append_message(
                                    state,
                                    active_id,
                                    "system",
                                    f'Indexed "{result.file_name}" ({result.chunks_indexed} chunks).',
                                )
                            else:
                                failed_count += 1
                                history_store.append_message(
                                    state,
                                    active_id,
                                    "system",
                                    f'Failed to index "{result.file_name}": {result.error}',
                                )

                        if indexed_count:
                            composer_status_container.success(f"Indexed {indexed_count} PDF file(s).")
                        if failed_count:
                            composer_status_container.warning(
                                f"{failed_count} PDF file(s) failed indexing. See chat timeline for details."
                            )

                if submission.question:
                    try:
                        available_pdfs = rag_service.list_pdfs()
                    except Exception as exc:  # noqa: BLE001
                        history_store.append_message(
                            state,
                            active_id,
                            "assistant",
                            f"Error loading indexed PDFs: {exc}",
                        )
                    else:
                        if not available_pdfs:
                            history_store.append_message(
                                state,
                                active_id,
                                "assistant",
                                "No PDFs are indexed yet. Attach a valid PDF and send again.",
                            )
                        else:
                            assistant_index = history_store.append_message(state, active_id, "assistant", "")
                            persist_history_state(history_store, state)
                            try:
                                handle = queue_manager.enqueue(
                                    generator_factory=lambda: rag_service.stream_answer(
                                        submission.question,
                                        settings.top_k,
                                        history_store.get_session_messages(state, active_id),
                                    ),
                                    timeout_seconds=settings.request_timeout_seconds,
                                )
                            except Exception as exc:  # noqa: BLE001
                                messages = history_store.get_session_messages(state, active_id)
                                if 0 <= assistant_index < len(messages):
                                    messages[assistant_index]["content"] = (
                                        f"Error while queuing request: {exc}"
                                    )
                                    session = history_store.get_session(state, active_id)
                                    if session is not None:
                                        session["messages"] = messages
                            else:
                                st.session_state["pending_handle"] = handle
                                st.session_state["pending_assistant_index"] = assistant_index
                                st.session_state["pending_session_id"] = active_id

                persist_history_state(history_store, state)
                st.session_state["composer_upload_nonce"] = st.session_state.get("composer_upload_nonce", 0) + 1
                should_rerun = True
            finally:
                st.session_state["composer_busy"] = False

            if should_rerun:
                st.rerun()

    with chat_messages_container:
        state = get_history_state(history_store)
        active_id = history_store.get_active_session_id(state)
        active_session = history_store.get_session(state, active_id) if active_id else None
        messages = list(active_session.get("messages") or []) if active_session else []

        pending_index = st.session_state.get("pending_assistant_index")
        pending_session_id = st.session_state.get("pending_session_id")
        for idx, message in enumerate(messages):
            if pending_session_id == active_id and pending_index is not None and idx == pending_index:
                continue
            render_chat_message(message["role"], message["content"])

        pending_handle = st.session_state.get("pending_handle")
        if pending_handle and pending_index is not None and pending_session_id == active_id:
            stream_pending_response(
                pending_handle,
                queue_manager,
                pending_index,
                pending_session_id,
                history_store,
            )


if __name__ == "__main__":
    main()
