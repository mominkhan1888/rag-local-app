"""Reusable Streamlit UI components."""
from __future__ import annotations

from dataclasses import dataclass
import html
from typing import List, Optional, Tuple

import streamlit as st


def create_upload_progress() -> Tuple[st.delta_generator.DeltaGenerator, st.delta_generator.DeltaGenerator]:
    """Create progress UI elements for long-running operations."""

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    return progress_bar, status_text


def update_upload_progress(
    progress_bar: st.delta_generator.DeltaGenerator,
    status_text: st.delta_generator.DeltaGenerator,
    progress: float,
    message: str,
) -> None:
    """Update upload progress indicators."""

    progress_bar.progress(progress)
    status_text.text(message)


@dataclass(frozen=True)
class ComposerSubmission:
    """Represents a single submit action from the chat composer."""

    submitted: bool
    question: str
    files: List[st.runtime.uploaded_file_manager.UploadedFile]


def render_chat_composer(disabled: bool, upload_nonce: int) -> ComposerSubmission:
    """Render a ChatGPT-style composer with file attachments and text input."""

    with st.form(key=f"chat_composer_form_{upload_nonce}"):
        uploaded_files = st.file_uploader(
            "Attach PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"chat_uploader_{upload_nonce}",
            disabled=disabled,
            label_visibility="collapsed",
        )
        question_col, send_col = st.columns([0.84, 0.16])
        question = question_col.text_input(
            "Ask a question about your PDFs",
            placeholder="Ask a question about your PDFs",
            key=f"chat_question_{upload_nonce}",
            disabled=disabled,
            label_visibility="collapsed",
        )
        submitted = send_col.form_submit_button("Send", disabled=disabled, use_container_width=True)
        files = uploaded_files or []
        if files:
            st.caption("Attachments: " + ", ".join(file.name for file in files))

    return ComposerSubmission(
        submitted=submitted,
        question=question.strip(),
        files=list(files),
    )


def render_chat_message(role: str, content: str) -> None:
    """Render a chat message with Streamlit's chat UI."""

    if role not in {"user", "assistant"}:
        escaped_content = html.escape(content).replace("\n", "<br>")
        st.markdown(
            f"<div class='chat-system-message'>{escaped_content}</div>",
            unsafe_allow_html=True,
        )
        return

    with st.chat_message(role):
        st.markdown(content)


def render_pdf_list(pdfs: List[str]) -> Optional[str]:
    """Render the PDF list and return the name of a PDF to delete, if any."""

    if not pdfs:
        st.caption("No PDFs indexed yet.")
        return None

    delete_target = None
    for pdf_name in pdfs:
        cols = st.columns([0.75, 0.25])
        cols[0].write(pdf_name)
        if cols[1].button("Delete", key=f"delete_{pdf_name}"):
            delete_target = pdf_name
    return delete_target


def render_loading_spinner(label: str) -> st.delta_generator.DeltaGenerator:
    """Render a loading spinner with a label."""

    return st.spinner(label)
