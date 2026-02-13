"""Reusable Streamlit UI components."""
from __future__ import annotations

from typing import List, Optional, Tuple

import streamlit as st


def render_file_uploader(label: str) -> Optional[st.runtime.uploaded_file_manager.UploadedFile]:
    """Render a PDF file uploader and return the uploaded file."""

    return st.file_uploader(label, type=["pdf"], accept_multiple_files=False)


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


def render_chat_message(role: str, content: str) -> None:
    """Render a chat message with Streamlit's chat UI."""

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
