"""
Code Documentation Assistant — Streamlit UI

Main application entrypoint. Provides a chat interface for asking
questions about an ingested codebase.
"""

import os
import streamlit as st
import logging

from src.config import (
    log_config,
    OLLAMA_MODEL,
    MODEL_TIER,
    EMBEDDING_MODEL,
)
from src.vector_store import get_vector_store
from src.ingest import ingest_codebase, clone_repo, load_existing_index
from src.query_engine import create_query_engine, query

logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(
    page_title="Code Documentation Assistant",
    page_icon="📖",
    layout="wide",
)


def init_session_state():
    """Initialise session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None
    if "ingested" not in st.session_state:
        st.session_state.ingested = False
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None


def render_sidebar():
    """Render the sidebar with ingestion controls and system info."""
    with st.sidebar:
        st.header("📁 Codebase Ingestion")

        # System info
        st.caption(
            f"**Model:** {OLLAMA_MODEL} ({MODEL_TIER} tier)  \n"
            f"**Embeddings:** {EMBEDDING_MODEL}"
        )

        st.divider()

        # Ingestion source selection
        source = st.radio(
            "Ingestion source:",
            ["GitHub URL", "Local path", "Mounted volume (/data/repos)"],
        )

        if source == "GitHub URL":
            repo_url = st.text_input(
                "Repository URL:",
                placeholder="https://github.com/user/repo",
            )
            repo_path = None
        elif source == "Local path":
            repo_path = st.text_input(
                "Path to codebase:",
                placeholder="/path/to/codebase",
            )
            repo_url = None
        else:
            # Docker Compose mounts repos to /data/repos
            repo_path = "/data/repos"
            repo_url = None
            # List available repos in the mount
            if os.path.exists(repo_path):
                repos = [
                    d for d in os.listdir(repo_path)
                    if os.path.isdir(os.path.join(repo_path, d))
                ]
                if repos:
                    selected = st.selectbox("Available repos:", repos)
                    repo_path = os.path.join(repo_path, selected)
                else:
                    st.warning("No repos found in /data/repos")
            else:
                st.warning("Mount path /data/repos not available")

        # Ingest button
        if st.button("🚀 Ingest Codebase", type="primary", use_container_width=True):
            try:
                with st.spinner("Ingesting codebase..."):
                    # Initialise vector store
                    vs = get_vector_store()
                    st.session_state.vector_store = vs

                    # Clone if GitHub URL
                    if repo_url:
                        actual_path = clone_repo(repo_url)
                    else:
                        actual_path = repo_path

                    if not actual_path or not os.path.exists(actual_path):
                        st.error("Invalid path or failed to clone repository")
                        return

                    # Run ingestion pipeline
                    index = ingest_codebase(actual_path, vs)
                    st.session_state.query_engine = create_query_engine(index)
                    st.session_state.ingested = True
                    st.success(
                        f"Ingested {vs.document_count} chunks. "
                        f"Ready to answer questions!"
                    )
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
                logger.exception("Ingestion error")

        st.divider()

        # Load existing index (if already ingested in a previous session)
        if not st.session_state.ingested:
            if st.button("📂 Load Existing Index", use_container_width=True):
                try:
                    with st.spinner("Loading existing index..."):
                        vs = get_vector_store()
                        if vs.document_count > 0:
                            st.session_state.vector_store = vs
                            index = load_existing_index(vs)
                            st.session_state.query_engine = create_query_engine(index)
                            st.session_state.ingested = True
                            st.success(
                                f"Loaded {vs.document_count} existing chunks."
                            )
                        else:
                            st.warning("No existing index found. Ingest a codebase first.")
                except Exception as e:
                    st.error(f"Failed to load index: {e}")
                    logger.exception("Load index error")


def render_chat():
    """Render the main chat interface."""
    st.title("📖 Code Documentation Assistant")

    if not st.session_state.ingested:
        st.info(
            "👈 Use the sidebar to ingest a codebase first, "
            "then ask questions about it here."
        )

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Source files"):
                    for src in msg["sources"]:
                        st.caption(
                            f"`{src['file_path']}` "
                            f"({src['language']}, score: {src['score']})"
                        )

    # Chat input
    if question := st.chat_input(
        "Ask about the codebase...",
        disabled=not st.session_state.ingested,
    ):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = query(st.session_state.query_engine, question)
                    st.markdown(result["answer"])

                    # Show sources
                    if result["sources"]:
                        with st.expander("📎 Source files"):
                            for src in result["sources"]:
                                st.caption(
                                    f"`{src['file_path']}` "
                                    f"({src['language']}, score: {src['score']})"
                                )

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    logger.exception("Query error")


def main():
    """Main application entrypoint."""
    log_config()
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
