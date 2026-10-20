"""Streamlit chat UI. Run with: streamlit run src/docmind/app.py"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from docmind.cli import DEFAULT_INDEX
from docmind.llm import OllamaChat, OllamaEmbedder
from docmind.rag import DocMind

st.set_page_config(page_title="DocMind", page_icon="📚", layout="wide")

with st.sidebar:
    st.title("DocMind")
    st.caption("Private RAG. Your documents never leave this machine.")
    model = st.text_input("Chat model", "qwen2.5:3b")
    embed_model = st.text_input("Embedding model", "nomic-embed-text")
    k = st.slider("Passages per answer", 2, 12, 5)


@st.cache_resource
def get_mind(model: str, embed_model: str) -> DocMind:
    return DocMind(DEFAULT_INDEX, OllamaEmbedder(embed_model), OllamaChat(model))


mind = get_mind(model, embed_model)

with st.sidebar:
    uploads = st.file_uploader(
        "Add documents", type=["pdf", "md", "txt"], accept_multiple_files=True
    )
    if uploads and st.button("Index uploads", width="stretch"):
        with tempfile.TemporaryDirectory() as tmp, st.spinner("Embedding..."):
            for f in uploads:
                (Path(tmp) / f.name).write_bytes(f.getvalue())
            added = mind.ingest(Path(tmp))
        st.success(f"Indexed {added} new chunks")
    st.metric("Chunks indexed", len(mind.store))

st.session_state.setdefault("messages", [])
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask your documents anything"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        answer = mind.ask(question, k=k, history=st.session_state.messages[:-1])
        text = st.write_stream(answer.tokens)
        if answer.hits:
            with st.expander("Sources"):
                for n, hit in enumerate(answer.hits, 1):
                    st.markdown(f"**[{n}] {hit.chunk.citation}**")
                    st.caption(hit.chunk.text[:600])
    st.session_state.messages.append({"role": "assistant", "content": text})
