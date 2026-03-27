from __future__ import annotations

import streamlit as st

from src.veridocs import VeriDocsPipeline


st.set_page_config(page_title="VeriDocs", page_icon="📚", layout="wide")
st.title("VeriDocs: Verified Document Q&A and Study Notes")
st.caption("Class-ready RAG demo using Hugging Face + retrieval + verification")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = VeriDocsPipeline()

pipeline: VeriDocsPipeline = st.session_state.pipeline

if not hasattr(pipeline, "is_model_ready"):
    st.session_state.pipeline = VeriDocsPipeline()
    pipeline = st.session_state.pipeline

if pipeline.is_model_ready():
    st.success("Model status: Ready")
else:
    st.info("Model status: Not loaded yet. It will load on first answer request.")

with st.sidebar:
    st.header("Index Setup")
    docs_path = st.text_input("Documents path", value="data/sample_docs")
    if st.button("Ingest Documents", use_container_width=True):
        with st.spinner("Indexing documents..."):
            count = pipeline.ingest_documents(docs_path)
        if count > 0:
            st.success(f"Indexed {count} chunks.")
        else:
            st.warning("No supported documents found.")

qa_tab, notes_tab = st.tabs(["Q&A", "Study Notes"])

with qa_tab:
    st.subheader("Ask a question")
    question = st.text_area("Question", placeholder="What are the main project goals?")
    if st.button("Generate Verified Answer", use_container_width=True):
        with st.spinner("Generating answer (first request may take longer while model downloads/loads)..."):
            result = pipeline.ask(question)
        st.markdown("### Answer")
        st.write(result.answer)
        st.markdown("### Verification")
        st.write(f"Status: **{result.verification_status}**")
        st.write(f"Confidence: **{result.confidence}**")
        st.markdown("### Citations")
        for citation in result.citations:
            st.write(f"- {citation}")
        st.markdown("### Evidence Quotes")
        for quote in result.evidence_quotes:
            st.text(quote)

with notes_tab:
    st.subheader("Generate study notes")
    topic = st.text_input("Topic", placeholder="RAG architecture")
    if st.button("Generate Notes", use_container_width=True):
        with st.spinner("Generating notes (first request may take longer while model downloads/loads)..."):
            result = pipeline.generate_study_notes(topic)
        st.markdown("### Study Notes")
        st.write(result.answer)
        st.write(f"Verification: **{result.verification_status}** | Confidence: **{result.confidence}**")
