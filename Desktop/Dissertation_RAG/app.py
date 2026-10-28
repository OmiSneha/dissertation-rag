"""
app.py
------
Streamlit UI for the Dissertation RAG Pipeline.

Pages / flow:
  1. Upload a PDF → ingests it into ChromaDB
  2. Ask questions → retrieval-adequacy check → Gemini answer
  3. Shows retrieved chunks, distances, and whether evidence was adequate

Dataset note (shown in sidebar):
  The RAGTruth QA subset has been selected as the benchmark dataset
  for the hallucination-detection experiments (future work).
"""

import streamlit as st

from rag_pipeline import ADEQUACY_THRESHOLD, ingest_document, query

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline – Hallucination Detection Dissertation",
    page_icon="📄",
    layout="wide",
)

# ── session state defaults ─────────────────────────────────────────────────────
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {question, result}


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG Pipeline")
    st.caption("Dissertation – Mid-Semester Demo")

    st.divider()

    st.subheader("How it works")
    st.markdown("""
1. **Upload** a PDF document
2. Text is extracted, split into chunks, embedded with **Gemini Embeddings**, and stored in **ChromaDB**
3. Your question is embedded and matched to the closest chunks via **cosine similarity**
4. **Retrieval-adequacy check**:
   - L2 distance **< threshold** → evidence adequate → **Gemini generates an answer**
   - L2 distance **≥ threshold** → evidence insufficient → **system abstains**
""")

    st.divider()

    st.subheader("Configuration")
    st.markdown(f"""
| Parameter | Value |
|---|---|
| Embedding model | `gemini-embedding-001` |
| Generation model | `gemini-3.8-flash` |
| Chunk size | 500 chars |
| Chunk overlap | 100 chars |
| Top-K retrieval | 3 chunks |
| Adequacy threshold (L2) | **{ADEQUACY_THRESHOLD}** |
""")

    st.divider()

    st.subheader("Dataset Selection")
    st.info(
        "**RAGTruth QA Subset** selected as benchmark dataset for "
        "hallucination-detection experiments.\n\n"
        "Human-annotated question-context-answer triples; binary labels "
        "will be derived from existing annotations in future work."
    )

    if st.session_state.collection_name:
        st.divider()
        st.success(
            f"**Document loaded**\n\n"
            f"📁 {st.session_state.filename}\n\n"
            f"🗂 {st.session_state.num_chunks} chunks stored"
        )
        if st.button("🗑 Clear document & chat"):
            st.session_state.collection_name = None
            st.session_state.filename = None
            st.session_state.num_chunks = 0
            st.session_state.chat_history = []
            st.rerun()


# ── main area ──────────────────────────────────────────────────────────────────
st.title("📄 RAG-Based Document Q&A")
st.caption(
    "A Machine Learning Approach for Hallucination Detection in "
    "Retrieval-Augmented Document Question-Answering Systems"
)

# ── STEP 1: Upload ─────────────────────────────────────────────────────────────
st.header("Step 1 – Upload Document")

uploaded = st.file_uploader(
    "Upload a PDF to begin",
    type=["pdf"],
    help="The PDF will be chunked, embedded, and stored in ChromaDB.",
)

if uploaded is not None:
    if st.session_state.filename != uploaded.name:
        with st.spinner(
            f"Ingesting **{uploaded.name}** — extracting text, chunking, "
            "embedding with Gemini, storing in ChromaDB …"
        ):
            try:
                result = ingest_document(uploaded.read(), uploaded.name)
                st.session_state.collection_name = result["collection_name"]
                st.session_state.filename = uploaded.name
                st.session_state.num_chunks = result["num_chunks"]
                st.session_state.chat_history = []

                st.success(
                    f"✅ **{uploaded.name}** ingested successfully — "
                    f"**{result['num_chunks']} chunks** stored in ChromaDB."
                )

                with st.expander("Preview first 3 chunks"):
                    for i, chunk in enumerate(result["preview_chunks"], 1):
                        st.markdown(f"**Chunk {i}**")
                        st.text(chunk)

            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")
    else:
        st.info(
            f"✅ **{uploaded.name}** is already loaded "
            f"({st.session_state.num_chunks} chunks). "
            "Ask a question below or upload a different file."
        )


# ── STEP 2: Ask questions ──────────────────────────────────────────────────────
if st.session_state.collection_name:
    st.divider()
    st.header("Step 2 – Ask a Question")

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Enter your question",
            placeholder="e.g. What is the main topic of this document?",
        )
        submitted = st.form_submit_button("Ask", use_container_width=True)

    if submitted and question.strip():
        with st.spinner("Retrieving context and generating answer …"):
            try:
                result = query(question.strip(), st.session_state.collection_name)
                st.session_state.chat_history.append(
                    {"question": question.strip(), "result": result}
                )
            except Exception as e:
                st.error(f"❌ Query failed: {e}")

    # ── Display chat history ───────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.divider()
        st.header("Results")

        for i, entry in enumerate(reversed(st.session_state.chat_history)):
            q = entry["question"]
            r = entry["result"]

            st.markdown(f"### Q{len(st.session_state.chat_history) - i}: {q}")

            # Adequacy badge
            best = r["best_distance"]
            threshold = r["threshold"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Best L2 Distance", f"{best:.4f}")
            col2.metric("Threshold", f"{threshold}")
            col3.metric(
                "Retrieval Adequacy",
                "✅ Adequate" if r["adequate"] else "❌ Insufficient",
            )

            # Answer
            if r["adequate"]:
                st.success(f"**Answer:**\n\n{r['answer']}")
            else:
                st.warning(f"**System abstained:**\n\n{r['answer']}")

            # Retrieved chunks detail
            with st.expander(f"View retrieved chunks and distances"):
                st.markdown(
                    f"*L2 distance < {threshold} = adequate evidence. "
                    f"L2 distance ≥ {threshold} = insufficient evidence.*"
                )
                for j, (chunk, dist) in enumerate(
                    zip(r["chunks"], r["distances"]), 1
                ):
                    adequate_flag = "✅" if dist < threshold else "❌"
                    st.markdown(
                        f"**Chunk {j}** — distance: `{dist:.4f}` {adequate_flag}"
                    )
                    st.text(chunk)
                    if j < len(r["chunks"]):
                        st.divider()

            st.divider()

else:
    st.info("⬆️ Upload a PDF document above to get started.")
