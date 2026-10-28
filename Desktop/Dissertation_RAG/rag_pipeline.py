"""
rag_pipeline.py
----------------
Core RAG pipeline functions (no UI logic here).

Steps
-----
1. ingest_document(pdf_bytes, filename)
   - Extract text from PDF bytes
   - Split into overlapping chunks
   - Embed each chunk with Gemini embedding model
   - Store in ChromaDB (persistent, per-document collection)

2. query(question, collection_name)
   - Embed the question
   - Retrieve top-K chunks by cosine similarity (L2 distance)
   - Retrieval-adequacy check against threshold
     * distance < THRESHOLD  →  evidence is adequate  →  generate answer
     * distance >= THRESHOLD →  evidence is insufficient → abstain
   - If adequate: send question + context to Gemini, return answer
   - Return a result dict with all intermediate data for display in UI

ChromaDB L2 distance notes
---------------------------
ChromaDB's default metric is L2 (squared Euclidean).
For normalised unit vectors the relationship to cosine similarity is:
    L2² = 2(1 - cosine_similarity)
So L2 = 0 means identical, L2 = √2 ≈ 1.41 means orthogonal.
A threshold of 0.9 (from the working reference project) corresponds to
roughly cosine_similarity > 0.595, which is a reasonable "related" cut-off.
"""

import hashlib
import io
import os

import chromadb
from dotenv import load_dotenv
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

# ── configuration ──────────────────────────────────────────────────────────────
CHROMA_DB_PATH   = "./chroma_db"
EMBED_MODEL      = "gemini-embedding-001"
GENERATE_MODEL   = "gemini-3.8-flash"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 100
TOP_K            = 3

# L2 distance threshold.
# Chunks with distance BELOW this are considered adequate evidence.
# Chunks with distance AT OR ABOVE this are considered too weak.
ADEQUACY_THRESHOLD = 0.9


# ── Gemini client (lazy) ───────────────────────────────────────────────────────
_client: genai.Client | None = None

def _gemini() -> genai.Client:
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        _client = genai.Client(api_key=key)
    return _client


# ── step 1: ingest ─────────────────────────────────────────────────────────────
def ingest_document(pdf_bytes: bytes, filename: str) -> dict:
    """
    Ingest a PDF (supplied as raw bytes).
    Returns { collection_name, num_chunks, preview_chunks }.
    """
    # Derive a stable collection name from the filename
    safe_name = "".join(c if c.isalnum() else "_" for c in filename.replace(".pdf", ""))
    collection_name = f"doc_{safe_name[:40]}"

    # 1a. Extract text
    reader = PdfReader(io.BytesIO(pdf_bytes))
    raw_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            raw_text += page_text + "\n"

    if not raw_text.strip():
        raise ValueError("No text could be extracted from the uploaded PDF.")

    # 1b. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(raw_text)

    # 1c. Embed
    client = _gemini()
    embeddings = []
    for chunk in chunks:
        resp = client.models.embed_content(model=EMBED_MODEL, contents=chunk)
        embeddings.append(resp.embeddings[0].values)

    # 1d. Store in ChromaDB (delete+recreate so re-upload is clean)
    chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    existing = [c.name for c in chroma.list_collections()]
    if collection_name in existing:
        chroma.delete_collection(collection_name)

    collection = chroma.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "l2"},
    )
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings)

    return {
        "collection_name": collection_name,
        "num_chunks": len(chunks),
        "preview_chunks": chunks[:3],
    }


# ── step 2: query ──────────────────────────────────────────────────────────────
def query(question: str, collection_name: str) -> dict:
    """
    Ask a question against an ingested document collection.

    Returns a dict with:
        adequate      bool   – whether evidence passed the threshold
        answer        str    – generated answer (or abstention message)
        chunks        list   – retrieved chunk texts
        distances     list   – L2 distances for each retrieved chunk
        threshold     float  – the configured ADEQUACY_THRESHOLD
        best_distance float  – lowest (best) distance found
    """
    client = _gemini()
    chroma  = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma.get_collection(name=collection_name)

    # 2a. Embed the question
    resp = client.models.embed_content(model=EMBED_MODEL, contents=question)
    q_embedding = resp.embeddings[0].values

    # 2b. Retrieve top-K chunks by L2 distance
    n = min(TOP_K, collection.count())
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n,
        include=["documents", "distances"],
    )
    chunks    = results["documents"][0]
    distances = results["distances"][0]
    best_dist = distances[0] if distances else float("inf")

    # 2c. Retrieval-adequacy check
    #     distance < threshold  →  adequate
    #     distance >= threshold →  abstain
    if best_dist >= ADEQUACY_THRESHOLD:
        return {
            "adequate":      False,
            "answer":        (
                "I cannot provide a reliable answer. "
                "The retrieved context does not contain sufficient evidence "
                f"for this question (best similarity distance: {best_dist:.4f}, "
                f"threshold: {ADEQUACY_THRESHOLD})."
            ),
            "chunks":        chunks,
            "distances":     distances,
            "threshold":     ADEQUACY_THRESHOLD,
            "best_distance": best_dist,
        }

    # 2d. Generate answer via Gemini (context-only prompt)
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the \
information in the context below. Do not use any outside knowledge.
If the answer is not in the context, say: \
"I could not find the answer in the provided document."

Context:
{context}

Question: {question}

Answer:"""

    gen_resp = client.models.generate_content(model=GENERATE_MODEL, contents=prompt)
    answer   = gen_resp.text.strip()

    return {
        "adequate":      True,
        "answer":        answer,
        "chunks":        chunks,
        "distances":     distances,
        "threshold":     ADEQUACY_THRESHOLD,
        "best_distance": best_dist,
    }
