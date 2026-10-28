"""
rag_pipeline.py
---------------
Document-based RAG pipeline:
  1. Ingest a PDF (extract, chunk, embed, store in ChromaDB)
  2. Query: embed question → retrieve chunks via cosine similarity
  3. Retrieval-adequacy check: abstain when evidence is insufficient
  4. Answer generation via Google Gemini API (context-only prompt)
"""

import os
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from google import genai

load_dotenv()

# ── constants ──────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "dissertation_docs"
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.0-flash"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
# ChromaDB returns L2 distances; a lower value means more similar.
# 1.0 is a practical cutoff for "context is too weak to trust".
ADEQUACY_THRESHOLD = 1.0


# ── clients (lazy-initialised) ─────────────────────────────────────────────────
_gemini_client: Optional[genai.Client] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in .env")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "l2"},
        )
    return _collection


# ── ingestion ──────────────────────────────────────────────────────────────────
def ingest_pdf(pdf_path: str) -> int:
    """
    Extract text from a PDF, split into chunks, embed with Gemini,
    store in ChromaDB.  Returns the number of chunks stored.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1. Extract text
    reader = PdfReader(str(path))
    raw_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            raw_text += page_text + "\n"

    if not raw_text.strip():
        raise ValueError("No text could be extracted from the PDF.")

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(raw_text)
    print(f"[ingest] Extracted {len(raw_text):,} chars → {len(chunks)} chunks")

    # 3. Embed
    client = _get_gemini()
    embeddings = []
    for chunk in chunks:
        resp = client.models.embed_content(model=EMBEDDING_MODEL, contents=chunk)
        embeddings.append(resp.embeddings[0].values)

    # 4. Store
    collection = _get_collection()
    doc_prefix = path.stem[:20]
    ids = [f"{doc_prefix}_{uuid.uuid4().hex[:8]}" for _ in chunks]

    collection.add(ids=ids, documents=chunks, embeddings=embeddings)
    print(f"[ingest] Stored {len(chunks)} chunks in ChromaDB (total: {collection.count()})")
    return len(chunks)


# ── retrieval ──────────────────────────────────────────────────────────────────
def retrieve(question: str) -> tuple[list[str], list[float]]:
    """
    Embed the question and return (chunks, distances) for the top-K results.
    Distances are L2; smaller = more similar.
    """
    client = _get_gemini()
    resp = client.models.embed_content(model=EMBEDDING_MODEL, contents=question)
    q_embedding = resp.embeddings[0].values

    collection = _get_collection()
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "distances"],
    )
    return results["documents"][0], results["distances"][0]


# ── adequacy check ─────────────────────────────────────────────────────────────
def is_evidence_adequate(distances: list[float]) -> bool:
    """
    Return True when the best (lowest-L2) chunk distance is below the
    ADEQUACY_THRESHOLD, indicating sufficient evidence.
    """
    if not distances:
        return False
    return distances[0] < ADEQUACY_THRESHOLD


# ── answer generation ──────────────────────────────────────────────────────────
def generate_answer(question: str, context: str) -> str:
    """
    Ask Gemini to answer the question using only the supplied context.
    """
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the information
provided in the context below. Do not add information from outside the context.
If the answer cannot be found in the context, say exactly:
"I could not find a supported answer in the retrieved context."

Context:
------------------------------------------------------------
{context}
------------------------------------------------------------

Question: {question}

Answer:"""

    client = _get_gemini()
    response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
    return response.text.strip()


# ── main RAG call ──────────────────────────────────────────────────────────────
def ask(question: str) -> dict:
    """
    Full RAG pipeline for a single question.

    Returns a dict with:
      question, retrieved_chunks, distances, adequate, answer
    """
    chunks, distances = retrieve(question)
    adequate = is_evidence_adequate(distances)

    if not adequate:
        return {
            "question": question,
            "retrieved_chunks": chunks,
            "distances": distances,
            "adequate": False,
            "answer": "I cannot answer reliably: the retrieved context does not contain sufficient evidence.",
        }

    context = "\n\n".join(chunks)
    answer = generate_answer(question, context)

    return {
        "question": question,
        "retrieved_chunks": chunks,
        "distances": distances,
        "adequate": True,
        "answer": answer,
    }
