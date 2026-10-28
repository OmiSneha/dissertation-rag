"""
server.py
---------
FastAPI backend that exposes the RAG pipeline over HTTP.
React frontend calls these endpoints.

Endpoints
---------
POST /ingest          multipart PDF upload → ingest into ChromaDB
POST /query           JSON { question, collection_name } → answer
GET  /health          liveness check
"""

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import ingest_document, query as rag_query

app = FastAPI(title="RAG Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    pdf_bytes = await file.read()
    try:
        result = ingest_document(pdf_bytes, file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class QueryRequest(BaseModel):
    question: str
    collection_name: str


@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = rag_query(req.question.strip(), req.collection_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
