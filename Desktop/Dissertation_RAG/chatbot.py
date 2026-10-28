"""
chatbot.py
----------
Interactive document Q&A chatbot.
  - Ingest a PDF once, then ask unlimited questions.
  - Retrieval-adequacy check abstains when evidence is insufficient.
  - If a trained classifier is available (results/), it scores each answer.

Usage:
    python chatbot.py --pdf path/to/document.pdf [--reingest]
"""

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd

from rag_pipeline import ask, ingest_pdf, COLLECTION_NAME
from feature_extraction import extract_features

RESULTS_DIR = Path("results")


def load_classifier():
    """Try to load the best available trained classifier (XGBoost > RF > LR)."""
    for name in ["xgboost", "randomforest", "logisticregression"]:
        path = RESULTS_DIR / f"{name}.pkl"
        if path.exists():
            scaler_path = RESULTS_DIR / "scaler.pkl"
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            clf = joblib.load(path)
            return name, clf, scaler
    return None, None, None


FEATURE_COLS = [
    "semantic_similarity",
    "lexical_precision",
    "lexical_recall",
    "lexical_f1",
    "nli_entailment",
    "atomic_claim_support_ratio",
]


def classify_answer(question: str, context: str, answer: str, clf, scaler) -> dict | None:
    """Run the trained classifier on the generated answer. Returns result dict or None."""
    try:
        feats = extract_features(question, context, answer)
        X = [[feats[c] for c in FEATURE_COLS]]
        if scaler:
            import numpy as np
            X = scaler.transform(X)
        pred = int(clf.predict(X)[0])
        prob = None
        if hasattr(clf, "predict_proba"):
            prob = float(clf.predict_proba(X)[0][1])
        return {"prediction": pred, "probability": prob, "features": feats}
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="RAG Document Chatbot")
    parser.add_argument("--pdf", required=True, help="Path to PDF document")
    parser.add_argument("--reingest", action="store_true", help="Re-ingest the PDF even if already stored")
    args = parser.parse_args()

    pdf_path = args.pdf
    if not Path(pdf_path).exists():
        print(f"[error] PDF not found: {pdf_path}")
        return

    # Ingest if needed
    import chromadb
    chroma = chromadb.PersistentClient(path="./chroma_db")
    collections = [c.name for c in chroma.list_collections()]
    if COLLECTION_NAME not in collections or args.reingest:
        print(f"[chatbot] Ingesting {pdf_path} ...")
        n = ingest_pdf(pdf_path)
        print(f"[chatbot] Ingested {n} chunks.")
    else:
        print(f"[chatbot] Using existing ChromaDB collection '{COLLECTION_NAME}'.")

    clf_name, clf, scaler = load_classifier()
    if clf:
        print(f"[chatbot] Loaded {clf_name} classifier for hallucination detection.")
    else:
        print("[chatbot] No trained classifier found. Run train_evaluate.py first for hallucination scoring.")

    print("\n" + "="*60)
    print("  RAG Document Chatbot")
    print("  Type 'exit' to quit.")
    print("="*60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ("exit", "quit", "q"):
            print("\nGoodbye!")
            break

        if not question:
            continue

        result = ask(question)

        print(f"\nAI: {result['answer']}")

        if result["adequate"]:
            print(f"\n  [Retrieval] Best chunk distance: {result['distances'][0]:.4f}")
            if clf:
                context_joined = "\n\n".join(result["retrieved_chunks"])
                clf_result = classify_answer(question, context_joined, result["answer"], clf, scaler)
                if "error" in clf_result:
                    print(f"  [Classifier] Error: {clf_result['error']}")
                else:
                    pred_label = "HALLUCINATED" if clf_result["prediction"] == 1 else "SUPPORTED"
                    prob_str = f" (confidence: {clf_result['probability']:.2f})" if clf_result["probability"] is not None else ""
                    print(f"  [Classifier] {pred_label}{prob_str}")
        else:
            print("  [Retrieval] Insufficient evidence — answer withheld.")


if __name__ == "__main__":
    main()
