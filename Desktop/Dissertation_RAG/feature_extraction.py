"""
feature_extraction.py
---------------------
Compute four evidence-support features for a (question, context, response) triple.

Feature 1 – Semantic similarity
    Cosine similarity between sentence-transformer embeddings of response and context.

Feature 2 – Lexical overlap (BERTScore-style token F1)
    Token-level precision, recall, F1 between response and context tokens.
    Uses a simple normalized-word approach (no stop-word removal intentionally,
    to stay faithful to what the classifier sees).

Feature 3 – NLI entailment
    Pretrained cross-encoder NLI model (roberta-large-mnli).
    Context is the premise; response is the hypothesis.
    Score = softmax probability of the 'entailment' class.

Feature 4 – Atomic-claim support ratio
    Split the response into sentences (atomic claims).
    For each sentence, use the NLI model to classify whether the context
    entails ('supported'), contradicts, or is neutral ('unsupported').
    Score = supported_count / total_assessable_count.
    Returns 0.0 when no assessable sentences are found.

All models are loaded once and reused.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── model constants ────────────────────────────────────────────────────────────
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
NLI_MODEL_NAME = "cross-encoder/nli-roberta-base"   # lighter than large-mnli, still strong
MAX_NLI_TOKENS = 512  # hard limit per model


# ── singletons ─────────────────────────────────────────────────────────────────
_semantic_model: SentenceTransformer | None = None
_nli_tokenizer: Any | None = None
_nli_model: Any | None = None
_device: str | None = None


def _get_device() -> str:
    global _device
    if _device is None:
        _device = "mps" if torch.backends.mps.is_available() else "cpu"
    return _device


def _get_semantic_model() -> SentenceTransformer:
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME, device=_get_device())
    return _semantic_model


def _get_nli_model():
    global _nli_tokenizer, _nli_model
    if _nli_model is None:
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        _nli_model.eval()
        _nli_model.to(_get_device())
    return _nli_tokenizer, _nli_model


# ── helpers ────────────────────────────────────────────────────────────────────
def _tokenize_words(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _nli_scores(premise: str, hypothesis: str) -> dict[str, float]:
    """
    Returns {'contradiction': p, 'neutral': p, 'entailment': p}
    The label order for cross-encoder/nli-roberta-base is
    [contradiction, neutral, entailment].
    """
    tokenizer, model = _get_nli_model()
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_NLI_TOKENS,
        padding=True,
    ).to(_get_device())

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    # label2id from model config
    id2label = model.config.id2label
    return {id2label[i].lower(): float(p) for i, p in enumerate(probs)}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex (no NLTK dependency)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ── feature functions ──────────────────────────────────────────────────────────
def semantic_similarity(response: str, context: str) -> float:
    """
    Cosine similarity between mean-pooled sentence embeddings.
    Range: [-1, 1] but practically [0, 1] for these models.
    """
    model = _get_semantic_model()
    embeddings = model.encode([response, context], convert_to_numpy=True, show_progress_bar=False)
    a, b = embeddings[0], embeddings[1]
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def lexical_overlap(response: str, context: str) -> tuple[float, float, float]:
    """
    Token-level precision, recall, F1 of response tokens against context tokens.
    Returns (precision, recall, f1).
    """
    resp_tokens = set(_tokenize_words(response))
    ctx_tokens = set(_tokenize_words(context))

    if not resp_tokens or not ctx_tokens:
        return 0.0, 0.0, 0.0

    common = resp_tokens & ctx_tokens
    precision = len(common) / len(resp_tokens)
    recall = len(common) / len(ctx_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def nli_entailment_score(response: str, context: str) -> float:
    """
    Entailment probability using NLI model.
    Context = premise; response = hypothesis.
    Returns probability that context entails response (0–1).
    """
    scores = _nli_scores(premise=context, hypothesis=response)
    return scores.get("entailment", 0.0)


def atomic_claim_support_ratio(response: str, context: str) -> float:
    """
    Split response into atomic sentences, test each against context via NLI.
    A sentence is 'supported' if entailment_prob > max(contradiction_prob, neutral_prob).
    Returns supported / total_assessable; 0.0 if no assessable sentences.

    Leakage note: only question, context, and response are used — no gold labels.
    """
    sentences = _split_sentences(response)
    if not sentences:
        return 0.0

    supported = 0
    assessable = 0

    for sent in sentences:
        if len(sent.split()) < 3:   # skip very short fragments
            continue
        assessable += 1
        scores = _nli_scores(premise=context, hypothesis=sent)
        if scores.get("entailment", 0.0) > scores.get("contradiction", 0.0) and \
           scores.get("entailment", 0.0) > scores.get("neutral", 0.0):
            supported += 1

    if assessable == 0:
        return 0.0
    return supported / assessable


# ── main API ───────────────────────────────────────────────────────────────────
def extract_features(question: str, context: str, response: str) -> dict[str, float]:
    """
    Compute all four features for one (question, context, response) triple.
    Returns a flat dict of floats suitable for a classifier row.
    """
    sem_sim = semantic_similarity(response, context)
    lex_p, lex_r, lex_f1 = lexical_overlap(response, context)
    nli_ent = nli_entailment_score(response, context)
    claim_ratio = atomic_claim_support_ratio(response, context)

    return {
        "semantic_similarity": sem_sim,
        "lexical_precision": lex_p,
        "lexical_recall": lex_r,
        "lexical_f1": lex_f1,
        "nli_entailment": nli_ent,
        "atomic_claim_support_ratio": claim_ratio,
    }
