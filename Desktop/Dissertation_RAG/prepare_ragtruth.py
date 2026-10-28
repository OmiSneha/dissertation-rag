"""
prepare_ragtruth.py
-------------------
Download the RAGTruth QA subset, inspect annotation fields,
map human annotations to binary response-level labels, and
write train / validation / test CSV files.

Label mapping (documented rule):
  1 — hallucination present  : response has >= 1 annotation whose type
                                is NOT 'implicit_true' (i.e. a real error span)
  0 — no hallucination found : all annotations are 'implicit_true', or
                               there are no annotations at all

Quality filter:
  Exclude records whose 'quality' field == 'low'
  (covers truncated responses and incorrect refusals as per dataset docs).

Split strategy:
  Use the official 'split' field where present.
  If 'split' is absent, use 80/10/10 stratified split.
  Validation set is carved from train only; test set is never touched
  during model development.
"""

import json
import os
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

OUT_DIR = Path("data/ragtruth")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── hallucination annotation types that count as real errors ───────────────────
# Source: RAGTruth paper Table 1 – annotation type definitions.
# 'implicit_true' means the span is technically absent from the context but is
# a universally accepted fact; it is NOT counted as a hallucination here.
HALLUCINATION_TYPES = {
    "evident_conflict",
    "subtle_conflict",
    "left-out_key_info",
    "unverifiable_info",
}


def load_ragtruth():
    print("[prepare] Loading RAGTruth from HuggingFace Hub ...")
    ds = load_dataset("wandb/RAGTruth", trust_remote_code=True)
    return ds


def to_dataframe(ds) -> pd.DataFrame:
    """Flatten all splits into a single DataFrame."""
    frames = []
    for split_name in ds:
        df = ds[split_name].to_pandas()
        df["hf_split"] = split_name
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    print(f"[prepare] Raw dataset: {len(full)} records across splits: {list(ds.keys())}")
    print(f"[prepare] Columns: {list(full.columns)}")
    return full


def filter_qa(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only QA task records; exclude summarization and data-to-text."""
    before = len(df)
    # The 'task_type' (or 'source_type') column identifies the task.
    task_col = _find_col(df, ["task_type", "source_type", "task"])
    if task_col:
        qa_mask = df[task_col].str.lower().str.contains("qa|question", na=False)
        df = df[qa_mask].copy()
    print(f"[prepare] QA filter: {before} → {len(df)} records")
    return df


def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude low-quality records (truncated, incorrect refusals)."""
    before = len(df)
    quality_col = _find_col(df, ["quality", "data_quality"])
    if quality_col:
        df = df[df[quality_col].str.lower() != "low"].copy()
    print(f"[prepare] Quality filter: {before} → {len(df)} records")
    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a binary response-level label from the annotation field.
    label = 1 if any annotation has a type in HALLUCINATION_TYPES
    label = 0 otherwise
    """
    ann_col = _find_col(df, ["annotations", "annotation", "labels", "label_list"])
    if ann_col is None:
        raise KeyError("Cannot find annotations column. Columns: " + str(list(df.columns)))

    def _has_hallucination(raw) -> int:
        if raw is None:
            return 0
        # raw may be a list of dicts, a JSON string, or already parsed
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return 0
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        for ann in raw:
            if isinstance(ann, dict):
                ann_type = (
                    ann.get("type")
                    or ann.get("label")
                    or ann.get("annotation_type")
                    or ""
                )
                if str(ann_type).lower() in {t.lower() for t in HALLUCINATION_TYPES}:
                    return 1
        return 0

    df = df.copy()
    df["label"] = df[ann_col].apply(_has_hallucination)

    print(f"[prepare] Label distribution:\n{df['label'].value_counts().to_string()}")
    print(f"[prepare] Hallucination rate: {df['label'].mean():.2%}")
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns needed for feature extraction and evaluation.
    Map dataset-specific column names to canonical names:
      question, context, response, label, split
    """
    col_map = {}
    q_col = _find_col(df, ["question", "query", "input"])
    c_col = _find_col(df, ["context", "evidence", "source_text", "passage"])
    r_col = _find_col(df, ["response", "answer", "output", "generated_text"])
    s_col = _find_col(df, ["hf_split", "split", "data_split"])

    for src, dst in [(q_col, "question"), (c_col, "context"), (r_col, "response"), (s_col, "split")]:
        if src:
            col_map[src] = dst

    df = df.rename(columns=col_map)
    keep = [c for c in ["question", "context", "response", "label", "split"] if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["question", "context", "response", "label"])
    print(f"[prepare] After column selection and dropna: {len(df)} records, cols: {list(df.columns)}")
    return df


def split_data(df: pd.DataFrame):
    """
    Use official split field if present; otherwise do 80/10/10 stratified split.
    Validation is always carved from train only.
    """
    if "split" in df.columns and df["split"].nunique() > 1:
        train = df[df["split"].isin(["train", "training"])].copy()
        test = df[df["split"].isin(["test", "testing"])].copy()
        val = df[df["split"].isin(["val", "validation", "dev"])].copy()

        if len(val) == 0 and len(train) > 0:
            train, val = train_test_split(
                train, test_size=0.1, random_state=42, stratify=train["label"]
            )
        print(f"[prepare] Split (official): train={len(train)}, val={len(val)}, test={len(test)}")
    else:
        train, temp = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )
        val, test = train_test_split(
            temp, test_size=0.5, random_state=42, stratify=temp["label"]
        )
        print(f"[prepare] Split (auto 80/10/10): train={len(train)}, val={len(val)}, test={len(test)}")

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def save(train, val, test):
    for name, df in [("train", train), ("val", val), ("test", test)]:
        out = OUT_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        n_hal = int(df["label"].sum())
        n_sup = len(df) - n_hal
        print(f"[prepare] Saved {name}.csv → {len(df)} records  (hallucination={n_hal}, supported={n_sup})")


def manual_check_sample(df: pd.DataFrame, n: int = 5):
    """Print a small sample for manual label sanity check."""
    print("\n[prepare] ── Manual sample check ──")
    sample = df.sample(min(n, len(df)), random_state=0)
    for _, row in sample.iterrows():
        print(f"\n  Q: {str(row.get('question', ''))[:120]}")
        print(f"  A: {str(row.get('response', ''))[:120]}")
        print(f"  Label: {int(row['label'])} ({'HALLUCINATION' if row['label'] else 'SUPPORTED'})")


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df."""
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None


def main():
    ds = load_ragtruth()
    df = to_dataframe(ds)
    df = filter_qa(df)
    df = filter_quality(df)
    df = map_labels(df)
    df = select_columns(df)
    manual_check_sample(df)
    train, val, test = split_data(df)
    save(train, val, test)
    print("\n[prepare] Done. Files written to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
