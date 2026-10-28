"""
llm_judge.py
------------
Independent LLM-based hallucination judge.

The judge receives (question, context, response) — never the ground-truth label.
It returns a structured prediction: hallucination_present (0/1) and optionally a
confidence string.

The judge is evaluated against the same held-out test labels used for the
ML classifiers.  It is a BASELINE, not a scorer of the classifiers.

Usage:
    python llm_judge.py [--test-csv data/ragtruth/test.csv]
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

load_dotenv()

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

JUDGE_MODEL = "gemini-2.0-flash"

# Fixed prompt — documented here; not tuned on any test examples.
JUDGE_PROMPT_TEMPLATE = """You are an expert fact-checker. Your task is to decide whether the RESPONSE
contains any content that is NOT supported by the CONTEXT.

Definitions:
- "Supported" means every factual claim in the RESPONSE can be directly verified from the CONTEXT.
- "Hallucinated" means the RESPONSE contains at least one claim that is absent from, or contradicts, the CONTEXT.

You must reply with a single JSON object and nothing else:
{{
  "hallucination_present": 0 or 1,
  "confidence": "high" | "medium" | "low",
  "brief_reason": "<one sentence>"
}}

QUESTION: {question}

CONTEXT:
{context}

RESPONSE:
{response}

Reply with JSON only:"""


def _get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env")
    return genai.Client(api_key=api_key)


def judge_one(client, question: str, context: str, response: str) -> dict:
    """
    Call the Gemini judge for a single example.
    Returns {'hallucination_present': int, 'confidence': str, 'brief_reason': str}
    on success, or {'hallucination_present': -1, 'error': str} on failure.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question[:800],
        context=context[:2000],
        response=response[:1000],
    )
    try:
        resp = client.models.generate_content(model=JUDGE_MODEL, contents=prompt)
        text = resp.text.strip()

        # Extract JSON from the response (handle markdown code blocks)
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            return {"hallucination_present": -1, "error": f"No JSON found: {text[:200]}"}

        parsed = json.loads(json_match.group())
        parsed["hallucination_present"] = int(parsed.get("hallucination_present", -1))
        return parsed

    except Exception as e:
        return {"hallucination_present": -1, "error": str(e)}


def run_judge(test_csv: str):
    df = pd.read_csv(test_csv)
    print(f"[judge] Evaluating {len(df)} test records ...")

    client = _get_gemini_client()
    predictions = []

    for i, row in df.iterrows():
        if i % 20 == 0:
            print(f"  {i}/{len(df)}")
        result = judge_one(
            client,
            question=str(row["question"]),
            context=str(row["context"]),
            response=str(row["response"]),
        )
        result["ground_truth"] = int(row["label"])
        predictions.append(result)

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(RESULTS_DIR / "llm_judge_predictions.csv", index=False)

    # Filter out errors (hallucination_present == -1)
    valid = pred_df[pred_df["hallucination_present"] >= 0].copy()
    error_count = len(pred_df) - len(valid)
    print(f"[judge] Valid predictions: {len(valid)} / {len(pred_df)}  (errors: {error_count})")

    if len(valid) == 0:
        print("[judge] No valid predictions; check API key and model.")
        return

    y_true = valid["ground_truth"].values
    y_pred = valid["hallucination_present"].values

    print("\n" + "="*60)
    print("  LLM Judge Results")
    print("="*60)
    print(f"  Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1        : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Macro-F1  : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"\n  Confusion matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Supported    : TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  Hallucinated : FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['supported', 'hallucinated'], zero_division=0)}")

    summary = {
        "model": "LLM_Judge_Gemini",
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "valid_predictions": len(valid),
        "errors": error_count,
    }
    with open(RESULTS_DIR / "llm_judge_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[judge] Summary saved to {RESULTS_DIR / 'llm_judge_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="data/ragtruth/test.csv")
    args = parser.parse_args()
    run_judge(args.test_csv)
