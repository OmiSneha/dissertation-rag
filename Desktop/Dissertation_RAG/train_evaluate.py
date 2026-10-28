"""
train_evaluate.py
-----------------
1. Load preprocessed RAGTruth train/val/test CSVs.
2. Compute features for every record (or load cached features).
3. Train Logistic Regression, Random Forest, XGBoost.
4. Evaluate on held-out test set: precision, recall, F1, confusion matrix,
   ROC-AUC, PR-AUC, macro-F1.
5. Feature ablation: remove one feature family at a time.
6. Save all results to results/.

Usage:
    python train_evaluate.py [--skip-features]
    --skip-features: load features from cache (features_train.csv etc.) if they exist.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from feature_extraction import extract_features

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/ragtruth")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "semantic_similarity",
    "lexical_precision",
    "lexical_recall",
    "lexical_f1",
    "nli_entailment",
    "atomic_claim_support_ratio",
]

ABLATION_GROUPS = {
    "no_semantic": ["semantic_similarity"],
    "no_lexical": ["lexical_precision", "lexical_recall", "lexical_f1"],
    "no_nli": ["nli_entailment"],
    "no_atomic": ["atomic_claim_support_ratio"],
}


# ── feature computation ────────────────────────────────────────────────────────
def compute_features_for_split(df: pd.DataFrame, split_name: str, force: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / f"features_{split_name}.csv"
    if not force and cache_path.exists():
        print(f"[features] Loading cached features: {cache_path}")
        return pd.read_csv(cache_path)

    print(f"[features] Computing features for {split_name} ({len(df)} records) ...")
    rows = []
    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  {i}/{len(df)}")
        feats = extract_features(
            question=str(row["question"]),
            context=str(row["context"]),
            response=str(row["response"]),
        )
        feats["label"] = int(row["label"])
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(cache_path, index=False)
    print(f"[features] Saved to {cache_path}")
    return feat_df


# ── classifiers ───────────────────────────────────────────────────────────────
def build_classifiers():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=1,  # adjusted after class distribution check
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
    }


# ── evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

    result = {
        "model": model_name,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=["supported", "hallucinated"], zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "pr_auc": average_precision_score(y_test, y_prob) if y_prob is not None else None,
    }
    return result


def print_result(r: dict):
    print(f"\n{'='*60}")
    print(f"  {r['model']}")
    print(f"{'='*60}")
    print(f"  Precision : {r['precision']:.4f}")
    print(f"  Recall    : {r['recall']:.4f}")
    print(f"  F1        : {r['f1']:.4f}")
    print(f"  Macro-F1  : {r['macro_f1']:.4f}")
    if r["roc_auc"]:
        print(f"  ROC-AUC   : {r['roc_auc']:.4f}")
    if r["pr_auc"]:
        print(f"  PR-AUC    : {r['pr_auc']:.4f}")
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"  [supported | hallucinated]")
    cm = r["confusion_matrix"]
    print(f"  Supported     : TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  Hallucinated  : FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n{r['classification_report']}")


# ── ablation ───────────────────────────────────────────────────────────────────
def run_ablation(classifiers: dict, scaler: StandardScaler, X_train, y_train, X_test, y_test, feature_cols: list[str]):
    print("\n\n" + "="*60)
    print("FEATURE ABLATION")
    print("="*60)
    ablation_results = {}

    for ablation_name, dropped_cols in ABLATION_GROUPS.items():
        remaining = [c for c in feature_cols if c not in dropped_cols]
        drop_idx = [feature_cols.index(c) for c in dropped_cols if c in feature_cols]
        keep_idx = [i for i in range(len(feature_cols)) if i not in drop_idx]

        X_tr_abl = X_train[:, keep_idx]
        X_te_abl = X_test[:, keep_idx]

        group_results = {}
        for clf_name, clf_template in classifiers.items():
            # clone the classifier
            clf = type(clf_template)(**clf_template.get_params())
            clf.fit(X_tr_abl, y_train)
            r = evaluate(clf, X_te_abl, y_test, f"{clf_name}+{ablation_name}")
            group_results[clf_name] = {"f1": r["f1"], "macro_f1": r["macro_f1"], "roc_auc": r["roc_auc"]}
            print(f"  {clf_name} | {ablation_name}: F1={r['f1']:.4f}  Macro-F1={r['macro_f1']:.4f}")
        ablation_results[ablation_name] = group_results

    return ablation_results


# ── main ────────────────────────────────────────────────────────────────────────
def main(skip_features: bool = False):
    # 1. Load preprocessed data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"[train] Dataset sizes — train:{len(train_df)}, val:{len(val_df)}, test:{len(test_df)}")

    # 2. Feature extraction
    force = not skip_features
    feat_train = compute_features_for_split(train_df, "train", force=force)
    feat_val = compute_features_for_split(val_df, "val", force=force)
    feat_test = compute_features_for_split(test_df, "test", force=force)

    # Use train+val for fitting; test is held out
    feat_trainval = pd.concat([feat_train, feat_val], ignore_index=True)

    X_trainval = feat_trainval[FEATURE_COLS].values
    y_trainval = feat_trainval["label"].values
    X_test = feat_test[FEATURE_COLS].values
    y_test = feat_test["label"].values

    # 3. Scale features (for LR; RF and XGB don't need it but it doesn't hurt)
    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, RESULTS_DIR / "scaler.pkl")

    # Adjust XGBoost scale_pos_weight for class imbalance
    n_neg = int((y_trainval == 0).sum())
    n_pos = int((y_trainval == 1).sum())
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"[train] Class distribution — supported:{n_neg}, hallucinated:{n_pos}, scale_pos_weight:{spw:.2f}")

    classifiers = build_classifiers()
    classifiers["XGBoost"].set_params(scale_pos_weight=spw)

    # 4. Train and evaluate
    all_results = []
    trained_models = {}

    for clf_name, clf in classifiers.items():
        print(f"\n[train] Training {clf_name} ...")
        clf.fit(X_trainval_scaled, y_trainval)
        trained_models[clf_name] = clf
        joblib.dump(clf, RESULTS_DIR / f"{clf_name.lower()}.pkl")

        result = evaluate(clf, X_test_scaled, y_test, clf_name)
        print_result(result)
        all_results.append(result)

    # 5. Ablation
    ablation = run_ablation(
        classifiers, scaler,
        X_trainval_scaled, y_trainval,
        X_test_scaled, y_test,
        FEATURE_COLS,
    )

    # 6. Save results
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "model": r["model"],
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1"], 4),
            "macro_f1": round(r["macro_f1"], 4),
            "roc_auc": round(r["roc_auc"], 4) if r["roc_auc"] else None,
            "pr_auc": round(r["pr_auc"], 4) if r["pr_auc"] else None,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "classifier_results.csv", index=False)

    with open(RESULTS_DIR / "ablation.json", "w") as f:
        json.dump(ablation, f, indent=2)

    print(f"\n[train] Results saved to {RESULTS_DIR.resolve()}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-features", action="store_true",
                        help="Load cached features instead of recomputing")
    args = parser.parse_args()
    main(skip_features=args.skip_features)
