"""Shared scoring so every baseline (local + colab) reports comparable numbers."""
import csv
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(ROOT, "results", "comparison.csv")

TASKS = ["sentiment", "sarcasm", "vulgar", "abuse", "target"]


def macro_f1(y_true, y_pred):
    labels = set(y_true) | set(y_pred)
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def accuracy(y_true, y_pred):
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0


def score_predictions(dev_rows, predictions):
    """predictions: dict[task] -> list of predicted labels, aligned to dev_rows order."""
    scores = {}
    for task in TASKS:
        y_true = [r[task] for r in dev_rows]
        y_pred = predictions[task]
        scores[task] = {
            "accuracy": round(accuracy(y_true, y_pred), 4),
            "macro_f1": round(macro_f1(y_true, y_pred), 4),
        }
    scores["overall_macro_f1"] = round(
        sum(s["macro_f1"] for s in scores.values() if isinstance(s, dict)) / len(TASKS), 4
    )
    return scores


SUPERVISION_REGIMES = {"none", "zero_shot", "linear_probe_supervised", "fine_tuned"}


def record_result(lang, baseline_name, scores, notes="", supervision="linear_probe_supervised"):
    """supervision must be one of SUPERVISION_REGIMES -- required so results
    from different supervision regimes (zero-shot prompting vs. a classifier
    head trained on the labeled HASOC split vs. a genuinely fine-tuned
    encoder) are never silently compared as if they were the same kind of
    result. Default is linear_probe_supervised since that's what most
    baselines in this repo are; zero-shot callers must pass it explicitly."""
    assert supervision in SUPERVISION_REGIMES, f"unknown supervision regime: {supervision}"
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    is_new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(
                ["lang", "baseline", "supervision", "sentiment_f1", "sarcasm_f1", "vulgar_f1",
                 "abuse_f1", "target_f1", "overall_macro_f1", "notes"]
            )
        w.writerow([
            lang, baseline_name, supervision,
            scores["sentiment"]["macro_f1"], scores["sarcasm"]["macro_f1"],
            scores["vulgar"]["macro_f1"], scores["abuse"]["macro_f1"],
            scores["target"]["macro_f1"], scores["overall_macro_f1"], notes,
        ])
    print(f"[{lang}/{baseline_name}] overall_macro_f1={scores['overall_macro_f1']}  "
          f"(sentiment={scores['sentiment']['macro_f1']}, sarcasm={scores['sarcasm']['macro_f1']}, "
          f"vulgar={scores['vulgar']['macro_f1']}, abuse={scores['abuse']['macro_f1']}, "
          f"target={scores['target']['macro_f1']})")


def majority_predict(train_rows, dev_rows):
    predictions = {}
    for task in TASKS:
        majority_label = Counter(r[task] for r in train_rows).most_common(1)[0][0]
        predictions[task] = [majority_label] * len(dev_rows)
    return predictions
