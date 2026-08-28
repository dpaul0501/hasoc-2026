"""Calibration + split conformal prediction for the winning baseline
(OCR text + char-ngram TF-IDF + logistic regression).

Point-accuracy (F1) says nothing about whether the model's confidence is
trustworthy -- this is the piece proposed early on as the FIRE paper's
"mathematically grounded contribution" and not yet built. Two things:

1. ECE (Expected Calibration Error) per task: bins predictions by confidence,
   compares within-bin accuracy to within-bin average confidence. Large gap
   = overconfident or underconfident.

2. Split conformal prediction: holds out half the dev set purely for
   calibration (never seen during training OR the other half's evaluation),
   computes a per-task threshold from that half, then measures empirical
   coverage and average prediction-set size on the other half. This gives a
   *guaranteed* (not just estimated) coverage rate -- directly usable as
   "flag this prediction for human review if the conformal set has >1 label
   in it" per the original plan's calibration angle.
"""
import csv
import os
import random
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import LANG_CONFIG, ROOT, load_split
from common.metrics import TASKS

ALPHA = 0.10  # target 90% coverage
SPLIT_SEED = 123


def load_ocr_text(lang):
    path = os.path.join(ROOT, "splits", f"{lang}_ocr_text.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r["ocr_text"] for r in csv.DictReader(f)}


def ece(confidences, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(confidences)
    err = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        err += (mask.sum() / total) * abs(bin_acc - bin_conf)
    return err


def conformal_threshold(calib_probs, calib_true_idx, alpha):
    # nonconformity score = 1 - predicted prob of the TRUE class
    n = len(calib_true_idx)
    scores = 1 - calib_probs[np.arange(n), calib_true_idx]
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def run_lang(lang):
    train_rows, dev_rows = load_split(lang)
    ocr_text = load_ocr_text(lang)

    rng = random.Random(SPLIT_SEED)
    dev_shuffled = dev_rows[:]
    rng.shuffle(dev_shuffled)
    half = len(dev_shuffled) // 2
    calib_rows, eval_rows = dev_shuffled[:half], dev_shuffled[half:]

    X_train_text = [ocr_text.get(r["id"], "") for r in train_rows]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=20000)
    X_train = vectorizer.fit_transform(X_train_text)

    X_calib = vectorizer.transform([ocr_text.get(r["id"], "") for r in calib_rows])
    X_eval = vectorizer.transform([ocr_text.get(r["id"], "") for r in eval_rows])

    results = []
    for task in TASKS:
        y_train = [r[task] for r in train_rows]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_train, y_train)
        classes = list(clf.classes_)

        # -- ECE on eval half, using max predicted prob as confidence --
        eval_probs = clf.predict_proba(X_eval)
        preds = eval_probs.argmax(axis=1)
        confidences = eval_probs.max(axis=1)
        y_eval_idx = np.array([classes.index(r[task]) for r in eval_rows])
        correct = (preds == y_eval_idx).astype(float)
        task_ece = ece(confidences, correct)

        # -- split conformal: threshold from calib half, coverage on eval half --
        calib_probs = clf.predict_proba(X_calib)
        y_calib_idx = np.array([classes.index(r[task]) for r in calib_rows])
        q_hat = conformal_threshold(calib_probs, y_calib_idx, ALPHA)

        pred_sets = eval_probs >= (1 - q_hat)
        covered = pred_sets[np.arange(len(eval_rows)), y_eval_idx]
        coverage = covered.mean()
        avg_set_size = pred_sets.sum(axis=1).mean()

        results.append({
            "lang": lang, "task": task, "n_classes": len(classes),
            "ece": round(task_ece, 4),
            "target_coverage": 1 - ALPHA,
            "empirical_coverage": round(float(coverage), 4),
            "avg_conformal_set_size": round(float(avg_set_size), 2),
        })
    return results


if __name__ == "__main__":
    all_results = []
    for lang in LANG_CONFIG:
        all_results.extend(run_lang(lang))

    out_path = os.path.join(ROOT, "results", "calibration_conformal.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)

    print(f"{'lang':8}{'task':12}{'n_cls':7}{'ECE':>8}{'target_cov':>12}{'emp_cov':>10}{'avg_set_size':>14}")
    for r in all_results:
        print(f"{r['lang']:8}{r['task']:12}{r['n_classes']:<7}{r['ece']:>8}"
              f"{r['target_coverage']:>12}{r['empirical_coverage']:>10}{r['avg_conformal_set_size']:>14}")
    print(f"\nsaved -> {out_path}")
