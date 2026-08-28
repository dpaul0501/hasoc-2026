"""OCR-text baseline: TF-IDF (char n-grams, script-agnostic) + Logistic Regression.

Char n-grams (not word n-grams) because OCR output is noisy and Tamil/Telugu
are agglutinative -- word-level tokenization would fragment badly on OCR
errors and morphology alike.
"""
import csv
import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import LANG_CONFIG, load_split, ROOT
from common.metrics import TASKS, score_predictions, record_result


def load_ocr_text(lang):
    path = os.path.join(ROOT, "splits", f"{lang}_ocr_text.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r["ocr_text"] for r in csv.DictReader(f)}


def run_lang(lang):
    train_rows, dev_rows = load_split(lang)
    ocr_text = load_ocr_text(lang)

    X_train_text = [ocr_text.get(r["id"], "") for r in train_rows]
    X_dev_text = [ocr_text.get(r["id"], "") for r in dev_rows]

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=20000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_dev = vectorizer.transform(X_dev_text)

    predictions = {}
    for task in TASKS:
        y_train = [r[task] for r in train_rows]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_train, y_train)
        predictions[task] = list(clf.predict(X_dev))

    scores = score_predictions(dev_rows, predictions)
    empty_frac = sum(1 for t in X_dev_text if not t.strip()) / len(X_dev_text)
    record_result(lang, "ocr_tfidf_logreg", scores, notes=f"empty_ocr_frac={empty_frac:.2f}")


if __name__ == "__main__":
    for lang in LANG_CONFIG:
        run_lang(lang)
