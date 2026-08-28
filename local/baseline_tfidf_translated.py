"""Same recipe as baseline_tfidf.py, but on machine-translated English text
instead of native-script OCR text. Direct comparison answers: does
normalizing code-mixed Tamil/Telugu to English before classification help?

Word-level n-grams here (not char n-grams like the native baseline) --
English is the standard case where word tokenization is well-behaved, and
using word-level makes this a fair "standard English NLP pipeline" test
rather than an apples-to-oranges vectorizer choice.
"""
import csv
import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import LANG_CONFIG, load_split, ROOT
from common.metrics import TASKS, score_predictions, record_result


def load_translated_text(lang):
    path = os.path.join(ROOT, "splits", f"{lang}_ocr_text_en.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r["ocr_text_en"] for r in csv.DictReader(f)}


def run_lang(lang):
    train_rows, dev_rows = load_split(lang)
    en_text = load_translated_text(lang)

    X_train_text = [en_text.get(r["id"], "") for r in train_rows]
    X_dev_text = [en_text.get(r["id"], "") for r in dev_rows]

    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                  stop_words="english", max_features=20000)
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
    record_result(lang, "translated_en_tfidf_logreg", scores, notes=f"empty_frac={empty_frac:.2f}")


if __name__ == "__main__":
    for lang in LANG_CONFIG:
        run_lang(lang)
