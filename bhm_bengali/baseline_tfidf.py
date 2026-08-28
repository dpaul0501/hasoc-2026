"""OCR/caption text + char-ngram TF-IDF + logistic regression, on BHM Bengali
Hateful Memes -- same exact recipe as the HASOC OCR+TF-IDF baseline (the one
that beat every VLM/LLM there), applied here as a replication test: does
"simple beats complex" hold on a larger (7109 vs 800), better-balanced
(37% vs 1.6-14% minority) dataset, or was it an artifact of HASOC's severe
imbalance and tiny sample size?
"""
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_split(name):
    with open(os.path.join(ROOT, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    train = load_split("train")
    test = load_split("test")

    X_train_text = [r["text"] for r in train]
    X_test_text = [r["text"] for r in test]
    y_train = [r["label"] for r in train]
    y_test = [r["label"] for r in test]

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=20000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    macro_f1 = f1_score(y_test, preds, average="macro")
    p, r, f1, support = precision_recall_fscore_support(y_test, preds, labels=["hate"], zero_division=0)
    print(f"macro_f1={macro_f1:.4f}")
    print(f"hate class: precision={p[0]:.4f} recall={r[0]:.4f} f1={f1[0]:.4f} support={support[0]}")


if __name__ == "__main__":
    main()
