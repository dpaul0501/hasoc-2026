"""IMSyPP/hate_speech_multilingual -- trained only on English/Italian/
Slovenian (confirmed via model card, NOT actually Bengali/Indic despite the
"multilingual" name). Included as a genuine negative control: does a
classifier with zero language overlap with Bengali completely fail, as
expected, or does something transfer? Maps its 4 classes (appropriate /
inappropriate / offensive / violent) to binary hate/non-hate: appropriate ->
non-hate, everything else -> hate.
"""
import json
import os

import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "IMSyPP/hate_speech_multilingual"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_split(name):
    with open(os.path.join(ROOT, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    test = load_split("test")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    texts = [r["text"] for r in test]
    y_true = [r["label"] for r in test]
    preds = []

    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt").to(DEVICE)
            logits = model(**enc).logits
            batch_preds = logits.argmax(dim=-1).cpu().numpy()
            preds.extend("non-hate" if p == 0 else "hate" for p in batch_preds)
            if (i // batch_size + 1) % 5 == 0:
                print(f"  {i+len(batch)}/{len(texts)} done")

    macro_f1 = f1_score(y_true, preds, average="macro")
    p, r, f1, support = precision_recall_fscore_support(y_true, preds, labels=["hate"], zero_division=0)
    print(f"\nMODEL={MODEL_NAME} n={len(texts)} (full test set)")
    print(f"macro_f1={macro_f1:.4f}")
    print(f"hate class: precision={p[0]:.4f} recall={r[0]:.4f} f1={f1[0]:.4f} support={support[0]}")


if __name__ == "__main__":
    main()
