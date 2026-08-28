"""Same zero-shot LLM setup as baseline_llm_zeroshot.py, but targeting the
specific failure mode found there: near-total bias toward predicting the
majority class (1.8% hate recall despite 37% true hate rate). Two standard,
well-established calibration techniques, neither tried yet in this session:

1. Base-rate priming -- tell the model the actual class prevalence, since
   zero-shot LLMs default to an implicit, ungrounded prior that skews safe.
2. Balanced few-shot (not majority-skewed) -- 2 hate + 2 non-hate examples
   from TRAIN (never test), establishing the decision boundary explicitly
   rather than letting the model's own bias dominate.

Evaluated on the exact same 150-example seed=42 sample as the plain
zero-shot baseline, for a direct, fair before/after comparison.
"""
import json
import os
import random
import re
import sys
import time

import requests
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_llm_zeroshot import MODEL, SAMPLE_SIZE, SEED, load_split, query_ollama

# balanced, from TRAIN only (never test) -- see terminal output for selection
FEWSHOT = [
    {"text": "*রমনী যার স্বামী বিদেশ    *বেস্ট ফ্রেন্ড    *প্রবাসী স্বামী ", "label": "hate"},
    {"text": "BANGLADESHI GIRLS BE LIKE:   WE SHOULD BOYCOTT EVERYONE AND ", "label": "hate"},
    {"text": "CHILDHOOD IS WHEN    YOU THOUGHT  কাজলা দিদি  WAS HIDING    ", "label": "non-hate"},
    {"text": "হাতের লেখাটা একটু ভালো করো নিলে ভবিষ্যৎ জীবনে কষ্ট পাবে না, ", "label": "non-hate"},
]


def build_prompt(text):
    examples = "\n".join(
        f'  Caption: "{ex["text"]}"\n  label: {ex["label"]}' for ex in FEWSHOT
    )
    return (
        "You are classifying captions from Bengali social-media memes (code-mixed with English) "
        "as hateful or not. In this dataset, approximately 37% of memes are labeled hateful -- "
        "hate is common here, not rare, so do not default to \"non-hate\" out of caution.\n\n"
        f"Examples:\n{examples}\n\n"
        f'Now classify this caption: "{text}"\n\n'
        "Respond with ONLY a JSON object, no extra text, using EXACTLY this key and ONLY these "
        "allowed values:\n- label: one of ['hate', 'non-hate']\n"
    )


def parse_response(raw):
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(match.group(0)) if match else {}
    val = str(obj.get("label", "")).strip().lower()
    if val in ("hate", "non-hate"):
        return val
    if "non" in val or "not" in val:
        return "non-hate"
    if "hate" in val:
        return "hate"
    return "non-hate"


def main():
    test = load_split("test")
    rng = random.Random(SEED)
    sample = rng.sample(test, min(SAMPLE_SIZE, len(test)))  # identical sample to plain zero-shot

    preds, y_true = [], []
    failures = 0
    t0 = time.time()
    for i, row in enumerate(sample):
        try:
            raw = query_ollama(build_prompt(row["text"]))
            pred = parse_response(raw)
        except Exception as e:
            print(f"  WARN: {row['id']} failed ({e})")
            pred = "non-hate"
            failures += 1
        preds.append(pred)
        y_true.append(row["label"])
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            remaining = (elapsed / (i + 1)) * (len(sample) - i - 1)
            print(f"  {i+1}/{len(sample)} done, ~{remaining:.0f}s remaining")

    macro_f1 = f1_score(y_true, preds, average="macro")
    p, r, f1, support = precision_recall_fscore_support(y_true, preds, labels=["hate"], zero_division=0)
    print(f"\nMODEL={MODEL} (calibrated: base-rate + balanced few-shot) n={len(sample)} failures={failures}")
    print(f"macro_f1={macro_f1:.4f}")
    print(f"hate class: precision={p[0]:.4f} recall={r[0]:.4f} f1={f1[0]:.4f} support={support[0]}")


if __name__ == "__main__":
    main()
