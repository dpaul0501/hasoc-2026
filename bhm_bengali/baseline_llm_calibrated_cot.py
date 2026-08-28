"""Same calibration as baseline_llm_calibrated.py (base-rate priming +
balanced few-shot), plus the piece that WASN'T tested there: explicit
chain-of-thought -- each demonstration now carries a one-sentence rationale,
and the model is asked to reason before giving its final label. Same 4
exemplars as baseline_llm_calibrated.py (only the CoT rationale is new), so
this isolates the CoT variable cleanly against that run's numbers.
"""
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_llm_zeroshot import MODEL, SAMPLE_SIZE, SEED, load_split, query_ollama
from sklearn.metrics import f1_score, precision_recall_fscore_support

FEWSHOT = [
    {"text": "*রমনী যার স্বামী বিদেশ    *বেস্ট ফ্রেন্ড    *প্রবাসী স্বামী ", "label": "hate",
     "reasoning": "This implies women with husbands working abroad are secretly unfaithful -- a misogynistic stereotype insinuating infidelity."},
    {"text": "BANGLADESHI GIRLS BE LIKE:   WE SHOULD BOYCOTT EVERYONE AND ", "label": "hate",
     "reasoning": "This uses the mocking 'X be like' format to generalize and ridicule an entire gender+nationality group."},
    {"text": "CHILDHOOD IS WHEN    YOU THOUGHT  কাজলা দিদি  WAS HIDING    ", "label": "non-hate",
     "reasoning": "A generic nostalgic childhood-memory joke referencing a well-known Bengali children's story character, no group or individual is targeted or demeaned."},
    {"text": "হাতের লেখাটা একটু ভালো করো নিলে ভবিষ্যৎ জীবনে কষ্ট পাবে না, ", "label": "non-hate",
     "reasoning": "Generic parental/life advice about improving handwriting, no hateful or offensive content."},
]


def build_prompt(text):
    examples = "\n\n".join(
        f'Caption: "{ex["text"]}"\nReasoning: {ex["reasoning"]}\nlabel: {ex["label"]}' for ex in FEWSHOT
    )
    return (
        "You are classifying captions from Bengali social-media memes (code-mixed with English) "
        "as hateful or not. In this dataset, approximately 37% of memes are labeled hateful -- "
        "hate is common here, not rare, so do not default to \"non-hate\" out of caution. "
        "For each caption, first reason about what it implies and who (if anyone) it targets or "
        "demeans, then give a final label.\n\n"
        f"Examples:\n{examples}\n\n"
        f'Now classify this caption: "{text}"\n\n'
        "Respond with a JSON object with keys \"reasoning\" (1 sentence) and \"label\" "
        "(one of ['hate', 'non-hate'])."
    )


def parse_response(raw):
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        matches = list(re.finditer(r"\{[^{}]*\}", raw, re.DOTALL))
        obj = json.loads(matches[-1].group(0)) if matches else {}
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
    sample = rng.sample(test, min(SAMPLE_SIZE, len(test)))  # identical sample to other BHM LLM runs

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
    print(f"\nMODEL={MODEL} (calibrated + CoT) n={len(sample)} failures={failures}")
    print(f"macro_f1={macro_f1:.4f}")
    print(f"hate class: precision={p[0]:.4f} recall={r[0]:.4f} f1={f1[0]:.4f} support={support[0]}")


if __name__ == "__main__":
    main()
