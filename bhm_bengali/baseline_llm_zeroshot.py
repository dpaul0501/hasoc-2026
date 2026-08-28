"""Zero-shot text LLM baseline on BHM Bengali captions -- same qwen2.5:7b
recipe proven working on HASOC (via /api/generate, confirmed reliable
endpoint from earlier debugging). Direct replication test against the
TF-IDF+LogReg baseline on the same test set.
"""
import json
import os
import random
import re
import time

import requests
from sklearn.metrics import f1_score, precision_recall_fscore_support

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("HASOC_LLM_MODEL", "qwen2.5:7b")
SAMPLE_SIZE = int(os.environ.get("BHM_SAMPLE_SIZE", "150"))
SEED = 42


def load_split(name):
    with open(os.path.join(ROOT, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def build_prompt(text):
    return (
        "This is the caption text from a Bengali social-media meme (code-mixed with English): "
        f"\"{text}\"\n\n"
        "Classify whether this meme is hateful. Respond with ONLY a JSON object, no extra text, "
        "using EXACTLY this key and ONLY these allowed values:\n"
        "- label: one of ['hate', 'non-hate']\n"
    )


def query_ollama(prompt, timeout=60, retries=3):
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
                timeout=timeout,
            )
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    else:
        raise last_exc
    content = resp.json()["response"]
    if not content.strip():
        raise ValueError("empty response content")
    return content


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
    sample = rng.sample(test, min(SAMPLE_SIZE, len(test)))

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
    print(f"\nMODEL={MODEL} n={len(sample)} failures={failures}")
    print(f"macro_f1={macro_f1:.4f}")
    print(f"hate class: precision={p[0]:.4f} recall={r[0]:.4f} f1={f1[0]:.4f} support={support[0]}")


if __name__ == "__main__":
    main()
