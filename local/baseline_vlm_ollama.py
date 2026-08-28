"""Zero-shot VLM baseline via local Ollama (llava:7b-v1.6-mistral-q4_K_M).

This is the number the fine-tuned/fused/augmented pipeline needs to beat --
per the RAG-Fused-DORA-vs-zero-shot-LLaVA precedent on Bengali hateful memes,
we expect this to underperform the OCR/CLIP baselines on the rarer classes.
Uses Ollama's JSON mode for structured, parseable output in one forward pass
per image (joint multi-task prompting, not 5 separate calls).
"""
import base64
import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import LANG_CONFIG, load_split
from common.metrics import TASKS, score_predictions, record_result

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = os.environ.get("HASOC_VLM_MODEL", "llava:7b-v1.6-mistral-q4_K_M")

LABEL_SPACE = {
    "tamil": {
        "sentiment": ["negative", "neutral", "positive"],
        "sarcasm": ["yes", "no"],
        "vulgar": ["vulgar", "not vulgar"],
        "abuse": ["abusive", "non-abusive"],
        "target": ["individual", "others", "social sub-groups", "political", "gender", "none"],
    },
    "telugu": {
        "sentiment": ["negative", "neutral", "positive"],
        "sarcasm": ["yes", "no"],
        "vulgar": ["vulgar", "not vulgar"],
        "abuse": ["abusive", "non-abusive"],
        "target": ["social sub-groups", "gender", "individual", "none", "others",
                    "political", "national origin", "religion"],
    },
}


def build_prompt(lang):
    labels = LABEL_SPACE[lang]
    return (
        "This image is a social-media meme in " + lang.capitalize() + " (code-mixed with English). "
        "Read any embedded text and classify the meme on all 5 fields below. "
        "Respond with ONLY a JSON object, no extra text, using EXACTLY these keys and "
        "ONLY these allowed values:\n"
        f"- sentiment: one of {labels['sentiment']}\n"
        f"- sarcasm: one of {labels['sarcasm']}\n"
        f"- vulgar: one of {labels['vulgar']}\n"
        f"- abuse: one of {labels['abuse']}\n"
        f"- target: one of {labels['target']}\n"
    )


def query_ollama(image_path, prompt, timeout=120):
    # /api/chat, not /api/generate: some models (e.g. qwen3-vl, which emits a
    # separate reasoning trace) were returning an empty "response" under
    # /api/generate + format=json -- confirmed via direct diagnostic that the
    # same request works fine as a chat message and the model answers normally.
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    if not content.strip():
        raise ValueError("empty response content")
    return content


def parse_response(raw, lang):
    labels = LABEL_SPACE[lang]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(match.group(0)) if match else {}

    out = {}
    missing = 0
    for task in TASKS:
        val = str(obj.get(task, "")).strip().lower()
        allowed = labels[task]
        if val in allowed:
            out[task] = val
        elif val:
            # non-empty but not an exact match -- try substring match before
            # giving up, e.g. model said "very negative" instead of "negative"
            hit = next((a for a in allowed if a in val or val in a), None)
            out[task] = hit if hit else allowed[0]
            if not hit:
                missing += 1
        else:
            # empty string is a substring of everything, so it would silently
            # "match" the first allowed label above without this branch --
            # that masked real empty-response failures as clean 0-parse-failure
            # runs in an earlier sweep. Treat missing fields as an explicit miss.
            out[task] = allowed[0]
            missing += 1
    return out, missing


def run_lang(lang):
    _, dev_rows = load_split(lang)
    prompt = build_prompt(lang)

    predictions = {task: [] for task in TASKS}
    request_failures = 0
    field_misses = 0
    t0 = time.time()
    for i, row in enumerate(dev_rows):
        try:
            raw = query_ollama(row["image_path"], prompt)
            parsed, missing = parse_response(raw, lang)
            field_misses += missing
        except Exception as e:
            print(f"  WARN: {row['id']} failed ({e}), falling back to first-label defaults")
            parsed = {task: LABEL_SPACE[lang][task][0] for task in TASKS}
            request_failures += 1
        for task in TASKS:
            predictions[task].append(parsed[task])
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining = rate * (len(dev_rows) - i - 1)
            print(f"  {lang}: {i+1}/{len(dev_rows)} done, ~{remaining:.0f}s remaining")

    scores = score_predictions(dev_rows, predictions)
    baseline_name = "zeroshot_vlm_" + MODEL.replace(":", "_").replace(".", "_")
    record_result(lang, baseline_name, scores,
                   notes=f"{MODEL}, request_failures={request_failures}/{len(dev_rows)}, "
                         f"field_misses={field_misses}/{len(dev_rows)*len(TASKS)}",
                   supervision="zero_shot")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for lang in LANG_CONFIG:
        if only and lang != only:
            continue
        run_lang(lang)
