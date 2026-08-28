"""Zero-shot text-only LLM baseline: OCR'd caption -> deepseek-r1 (or any
Ollama text model), same 5-task JSON prompt as the VLM baseline but reasoning
over text alone, no image. Distinct axis from baseline_vlm_ollama.py (vision)
and baseline_tfidf.py (supervised/trained) -- this is zero-shot + text-only,
answering: does a strong text reasoning model beat simple TF-IDF zero-shot,
without ever seeing the image?
"""
import csv
import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import LANG_CONFIG, ROOT, load_split
from common.metrics import TASKS, score_predictions, record_result
from local.baseline_vlm_ollama import LABEL_SPACE

MODEL = os.environ.get("HASOC_LLM_MODEL", "deepseek-r1:7b")


def load_ocr_text(lang):
    path = os.path.join(ROOT, "splits", f"{lang}_ocr_text.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r["ocr_text"] for r in csv.DictReader(f)}


def build_prompt(lang, text):
    labels = LABEL_SPACE[lang]
    return (
        f"This is the OCR-extracted text from a social-media meme in {lang.capitalize()} "
        "(code-mixed with English, transliterated/translated where noted): \"" + text + "\"\n\n"
        "Classify it on all 5 fields below. Respond with ONLY a JSON object, no extra text, "
        "using EXACTLY these keys and ONLY these allowed values:\n"
        f"- sentiment: one of {labels['sentiment']}\n"
        f"- sarcasm: one of {labels['sarcasm']}\n"
        f"- vulgar: one of {labels['vulgar']}\n"
        f"- abuse: one of {labels['abuse']}\n"
        f"- target: one of {labels['target']}\n"
    )


def query_ollama(prompt, timeout=120, retries=3):
    # root cause found via Ollama's own server log (not a guess): deepseek-r1
    # emits a <think>...</think> reasoning trace before its JSON answer, and
    # /api/chat's format="json" grammar-constrained parser
    # (common_chat_peg_parse) crashes trying to force that into strict JSON,
    # producing a 500 *after* the model already finished a correct answer --
    # reproduced even with format="json" removed, so the chat endpoint's
    # template/parser itself is the problem, not a request parameter.
    # /api/generate (raw completion, no chat template) sidesteps it entirely
    # -- confirmed working for both deepseek-r1 and qwen2.5:7b before making
    # this the default for every model, not just the one that needed it.
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0},
                },
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
            hit = next((a for a in allowed if a in val or val in a), None)
            out[task] = hit if hit else allowed[0]
            if not hit:
                missing += 1
        else:
            out[task] = allowed[0]
            missing += 1
    return out, missing


def run_lang(lang):
    _, dev_rows = load_split(lang)
    ocr_text = load_ocr_text(lang)

    predictions = {task: [] for task in TASKS}
    request_failures = 0
    field_misses = 0
    for row in dev_rows:
        text = ocr_text.get(row["id"], "")
        prompt = build_prompt(lang, text)
        try:
            raw = query_ollama(prompt)
            parsed, missing = parse_response(raw, lang)
            field_misses += missing
        except Exception as e:
            print(f"  WARN: {row['id']} failed ({e})")
            parsed = {task: LABEL_SPACE[lang][task][0] for task in TASKS}
            request_failures += 1
        for task in TASKS:
            predictions[task].append(parsed[task])

    scores = score_predictions(dev_rows, predictions)
    baseline_name = "zeroshot_textllm_" + MODEL.replace(":", "_").replace(".", "_")
    record_result(lang, baseline_name, scores,
                   notes=f"{MODEL}, text-only (OCR), request_failures={request_failures}/{len(dev_rows)}, "
                         f"field_misses={field_misses}/{len(dev_rows)*len(TASKS)}",
                   supervision="zero_shot")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for lang in LANG_CONFIG:
        if only and lang != only:
            continue
        run_lang(lang)
