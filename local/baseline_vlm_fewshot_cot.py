"""Same model as the zero-shot VLM baseline (qwen2.5vl by default), but with
few-shot in-context examples (real training images + true labels, including
at least one genuine vulgar/abusive positive per language, since those are
what the bare zero-shot prompt was missing entirely -- 0% recall) and an
explicit chain-of-thought instruction (reason first, then answer).

This directly tests the "models are losing because of a weak zero-shot
prompt, not because they can't do the task" hypothesis, holding the model
fixed so it's a clean comparison against baseline_vlm_ollama.py's numbers,
not confounded by picking a different model.
"""
import base64
import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import LANG_CONFIG, ROOT, load_split, load_train_rows
from common.metrics import TASKS, score_predictions, record_result
from local.baseline_vlm_ollama import LABEL_SPACE

MODEL = os.environ.get("HASOC_VLM_MODEL", "qwen2.5vl:latest")

# real training examples, hand-picked to include a genuine vulgar+abusive
# positive per language (the classes the zero-shot baseline caught 0% of)
FEWSHOT_IDS = {
    "tamil": ["image_tamil_0379.jpg", "image_tamil_0147.jpg", "image_tamil_0543.jpg"],
    "telugu": ["image_telugu_0850.png", "image_telugu_0524.png", "image_telugu_0375.png"],
}


def build_system_instructions(lang):
    labels = LABEL_SPACE[lang]
    return (
        f"You classify social-media memes in {lang.capitalize()} (code-mixed with English) "
        "on 5 fields. First reason step by step about the image and any embedded text "
        "(sentiment, whether it's sarcastic, whether the language/imagery is vulgar, "
        "whether it's abusive toward someone, and who/what it targets). "
        "Then give your final answer as a JSON object on its own line, with EXACTLY "
        "these keys and ONLY these allowed values:\n"
        f"- sentiment: one of {labels['sentiment']}\n"
        f"- sarcasm: one of {labels['sarcasm']}\n"
        f"- vulgar: one of {labels['vulgar']}\n"
        f"- abuse: one of {labels['abuse']}\n"
        f"- target: one of {labels['target']}\n"
    )


def build_fewshot_messages(lang):
    train_rows = {r["id"]: r for r in load_train_rows(lang)}
    messages = [{"role": "system", "content": build_system_instructions(lang)}]
    for img_id in FEWSHOT_IDS[lang]:
        row = train_rows[img_id]
        img_path = row["image_path"]
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        answer = {task: row[task] for task in TASKS}
        rationale = (
            f"Reasoning: this meme's sentiment reads as {row['sentiment']}; "
            f"sarcasm={row['sarcasm']}; the language/imagery is {row['vulgar']}; "
            f"it is {row['abuse']} toward its target ({row['target']})."
        )
        messages.append({"role": "user", "content": "Classify this meme.", "images": [b64]})
        messages.append({"role": "assistant", "content": f"{rationale}\n{json.dumps(answer)}"})
    return messages


def query_ollama(image_path, fewshot_messages, timeout=180):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    messages = fewshot_messages + [{"role": "user", "content": "Classify this meme.", "images": [b64]}]
    resp = requests.post(
        "http://localhost:11434/api/chat",
        # 3 few-shot images + rationales needs ~4600 tokens, over the model's
        # default 4096 num_ctx (confirmed via the server's actual error
        # message) -- 8192 gives headroom without guessing
        json={"model": MODEL, "messages": messages, "stream": False,
              "options": {"temperature": 0, "num_ctx": 8192}},
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    if not content.strip():
        raise ValueError("empty response content")
    return content


def parse_response(raw, lang):
    labels = LABEL_SPACE[lang]
    # take the LAST {...} block -- the CoT rationale comes before the JSON,
    # and (as seen with deepseek-r1 earlier) reasoning prose can occasionally
    # contain stray braces, so search from the end rather than the start
    matches = list(re.finditer(r"\{[^{}]*\}", raw, re.DOTALL))
    obj = {}
    if matches:
        try:
            obj = json.loads(matches[-1].group(0))
        except json.JSONDecodeError:
            obj = {}

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


def pilot_subset(dev_rows, n, seed=7):
    # prioritize including the rare vulgar/abusive positives -- Tamil dev has
    # only 1 vulgar + 2 abusive examples total, so a plain random subset of
    # size 35 could easily miss them, making the pilot uninformative for the
    # exact question ("does better prompting help catch the minority class")
    import random
    priority = [r for r in dev_rows if r["vulgar"] == "vulgar" or r["abuse"] == "abusive"]
    rest = [r for r in dev_rows if r not in priority]
    rng = random.Random(seed)
    rng.shuffle(rest)
    return priority + rest[:max(0, n - len(priority))]


def run_lang(lang, pilot_size=None):
    _, dev_rows = load_split(lang)
    if pilot_size:
        dev_rows = pilot_subset(dev_rows, pilot_size)
    fewshot_messages = build_fewshot_messages(lang)

    predictions = {task: [] for task in TASKS}
    request_failures = 0
    field_misses = 0
    t0 = time.time()
    for i, row in enumerate(dev_rows):
        try:
            raw = query_ollama(row["image_path"], fewshot_messages)
            parsed, missing = parse_response(raw, lang)
            field_misses += missing
        except Exception as e:
            print(f"  WARN: {row['id']} failed ({e})")
            parsed = {task: LABEL_SPACE[lang][task][0] for task in TASKS}
            request_failures += 1
        for task in TASKS:
            predictions[task].append(parsed[task])
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            remaining = (elapsed / (i + 1)) * (len(dev_rows) - i - 1)
            print(f"  {lang}: {i+1}/{len(dev_rows)} done, ~{remaining:.0f}s remaining")

    scores = score_predictions(dev_rows, predictions)
    baseline_name = "fewshot_cot_vlm_" + MODEL.replace(":", "_").replace(".", "_")
    pilot_note = f"PILOT n={len(dev_rows)} (not full 160-dev, not directly comparable), " if pilot_size else ""
    record_result(lang, baseline_name, scores,
                   notes=f"{pilot_note}{MODEL}, 3-shot incl. vulgar+abusive positive, CoT, "
                         f"request_failures={request_failures}/{len(dev_rows)}, "
                         f"field_misses={field_misses}/{len(dev_rows)*len(TASKS)}",
                   supervision="zero_shot")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    pilot_size = int(os.environ.get("HASOC_PILOT_SIZE", "35"))
    for lang in LANG_CONFIG:
        if only and lang != only:
            continue
        run_lang(lang, pilot_size=pilot_size)
