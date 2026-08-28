"""Retrieval-augmented few-shot + CoT VLM baseline.

Builds on baseline_vlm_fewshot_cot.py's finding (fixed 3-shot incl. genuine
vulgar+abusive positives fixed qwen2.5vl's 0%-recall problem) with one
change: the "typical" negative example is no longer fixed -- it's the
CLIP-nearest training neighbor to the query image, selected dynamically per
example. The vulgar/abusive exemplars stay FIXED (see retrieval_index.py's
finding: pure similarity search can't reliably find same-class matches for a
13/640-example minority class, so retrieval is only used where it's actually
informative -- picking a *relevant* typical/majority example, not hunting
for a needle-in-haystack rare one).
"""
import base64
import json
import os
import re
import sys
import time

import requests
import torch
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import LANG_CONFIG, ROOT, load_split, load_train_rows
from common.metrics import TASKS, score_predictions, record_result
from local.baseline_vlm_ollama import LABEL_SPACE
from local.baseline_vlm_fewshot_cot import build_system_instructions, pilot_subset
from local.retrieval_index import DEVICE, build_or_load_index, embed_images, retrieve_neighbors

MODEL = os.environ.get("HASOC_VLM_MODEL", "qwen2.5vl:latest")

# fixed rare-class exemplars -- v1's set (image_tamil_0147/0543,
# image_telugu_0524/0375) all shared target=individual, giving the model
# zero in-context grounding for the other 5-7 target categories -- exactly
# why `target` was the single worst-performing task in both languages
# (Tamil gap -0.077, Telugu gap -0.217 vs baseline). Replaced with examples
# spanning distinct target categories, selected from train_core only (never
# prompt_val or dev, to keep those clean for validation/final reporting).
FIXED_EXEMPLAR_IDS = {
    # trimmed to 3 (was 4): 5-image prompts measured at ~101s/call, ~2.2x
    # the 3-image version's ~46s -- keeping 3 distinct non-"individual"
    # target categories (the actual bug fix) while limiting the latency hit
    "tamil": [
        "image_tamil_0935.jpg",  # vulgar, target=gender
        "image_tamil_0571.jpg",  # abusive, target=others
        "image_tamil_0981.jpg",  # typical, target=social sub-groups
    ],
    "telugu": [
        "image_telugu_0298.png",  # vulgar, target=political
        "image_telugu_0377.png",  # abusive, target=social sub-groups
        "image_telugu_0091.png",  # typical, target=social sub-groups
    ],
}


def build_demo_turn(row):
    with open(row["image_path"], "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    answer = {task: row[task] for task in TASKS}
    rationale = (
        f"Reasoning: this meme's sentiment reads as {row['sentiment']}; "
        f"sarcasm={row['sarcasm']}; the language/imagery is {row['vulgar']}; "
        f"it is {row['abuse']} toward its target ({row['target']})."
    )
    return [
        {"role": "user", "content": "Classify this meme.", "images": [b64]},
        {"role": "assistant", "content": f"{rationale}\n{json.dumps(answer)}"},
    ]


def query_ollama(image_path, messages, timeout=180):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    full_messages = messages + [{"role": "user", "content": "Classify this meme.", "images": [b64]}]
    resp = requests.post(
        "http://localhost:11434/api/chat",
        # 4 fixed exemplars (up from 2) + 1 dynamic neighbor = 5 demo images
        # now, needs more headroom than the 3-image version's 8192
        json={"model": MODEL, "messages": full_messages, "stream": False,
              "options": {"temperature": 0, "num_ctx": 16384}},
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    if not content.strip():
        raise ValueError("empty response content")
    return content


def parse_response(raw, lang):
    labels = LABEL_SPACE[lang]
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


def run_lang(lang, clip_model, clip_proc, pilot_size=35, eval_mode="val", run_id=None):
    """eval_mode="val": iterate/tune here (train_core exemplars+index, evaluate
    on prompt_val, never touches dev). eval_mode="dev": one-shot final check
    on the untouched 160-example dev set once a design is settled.

    run_id namespaces the checkpoint file and result label WITHOUT affecting
    which data gets used (that's eval_mode alone) -- needed so a control run
    with different exemplars on the same eval_mode can't silently reuse (and
    get contaminated by) another run's cached checkpoint. Defaults to
    eval_mode when not given."""
    run_id = run_id or eval_mode
    if eval_mode == "val":
        from common.val_split import make_val_split
        train_core, prompt_val, _ = make_val_split(lang)
        train_rows_by_id = {r["id"]: r for r in train_core}
        eval_rows = prompt_val
        index = build_or_load_index(lang, clip_model, clip_proc, rows=train_core, cache_suffix="_core")
    else:
        train_rows_by_id = {r["id"]: r for r in load_train_rows(lang)}
        _, eval_rows = load_split(lang)
        index = build_or_load_index(lang, clip_model, clip_proc)

    if pilot_size:
        eval_rows = pilot_subset(eval_rows, pilot_size)

    system_msg = {"role": "system", "content": build_system_instructions(lang)}
    fixed_exemplar_rows = [train_rows_by_id[i] for i in FIXED_EXEMPLAR_IDS[lang]]
    fixed_turns = []
    for row in fixed_exemplar_rows:
        fixed_turns.extend(build_demo_turn(row))
    exclude_ids = set(FIXED_EXEMPLAR_IDS[lang])

    # checkpoint to disk per-example -- this run has been killed by sleep/
    # session-boundary interruptions 3 times in a row, always losing all
    # progress because predictions only existed in memory until the very
    # end. Every completed example's prediction+failure-flags are now
    # written immediately, and already-checkpointed examples are skipped on
    # restart, so a 4th interruption loses at most one in-flight example.
    ckpt_path = os.path.join(ROOT, "results", f"{lang}_retrieval_cot_{run_id}_checkpoint.json")
    checkpoint = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            checkpoint = json.load(f)
        print(f"  resuming from checkpoint: {len(checkpoint)}/{len(eval_rows)} already done")

    predictions = {task: [] for task in TASKS}
    request_failures = 0
    field_misses = 0
    t0 = time.time()
    for i, row in enumerate(eval_rows):
        if row["id"] in checkpoint:
            entry = checkpoint[row["id"]]
            parsed = entry["parsed"]
            field_misses += entry["missing"]
            request_failures += entry["failed"]
        else:
            try:
                q_emb = embed_images([row], clip_model, clip_proc)[0]
                neighbor_row, _sim = retrieve_neighbors(q_emb, index, k=1, exclude_ids=exclude_ids)[0]
                dynamic_turn = build_demo_turn(neighbor_row)
                messages = [system_msg] + dynamic_turn + fixed_turns

                raw = query_ollama(row["image_path"], messages)
                parsed, missing = parse_response(raw, lang)
                field_misses += missing
                checkpoint[row["id"]] = {"parsed": parsed, "missing": missing, "failed": 0}
            except Exception as e:
                print(f"  WARN: {row['id']} failed ({e})")
                parsed = {task: LABEL_SPACE[lang][task][0] for task in TASKS}
                request_failures += 1
                checkpoint[row["id"]] = {"parsed": parsed, "missing": 0, "failed": 1}
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f)
        for task in TASKS:
            predictions[task].append(parsed[task])
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            remaining = (elapsed / (i + 1)) * (len(eval_rows) - i - 1)
            print(f"  {lang}: {i+1}/{len(eval_rows)} done, ~{remaining:.0f}s remaining")

    scores = score_predictions(eval_rows, predictions)
    baseline_name = "retrieval_cot_vlm_" + MODEL.replace(":", "_").replace(".", "_") + f"__{run_id}"
    pilot_note = f"PILOT n={len(eval_rows)}, " if pilot_size else ""
    n_fixed = len(FIXED_EXEMPLAR_IDS[lang])
    record_result(lang, baseline_name, scores,
                   notes=f"{pilot_note}{eval_mode.upper()} data, {MODEL}, 1 dynamic CLIP-retrieved neighbor + "
                         f"{n_fixed} fixed exemplars ({FIXED_EXEMPLAR_IDS[lang]}), CoT, "
                         f"request_failures={request_failures}/{len(eval_rows)}, "
                         f"field_misses={field_misses}/{len(eval_rows)*len(TASKS)}",
                   supervision="zero_shot")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    eval_mode = os.environ.get("HASOC_EVAL_MODE", "val")
    run_id = os.environ.get("HASOC_RUN_ID", eval_mode)
    pilot_size = int(os.environ.get("HASOC_PILOT_SIZE", "35"))
    # load CLIP once, reuse across languages -- reloading per-language was
    # both wasteful and (per the v1 crash) apparently left the underlying
    # httpx client in a bad state on the second load after a transient DNS
    # hiccup on the first
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    for lang in LANG_CONFIG:
        if only and lang != only:
            continue
        run_lang(lang, clip_model, clip_proc, pilot_size=pilot_size, eval_mode=eval_mode, run_id=run_id)
