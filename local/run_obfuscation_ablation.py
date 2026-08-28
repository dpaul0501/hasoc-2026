"""Modality-reliance ablation: re-evaluate CLIP probe, DINOv2 probe, and the
best zero-shot VLM (qwen2.5vl) on the obfuscated dev sets, then compute
TextReliance / VisualReliance per task -- this is the FIRE paper's
mathematically-grounded contribution.

Classifier heads (CLIP/DINOv2 logistic regression) are trained on CLEAN
train-set images, exactly as in the original baselines -- only dev-time
inputs are swapped to the obfuscated versions. This measures inference-time
reliance on each modality for an otherwise-normally-trained model, not
"what happens if you train on corrupted data" (a different, less relevant
question).
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VLM_MODEL = "qwen2.5vl:latest"
# baseline_vlm_ollama.MODEL is resolved from this env var at import time --
# must be set BEFORE the import below, or query_ollama() silently falls back
# to its llava default regardless of VLM_MODEL above (caught before running,
# not after, this time).
os.environ["HASOC_VLM_MODEL"] = VLM_MODEL

from common.data import LANG_CONFIG, ROOT, load_split
from common.metrics import TASKS, record_result, score_predictions
from local.baseline_vlm_ollama import LABEL_SPACE, MODEL, build_prompt, parse_response, query_ollama

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
assert MODEL == VLM_MODEL, f"env var override failed: MODEL={MODEL!r}"


def obfuscated_path(lang, condition, image_id):
    return os.path.join(ROOT, "obfuscated", lang, condition, image_id)


def embed_clip(rows, model, processor, batch_size=16):
    out = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        images = [Image.open(r["_path"]).convert("RGB") for r in batch]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feats = model.get_image_features(**inputs).pooler_output
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)


def embed_dinov2(rows, model, processor, batch_size=16):
    out = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        images = [Image.open(r["_path"]).convert("RGB") for r in batch]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            o = model(**inputs)
            feats = o.pooler_output if o.pooler_output is not None else o.last_hidden_state[:, 0]
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)


def run_probe_condition(lang, condition, clip_model, clip_proc, dino_model, dino_proc, results):
    train_rows, dev_rows = load_split(lang)
    dev_rows_cond = [dict(r, _path=obfuscated_path(lang, condition, r["id"])) for r in dev_rows]
    train_rows_clean = [dict(r, _path=r["image_path"]) for r in train_rows]

    for name, embed_fn, model, proc in [
        ("clip_image_linear_probe", embed_clip, clip_model, clip_proc),
        ("dinov2_image_linear_probe", embed_dinov2, dino_model, dino_proc),
    ]:
        X_train = embed_fn(train_rows_clean, model, proc)
        X_dev = embed_fn(dev_rows_cond, model, proc)
        predictions = {}
        for task in TASKS:
            y_train = [r[task] for r in train_rows]
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(X_train, y_train)
            predictions[task] = list(clf.predict(X_dev))
        scores = score_predictions(dev_rows, predictions)
        baseline_name = f"{name}__{condition}"
        record_result(lang, baseline_name, scores, notes=f"dev images {condition}",
                       supervision="linear_probe_supervised")
        results[(lang, name, condition)] = scores


def run_vlm_condition(lang, condition, results):
    _, dev_rows = load_split(lang)
    prompt = build_prompt(lang)
    predictions = {task: [] for task in TASKS}
    request_failures = 0
    field_misses = 0
    for row in dev_rows:
        img_path = obfuscated_path(lang, condition, row["id"])
        try:
            raw = query_ollama(img_path, prompt)
            parsed, missing = parse_response(raw, lang)
            field_misses += missing
        except Exception as e:
            print(f"  WARN: {row['id']} ({condition}) failed ({e})")
            parsed = {task: LABEL_SPACE[lang][task][0] for task in TASKS}
            request_failures += 1
        for task in TASKS:
            predictions[task].append(parsed[task])
    scores = score_predictions(dev_rows, predictions)
    baseline_name = f"zeroshot_vlm_qwen2_5vl__{condition}"
    record_result(lang, baseline_name, scores,
                   notes=f"dev images {condition}, request_failures={request_failures}/{len(dev_rows)}, "
                         f"field_misses={field_misses}/{len(dev_rows)*len(TASKS)}",
                   supervision="zero_shot")
    results[(lang, "zeroshot_vlm_qwen2_5vl", condition)] = scores


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-vlm", action="store_true", help="skip the (slow) VLM condition runs")
    args = parser.parse_args()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
    dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    results = {}
    for lang in LANG_CONFIG:
        for condition in ["text_obfuscated", "visual_obfuscated"]:
            print(f"=== {lang} / {condition} (probes) ===")
            run_probe_condition(lang, condition, clip_model, clip_proc, dino_model, dino_proc, results)
            if not args.skip_vlm:
                print(f"=== {lang} / {condition} (qwen2.5vl) ===")
                run_vlm_condition(lang, condition, results)

    with open(os.path.join(ROOT, "results", "obfuscation_raw_scores.json"), "w") as f:
        json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in results.items()}, f, indent=2)
