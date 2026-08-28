"""Colab GPU cell: late-fusion baseline -- frozen CLIP image embedding
concatenated with frozen indic-bert [CLS] text embedding, per-task
logistic-regression head. Classic pre-VLM multimodal baseline: isolates
whether simple concatenation of the two modalities beats either alone
(image-only CLIP probe: local/baseline_clip_probe.py; OCR-text-only:
colab/indic_bert_multitask.py) without any cross-attention or fine-tuning.

Run after the setup cell described in colab/indic_bert_multitask.py
(drive mount + sys.path + pip installs, plus `pip install -q pillow`).
"""
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

from common.data import LANG_CONFIG, load_split, ROOT
from common.metrics import TASKS, score_predictions, record_result

CLIP_NAME = "openai/clip-vit-base-patch32"
TEXT_NAME = "ai4bharat/indic-bert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_ocr_text(lang):
    path = os.path.join(ROOT, "splits", f"{lang}_ocr_text.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r["ocr_text"] for r in csv.DictReader(f)}


def embed_images(rows, clip_model, clip_processor, batch_size=32):
    out = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        inputs = clip_processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            # transformers>=5 returns BaseModelOutputWithPooling; .pooler_output
            # is already the projected embedding (confirmed empirically in the
            # local CLIP probe -- do not re-apply visual_projection)
            feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)


def embed_text(rows, ocr_text, tokenizer, text_model, batch_size=32, max_len=128):
    out = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        texts = [ocr_text.get(r["id"], "") for r in batch]
        enc = tokenizer(texts, truncation=True, padding="max_length",
                         max_length=max_len, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            hidden = text_model(**enc).last_hidden_state[:, 0]  # [CLS]
        out.append(hidden.cpu().numpy())
    return np.concatenate(out, axis=0)


def run_lang(lang, clip_model, clip_processor, tokenizer, text_model):
    train_rows, dev_rows = load_split(lang)
    ocr_text = load_ocr_text(lang)

    img_train = embed_images(train_rows, clip_model, clip_processor)
    img_dev = embed_images(dev_rows, clip_model, clip_processor)
    txt_train = embed_text(train_rows, ocr_text, tokenizer, text_model)
    txt_dev = embed_text(dev_rows, ocr_text, tokenizer, text_model)

    X_train = np.concatenate([img_train, txt_train], axis=1)
    X_dev = np.concatenate([img_dev, txt_dev], axis=1)

    predictions = {}
    for task in TASKS:
        y_train = [r[task] for r in train_rows]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_train, y_train)
        predictions[task] = list(clf.predict(X_dev))

    scores = score_predictions(dev_rows, predictions)
    record_result(lang, "late_fusion_clip_indicbert", scores,
                   notes=f"{CLIP_NAME} + {TEXT_NAME}, frozen, concat+logreg")


if __name__ == "__main__":
    clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_NAME)
    text_model = AutoModel.from_pretrained(TEXT_NAME).to(DEVICE).eval()

    for lang in LANG_CONFIG:
        print(f"=== {lang} ===")
        run_lang(lang, clip_model, clip_processor, tokenizer, text_model)
