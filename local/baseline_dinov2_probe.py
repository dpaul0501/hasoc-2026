"""Image-only baseline: frozen DINOv2 embeddings + logistic-regression linear probe.

Same recipe as baseline_clip_probe.py but a different vision pretraining
objective -- DINOv2 is self-supervised (no text alignment), so it should
pick up dense/structural visual cues (iconography, layout, gestures) that
CLIP's text-contrastive training might under-weight if they're not verbally
salient. Comparing the two probes' per-task F1 is the ablation point, not
either number in isolation.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import LANG_CONFIG, load_split
from common.metrics import TASKS, score_predictions, record_result

MODEL_NAME = "facebook/dinov2-base"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def embed_images(rows, model, processor, batch_size=16):
    embeddings = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
            # DINOv2 pooler_output is the CLS token pooled -- confirmed a raw
            # 768-dim feature (not projected/contrastive like CLIP), so no
            # extra projection step needed here.
            feats = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
        embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def run_lang(lang, model, processor):
    train_rows, dev_rows = load_split(lang)

    X_train = embed_images(train_rows, model, processor)
    X_dev = embed_images(dev_rows, model, processor)

    predictions = {}
    for task in TASKS:
        y_train = [r[task] for r in train_rows]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_train, y_train)
        predictions[task] = list(clf.predict(X_dev))

    scores = score_predictions(dev_rows, predictions)
    record_result(lang, "dinov2_image_linear_probe", scores, notes=f"{MODEL_NAME}, device={DEVICE}")


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    for lang in LANG_CONFIG:
        run_lang(lang, model, processor)
