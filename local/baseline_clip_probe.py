"""Image-only baseline: frozen CLIP embeddings + logistic-regression linear probe.

Isolates how much signal is in the image alone (no text/OCR), on Apple
Silicon MPS. Serves as the "vision floor" the OCR-text and VLM baselines
should beat if multimodality is actually paying off.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import LANG_CONFIG, load_split
from common.metrics import TASKS, score_predictions, record_result

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def embed_images(rows, model, processor, batch_size=16):
    embeddings = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            # NOTE: in transformers>=5, get_image_features returns a
            # BaseModelOutputWithPooling whose .pooler_output is already the
            # projected 512-dim embedding (not the raw 768-dim vision hidden
            # state) -- confirmed empirically, do not re-apply visual_projection.
            feats = model.get_image_features(**inputs).pooler_output
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
    record_result(lang, "clip_image_linear_probe", scores, notes=f"{MODEL_NAME}, device={DEVICE}")


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    for lang in LANG_CONFIG:
        run_lang(lang, model, processor)
