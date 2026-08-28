"""From a protective/moderation standpoint, macro-F1 is the wrong lens for
vulgar/abuse (see calibration_conformal.py findings) -- what matters is:
of the actually-vulgar/actually-abusive memes, how many does each approach
catch (recall), and at what false-alarm cost (precision)?

Covers every FAST local baseline (majority, OCR+TFIDF, CLIP probe, DINOv2
probe) directly. VLM zero-shot models are added separately since re-querying
160 images/model is slow -- see recall_comparison_vlm.py.
"""
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import LANG_CONFIG, ROOT, load_split

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
POSITIVE = {"vulgar": "vulgar", "abuse": "abusive"}


def load_ocr_text(lang):
    path = os.path.join(ROOT, "splits", f"{lang}_ocr_text.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r["ocr_text"] for r in csv.DictReader(f)}


def report(lang, method, task, y_true, y_pred):
    pos = POSITIVE[task]
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[pos], zero_division=0
    )
    n_pos = sum(1 for y in y_true if y == pos)
    n_flagged = sum(1 for y in y_pred if y == pos)
    print(f"{lang:8}{method:28}{task:8}precision={p[0]:.3f}  recall={r[0]:.3f}  "
          f"f1={f1[0]:.3f}  (caught {int(r[0]*n_pos)}/{n_pos} true positives, flagged {n_flagged} total)")
    return {"lang": lang, "method": method, "task": task, "precision": round(float(p[0]), 4),
            "recall": round(float(r[0]), 4), "f1": round(float(f1[0]), 4), "n_pos": n_pos}


def embed_clip(rows, model, processor, batch_size=16):
    out = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feats = model.get_image_features(**inputs).pooler_output
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)


def embed_dinov2(rows, model, processor, batch_size=16):
    out = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            o = model(**inputs)
            feats = o.pooler_output if o.pooler_output is not None else o.last_hidden_state[:, 0]
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    all_results = []
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
    dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    for lang in LANG_CONFIG:
        train_rows, dev_rows = load_split(lang)
        ocr_text = load_ocr_text(lang)
        y_dev = {task: [r[task] for r in dev_rows] for task in POSITIVE}

        # majority baseline
        for task in POSITIVE:
            y_train = [r[task] for r in train_rows]
            majority = max(set(y_train), key=y_train.count)
            preds = [majority] * len(dev_rows)
            all_results.append(report(lang, "majority_class", task, y_dev[task], preds))

        # OCR + TF-IDF
        X_train_text = [ocr_text.get(r["id"], "") for r in train_rows]
        X_dev_text = [ocr_text.get(r["id"], "") for r in dev_rows]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=20000)
        X_train_tfidf = vec.fit_transform(X_train_text)
        X_dev_tfidf = vec.transform(X_dev_text)
        for task in POSITIVE:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(X_train_tfidf, [r[task] for r in train_rows])
            preds = clf.predict(X_dev_tfidf)
            all_results.append(report(lang, "ocr_tfidf_logreg", task, y_dev[task], preds))

        # CLIP probe
        X_train_clip = embed_clip(train_rows, clip_model, clip_proc)
        X_dev_clip = embed_clip(dev_rows, clip_model, clip_proc)
        for task in POSITIVE:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(X_train_clip, [r[task] for r in train_rows])
            preds = clf.predict(X_dev_clip)
            all_results.append(report(lang, "clip_image_linear_probe", task, y_dev[task], preds))

        # DINOv2 probe
        X_train_dino = embed_dinov2(train_rows, dino_model, dino_proc)
        X_dev_dino = embed_dinov2(dev_rows, dino_model, dino_proc)
        for task in POSITIVE:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(X_train_dino, [r[task] for r in train_rows])
            preds = clf.predict(X_dev_dino)
            all_results.append(report(lang, "dinov2_image_linear_probe", task, y_dev[task], preds))

    out_path = os.path.join(ROOT, "results", "recall_comparison.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
