"""Materialize text-obfuscated and visual-obfuscated versions of every DEV-set
image (both languages) to disk, using the cached OCR boxes. Train-set images
are untouched -- the CLIP/DINOv2 linear probes keep their classifier heads
trained on clean data; only the dev-time inputs are corrupted, which is the
correct design for a reliance/robustness ablation (measure how much a
classifier trained normally degrades when one modality is knocked out at
inference, not retrain on corrupted data).
"""
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import LANG_CONFIG, load_split, ROOT
from common.obfuscate import text_obfuscate, visual_obfuscate

OUT_ROOT = os.path.join(ROOT, "obfuscated")


def run_lang(lang):
    _, dev_rows = load_split(lang)
    with open(os.path.join(ROOT, "splits", f"{lang}_ocr_boxes.json"), encoding="utf-8") as f:
        boxes_by_id = json.load(f)

    text_dir = os.path.join(OUT_ROOT, lang, "text_obfuscated")
    visual_dir = os.path.join(OUT_ROOT, lang, "visual_obfuscated")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)

    n_no_boxes = 0
    for row in dev_rows:
        boxes = boxes_by_id.get(row["id"], [])
        if not boxes:
            n_no_boxes += 1
        img = Image.open(row["image_path"]).convert("RGB")
        text_obfuscate(img, boxes).save(os.path.join(text_dir, row["id"]))
        visual_obfuscate(img, boxes).save(os.path.join(visual_dir, row["id"]))

    print(f"{lang}: materialized {len(dev_rows)} x 2 obfuscated images "
          f"({n_no_boxes} had no detected text boxes, obfuscation is a no-op/all-blank for those)")


if __name__ == "__main__":
    for lang in LANG_CONFIG:
        run_lang(lang)
