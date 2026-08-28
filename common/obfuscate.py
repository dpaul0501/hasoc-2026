"""Image obfuscation for the modality-reliance ablation.

Two conditions per meme, built from the cached OCR bounding boxes
(splits/{lang}_ocr_boxes.json, produced by local/ocr_extract_boxes.py):

  - text_obfuscate:   black out the caption-text regions, keep everything
                       else -- "can you still tell it's hateful without
                       reading the words?"
  - visual_obfuscate: keep only the caption-text regions (on a blank grey
                       canvas), black out everything else -- "is the crop of
                       just the caption enough, without the surrounding
                       scene/iconography?"

Boxes are padded slightly since CRAFT/EasyOCR boxes are tight around glyphs
and a few leftover pixels at the edges would leak signal.
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

PAD_PX = 6


def load_boxes(lang, root):
    path = os.path.join(root, "splits", f"{lang}_ocr_boxes.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _padded(box, w, h):
    x0, y0, x1, y1 = box
    return (
        max(0, x0 - PAD_PX),
        max(0, y0 - PAD_PX),
        min(w, x1 + PAD_PX),
        min(h, y1 + PAD_PX),
    )


def text_obfuscate(image: Image.Image, boxes: list) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for box in boxes:
        draw.rectangle(_padded(box, w, h), fill=(0, 0, 0))
    return img


def visual_obfuscate(image: Image.Image, boxes: list) -> Image.Image:
    w, h = image.size
    canvas = Image.new("RGB", (w, h), (128, 128, 128))
    for box in boxes:
        x0, y0, x1, y1 = (int(v) for v in _padded(box, w, h))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = image.crop((x0, y0, x1, y1))
        canvas.paste(crop, (x0, y0))
    return canvas
