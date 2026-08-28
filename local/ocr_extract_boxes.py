"""Extract OCR text-region bounding boxes (not just flat text) and cache to
JSON, one file per language. Needed for the obfuscation ablation: masking
out (or isolating) exactly the pixels where embedded caption text sits,
rather than a crude fixed top/bottom-band guess.

Separate from ocr_extract.py (which only cached flat text via
paragraph=True/detail=0) so the already-verified text cache isn't touched --
this only adds box coordinates using the same two engines already validated:
EasyOCR for Telugu, ocr_tamil/PARSeq for Tamil.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import LANG_CONFIG, load_train_rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def boxes_for_tamil(engine, image_path):
    # verified empirically: craft_detect(image_array) returns
    # (cropped_images, prediction_result) where prediction_result is a list
    # of (quad_points[4,2], line_number) tuples -- NOT the bare quad list
    # this function originally assumed before checking the real API.
    img = np.array(Image.open(image_path).convert("RGB"))
    _, prediction_result = engine.craft_detect(img)
    boxes = []
    for quad, _line_num in prediction_result:
        xs = quad[:, 0]
        ys = quad[:, 1]
        boxes.append([float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))])
    return boxes


def boxes_for_telugu(reader, image_path):
    result = reader.readtext(image_path, detail=1, paragraph=False)
    boxes = []
    for (quad, text, conf) in result:
        # coordinates come back as a mix of python int and numpy.int32 --
        # cast to plain float so json.dump doesn't choke on numpy scalars
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1]) for p in quad]
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
    return boxes


def extract_lang(lang):
    out_path = os.path.join(ROOT, "splits", f"{lang}_ocr_boxes.json")
    rows = load_train_rows(lang)

    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = json.load(f)

    if lang == "tamil":
        from ocr_tamil.ocr import OCR
        engine = OCR(detect=True)
        get_boxes = lambda p: boxes_for_tamil(engine, p)
    else:
        import easyocr
        reader = easyocr.Reader(["te", "en"], gpu=False, verbose=False)
        get_boxes = lambda p: boxes_for_telugu(reader, p)

    todo = [r for r in rows if r["id"] not in done]
    print(f"{lang}: {len(done)} cached, {len(todo)} to process")

    for i, row in enumerate(todo):
        try:
            done[row["id"]] = get_boxes(row["image_path"])
        except Exception as e:
            print(f"  WARN: {row['id']} failed ({e})")
            done[row["id"]] = []
        if (i + 1) % 50 == 0:
            print(f"  {lang}: {i+1}/{len(todo)} done")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(done, f)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(done, f)
    print(f"{lang}: box extraction complete -> {out_path}")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for lang in LANG_CONFIG:
        if only and lang != only:
            continue
        extract_lang(lang)
