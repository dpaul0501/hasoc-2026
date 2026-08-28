"""Extract embedded meme text and cache to CSV.

Engine per language (picked after live testing, not guessed):
  - telugu: EasyOCR ['te','en'] -- works cleanly out of the box.
  - tamil:  EasyOCR's hosted 'tamil.pth' checkpoint has a known upstream
            size-mismatch bug (143 vs 127 output classes) against the
            current package's character list -- fails on every install/pin
            tried. Using `ocr_tamil` (PARSeq-based, github.com/gnana70/tamil_ocr)
            instead, confirmed working on a sample image.

Both the OCR+TF-IDF baseline and (after copying the CSV over) the Colab
indic-bert baseline consume this cached text, so OCR only runs once.
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import LANG_CONFIG, load_train_rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def flatten_ocr_tamil_result(result):
    # ocr_tamil.predict returns nested lists, e.g. [['word1', 'word2', ...]]
    words = []
    def _walk(x):
        if isinstance(x, str):
            words.append(x)
        else:
            for item in x:
                _walk(item)
    _walk(result)
    return " ".join(words)


def extract_lang(lang):
    out_path = os.path.join(ROOT, "splits", f"{lang}_ocr_text.csv")
    rows = load_train_rows(lang)

    done = {}
    if os.path.exists(out_path):
        with open(out_path, newline="", encoding="utf-8") as f:
            done = {r["id"]: r["ocr_text"] for r in csv.DictReader(f)}

    if lang == "tamil":
        from ocr_tamil.ocr import OCR
        engine = OCR(detect=True)
        def run(image_path):
            return flatten_ocr_tamil_result(engine.predict(image_path))
    else:
        import easyocr
        reader = easyocr.Reader(["te", "en"], gpu=False, verbose=False)
        def run(image_path):
            result = reader.readtext(image_path, detail=0, paragraph=True)
            return " ".join(result)

    todo = [r for r in rows if r["id"] not in done]
    print(f"{lang}: {len(done)} cached, {len(todo)} to OCR")

    t0 = time.time()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["id", "ocr_text"])
        for i, row in enumerate(todo):
            try:
                text = run(row["image_path"]).replace("\n", " ").strip()
            except Exception as e:
                text = ""
                print(f"  WARN: OCR failed for {row['id']}: {e}")
            w.writerow([row["id"], text])
            f.flush()
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = elapsed / (i + 1)
                remaining = rate * (len(todo) - i - 1)
                print(f"  {lang}: {i+1}/{len(todo)} done, ~{remaining:.0f}s remaining")

    print(f"{lang}: OCR extraction complete -> {out_path}")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for lang in LANG_CONFIG:
        if only and lang != only:
            continue
        extract_lang(lang)
