"""Translate cached OCR text (Tamil/Telugu, code-mixed) to English via a
local Ollama text model, cache to splits/{lang}_ocr_text_en.csv.

Answers: does normalizing code-mixed low-resource text to English (a
high-resource pivot) before classification help or hurt, versus processing
the native-script text directly (local/baseline_tfidf.py)?
"""
import csv
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import LANG_CONFIG, ROOT

TRANSLATE_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/chat"


def translate(text, lang):
    if not text.strip():
        return ""
    prompt = (
        f"Translate the following {lang.capitalize()} (code-mixed with English) "
        f"meme caption text to English. Output ONLY the translation, no explanation, "
        f"no quotes:\n\n{text}"
    )
    resp = requests.post(OLLAMA_URL, json={
        "model": TRANSLATE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }, timeout=60)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def run_lang(lang):
    src_path = os.path.join(ROOT, "splits", f"{lang}_ocr_text.csv")
    out_path = os.path.join(ROOT, "splits", f"{lang}_ocr_text_en.csv")

    with open(src_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    done = {}
    if os.path.exists(out_path):
        with open(out_path, newline="", encoding="utf-8") as f:
            done = {r["id"]: r["ocr_text_en"] for r in csv.DictReader(f)}

    todo = [r for r in rows if r["id"] not in done]
    print(f"{lang}: {len(done)} cached, {len(todo)} to translate")

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["id", "ocr_text_en"])
        for i, row in enumerate(todo):
            try:
                en = translate(row["ocr_text"], lang).replace("\n", " ")
            except Exception as e:
                en = ""
                print(f"  WARN: {row['id']} failed ({e})")
            w.writerow([row["id"], en])
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"  {lang}: {i+1}/{len(todo)} translated")

    print(f"{lang}: translation complete -> {out_path}")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for lang in LANG_CONFIG:
        if only and lang != only:
            continue
        run_lang(lang)
