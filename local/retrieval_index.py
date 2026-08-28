"""CLIP-embedding retrieval index over the training set, per language.
Shared foundation for:
  - retrieval-augmented dynamic few-shot (nearest-neighbor demonstrations
    instead of the fixed 3-shot set used in the pilot)
  - the agent pipeline (OCR tool + this retrieval tool + LLM/VLM reasoning)

Caches train-set CLIP embeddings to disk once (reused by every downstream
experiment) since re-embedding 640 images per language is the expensive part.
"""
import csv
import os
import pickle
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import LANG_CONFIG, ROOT, load_split

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CACHE_DIR = os.path.join(ROOT, "splits")


def embed_images(rows, model, processor, batch_size=16):
    out = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feats = model.get_image_features(**inputs).pooler_output
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)


def build_or_load_index(lang, model=None, processor=None, rows=None, cache_suffix=""):
    # cache_suffix lets callers build a SEPARATE index over a custom row
    # subset (e.g. train_core only, excluding prompt_val) without colliding
    # with the default full-640-train cache -- needed once validation-split
    # iteration started, since retrieving a prompt_val row as a demonstration
    # while evaluating on prompt_val would leak.
    cache_path = os.path.join(CACHE_DIR, f"{lang}_clip_train_index{cache_suffix}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if model is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if rows is None:
        rows, _ = load_split(lang)
    train_rows = rows
    embeddings = embed_images(train_rows, model, processor)
    # normalize for cosine similarity via dot product
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = {"rows": train_rows, "embeddings": embeddings}
    with open(cache_path, "wb") as f:
        pickle.dump(index, f)
    return index


def retrieve_neighbors(query_embedding, index, k=3, exclude_ids=None):
    q = query_embedding / np.linalg.norm(query_embedding)
    sims = index["embeddings"] @ q
    order = np.argsort(-sims)
    exclude_ids = exclude_ids or set()
    picked = []
    for idx in order:
        row = index["rows"][idx]
        if row["id"] in exclude_ids:
            continue
        picked.append((row, float(sims[idx])))
        if len(picked) == k:
            break
    return picked


def load_ocr_text(lang):
    path = os.path.join(ROOT, "splits", f"{lang}_ocr_text.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r["ocr_text"] for r in csv.DictReader(f)}


def build_or_load_text_index(lang):
    # char n-gram TF-IDF over OCR text -- same recipe as the winning
    # OCR+TF-IDF baseline. Text similarity, not visual similarity, since the
    # obfuscation ablation showed vulgar/abuse carry almost no visual signal
    # -- CLIP retrieval alone was confirmed (see __main__ check) to surface
    # visually-similar-but-wrong-class neighbors for exactly these tasks.
    cache_path = os.path.join(CACHE_DIR, f"{lang}_tfidf_train_index.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    train_rows, _ = load_split(lang)
    ocr_text = load_ocr_text(lang)
    texts = [ocr_text.get(r["id"], "") for r in train_rows]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=20000)
    matrix = vectorizer.fit_transform(texts)
    index = {"rows": train_rows, "vectorizer": vectorizer, "matrix": matrix}
    with open(cache_path, "wb") as f:
        pickle.dump(index, f)
    return index


def retrieve_text_neighbors(query_text, index, k=3, exclude_ids=None):
    q_vec = index["vectorizer"].transform([query_text])
    sims = (index["matrix"] @ q_vec.T).toarray().flatten()
    order = np.argsort(-sims)
    exclude_ids = exclude_ids or set()
    picked = []
    for idx in order:
        row = index["rows"][idx]
        if row["id"] in exclude_ids:
            continue
        picked.append((row, float(sims[idx])))
        if len(picked) == k:
            break
    return picked


if __name__ == "__main__":
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    for lang in LANG_CONFIG:
        clip_index = build_or_load_index(lang, model, processor)
        text_index = build_or_load_text_index(lang)
        ocr_text = load_ocr_text(lang)
        print(f"{lang}: indexed {len(clip_index['rows'])} training images (CLIP + TF-IDF)")

        query_row = next(r for r in clip_index["rows"] if r["vulgar"] == "vulgar")
        print(f"  query: {query_row['id']} (vulgar={query_row['vulgar']}, abuse={query_row['abuse']})")

        q_emb = embed_images([query_row], model, processor)[0]
        clip_neighbors = retrieve_neighbors(q_emb, clip_index, k=3, exclude_ids={query_row["id"]})
        print("  CLIP (visual) neighbors:")
        for row, sim in clip_neighbors:
            print(f"    {row['id']} sim={sim:.3f} vulgar={row['vulgar']} abuse={row['abuse']}")

        q_text = ocr_text.get(query_row["id"], "")
        text_neighbors = retrieve_text_neighbors(q_text, text_index, k=3, exclude_ids={query_row["id"]})
        print("  TF-IDF (text) neighbors:")
        for row, sim in text_neighbors:
            print(f"    {row['id']} sim={sim:.3f} vulgar={row['vulgar']} abuse={row['abuse']}")
