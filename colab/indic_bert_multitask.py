"""Colab GPU cell: OCR text -> fine-tuned xlm-roberta-base, multi-task head.

This is the "OCR-based HuggingFace model" baseline: PaddleOCR-class engines
aren't installable on this Python build locally (see local/ocr_extract.py
docstring), so OCR happened locally (EasyOCR for Telugu, ocr_tamil/PARSeq for
Tamil) and the extracted text is cached in splits/{lang}_ocr_text.csv. This
cell is the actual HF-model half of that baseline: a shared ALBERT encoder
(xlm-roberta-base, covers both Tamil and Telugu) with 5 task-specific
linear heads trained jointly, one multi-task model per language.

SETUP (do this once in Colab, in an earlier cell):
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = '/content/drive/MyDrive/hasoc'   # <- wherever you uploaded the repo
    # repo must include: common/, splits/{lang}_split.csv, splits/{lang}_ocr_text.csv,
    # Tamil_HASOC/, Telugu_HASOC/ (train CSVs + images_all/, images not actually
    # needed by this cell but load_train_rows() builds image_path regardless)
    import sys; sys.path.insert(0, ROOT)
    !pip install -q transformers datasets accelerate scikit-learn

Then run this file's contents as the next cell(s).
"""
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ROOT must already be on sys.path from the setup cell above.
from common.data import LANG_CONFIG, load_split, ROOT
from common.metrics import TASKS, score_predictions, record_result

MODEL_NAME = "xlm-roberta-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128
EPOCHS = 8
BATCH_SIZE = 16
LR = 2e-5

# target label space differs Tamil vs Telugu -- build per-language, not global
LABEL_SPACE = {
    "tamil": {
        "sentiment": ["negative", "neutral", "positive"],
        "sarcasm": ["yes", "no"],
        "vulgar": ["vulgar", "not vulgar"],
        "abuse": ["abusive", "non-abusive"],
        "target": ["individual", "others", "social sub-groups", "political", "gender", "none"],
    },
    "telugu": {
        "sentiment": ["negative", "neutral", "positive"],
        "sarcasm": ["yes", "no"],
        "vulgar": ["vulgar", "not vulgar"],
        "abuse": ["abusive", "non-abusive"],
        "target": ["social sub-groups", "gender", "individual", "none", "others",
                    "political", "national origin", "religion"],
    },
}


def load_ocr_text(lang):
    path = os.path.join(ROOT, "splits", f"{lang}_ocr_text.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r["ocr_text"] for r in csv.DictReader(f)}


class MemeTextDataset(Dataset):
    def __init__(self, rows, ocr_text, tokenizer, label_space):
        self.rows = rows
        self.ocr_text = ocr_text
        self.tokenizer = tokenizer
        self.label_space = label_space

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = self.ocr_text.get(row["id"], "")
        enc = self.tokenizer(text, truncation=True, padding="max_length",
                              max_length=MAX_LEN, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        for task in TASKS:
            item[f"label_{task}"] = torch.tensor(self.label_space[task].index(row[task]))
        return item


class MultiTaskIndicBert(nn.Module):
    def __init__(self, label_space):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden, len(label_space[task])) for task in TASKS
        })

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        pooled = out.last_hidden_state[:, 0]  # [CLS]
        return {task: head(pooled) for task, head in self.heads.items()}


def run_lang(lang):
    label_space = LABEL_SPACE[lang]
    train_rows, dev_rows = load_split(lang)
    ocr_text = load_ocr_text(lang)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = MemeTextDataset(train_rows, ocr_text, tokenizer, label_space)
    dev_ds = MemeTextDataset(dev_rows, ocr_text, tokenizer, label_space)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MultiTaskIndicBert(label_space).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # inverse-frequency class weights per task -- vulgar/abuse are severely
    # imbalanced (e.g. tamil vulgar is 13/800), plain CE would just predict
    # the majority class every time
    class_weights = {}
    for task in TASKS:
        counts = np.array([sum(1 for r in train_rows if r[task] == lab) for lab in label_space[task]])
        weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
        class_weights[task] = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"],
                             batch.get("token_type_ids"))
            loss = sum(
                nn.functional.cross_entropy(outputs[task], batch[f"label_{task}"],
                                             weight=class_weights[task])
                for task in TASKS
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  {lang} epoch {epoch+1}/{EPOCHS} loss={total_loss/len(train_loader):.4f}")

    model.eval()
    predictions = {task: [] for task in TASKS}
    with torch.no_grad():
        for batch in dev_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"],
                             batch.get("token_type_ids"))
            for task in TASKS:
                preds = outputs[task].argmax(dim=-1).cpu().numpy()
                predictions[task].extend(label_space[task][p] for p in preds)

    scores = score_predictions(dev_rows, predictions)
    record_result(lang, "ocr_indicbert_multitask", scores,
                   notes=f"{MODEL_NAME}, {EPOCHS} epochs, class-weighted CE",
                   supervision="fine_tuned")


if __name__ == "__main__":
    for lang in LANG_CONFIG:
        print(f"=== {lang} ===")
        run_lang(lang)
