"""Kaggle kernel: fine-tune xlm-roberta-base (multi-task, 5 HASOC subtasks)
on Tamil OCR text. Mirror of colab/indic_bert_multitask.py, adapted for
Kaggle's dataset-mount layout instead of Google Drive.

Dataset is mounted at /kaggle/input/<slug>/ with the same folder structure
as the local repo (common/, splits/, Tamil_HASOC/) -- common/data.py's ROOT
resolution (derived from __file__) works unmodified once sys.path points
here. Only override needed: RESULTS_CSV, since /kaggle/input/ is read-only
and results must go to /kaggle/working/ instead.
"""
import glob
import os
import sys

# don't hardcode the mount path -- v1 guessed /kaggle/input/<slug> and got
# ModuleNotFoundError: 'common'; v2's one-level glob found /kaggle/input
# contains just ['datasets'], i.e. the real layout has an extra nesting
# level (/kaggle/input/datasets/<slug>/...). Recursive search instead of
# guessing a fixed depth a third time.
print("=== /kaggle/input contents ===", os.listdir("/kaggle/input"))
candidates = glob.glob("/kaggle/input/**/common", recursive=True)
if not candidates:
    raise RuntimeError(f"no mounted dataset contains common/ -- saw: {os.listdir('/kaggle/input')}")
dataset_root = os.path.dirname(candidates[0])
print("using dataset_root:", dataset_root, "contents:", os.listdir(dataset_root))
sys.path.insert(0, dataset_root)

import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

import common.metrics as metrics_mod
metrics_mod.RESULTS_CSV = "/kaggle/working/comparison_tamil.csv"

from common.data import load_split, ROOT
from common.metrics import TASKS, score_predictions, record_result

MODEL_NAME = "xlm-roberta-base"


def pick_device():
    # v4 crashed with "CUDA error: no kernel image is available for execution
    # on the device" -- Kaggle assigned a Tesla P100 (sm_60, Pascal) this
    # time, and the preinstalled torch build only ships kernels for
    # sm_70+. Rather than gamble on which GPU a future run gets, actually
    # test a real op before committing to cuda; fall back to CPU if it fails.
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda") + torch.zeros(1, device="cuda")
        return "cuda"
    except RuntimeError as e:
        print(f"cuda present but unusable ({e}), falling back to cpu")
        return "cpu"


DEVICE = pick_device()
MAX_LEN = 128
EPOCHS = 8
BATCH_SIZE = 16
LR = 2e-5
LANG = "tamil"

LABEL_SPACE = {
    "sentiment": ["negative", "neutral", "positive"],
    "sarcasm": ["yes", "no"],
    "vulgar": ["vulgar", "not vulgar"],
    "abuse": ["abusive", "non-abusive"],
    "target": ["individual", "others", "social sub-groups", "political", "gender", "none"],
}


def load_ocr_text():
    path = os.path.join(ROOT, "splits", f"{LANG}_ocr_text.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r["ocr_text"] for r in csv.DictReader(f)}


class MemeTextDataset(Dataset):
    def __init__(self, rows, ocr_text, tokenizer):
        self.rows = rows
        self.ocr_text = ocr_text
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = self.ocr_text.get(row["id"], "")
        enc = self.tokenizer(text, truncation=True, padding="max_length",
                              max_length=MAX_LEN, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        for task in TASKS:
            item[f"label_{task}"] = torch.tensor(LABEL_SPACE[task].index(row[task]))
        return item


class MultiTaskIndicBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden, len(LABEL_SPACE[task])) for task in TASKS
        })

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        pooled = out.last_hidden_state[:, 0]
        return {task: head(pooled) for task, head in self.heads.items()}


def main():
    print(f"device: {DEVICE}")
    train_rows, dev_rows = load_split(LANG)
    ocr_text = load_ocr_text()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = MemeTextDataset(train_rows, ocr_text, tokenizer)
    dev_ds = MemeTextDataset(dev_rows, ocr_text, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MultiTaskIndicBert().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    class_weights = {}
    for task in TASKS:
        counts = np.array([sum(1 for r in train_rows if r[task] == lab) for lab in LABEL_SPACE[task]])
        weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
        class_weights[task] = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
            loss = sum(
                nn.functional.cross_entropy(outputs[task], batch[f"label_{task}"], weight=class_weights[task])
                for task in TASKS
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}/{EPOCHS} loss={total_loss/len(train_loader):.4f}")

    model.eval()
    predictions = {task: [] for task in TASKS}
    with torch.no_grad():
        for batch in dev_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
            for task in TASKS:
                preds = outputs[task].argmax(dim=-1).cpu().numpy()
                predictions[task].extend(LABEL_SPACE[task][p] for p in preds)

    scores = score_predictions(dev_rows, predictions)
    record_result(LANG, "ocr_indicbert_multitask", scores,
                   notes=f"{MODEL_NAME}, {EPOCHS} epochs, class-weighted CE, kaggle",
                   supervision="fine_tuned")
    print("DONE:", scores)


if __name__ == "__main__":
    main()
