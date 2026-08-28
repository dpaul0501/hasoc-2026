"""Kaggle kernel: fine-tune xlm-roberta-base (multi-task, 5 HASOC subtasks)
on Tamil OCR text -- v2, fixing the same training-collapse bugs found and
fixed on the BHM Bengali side-investigation:

1. Single LR for both the pretrained encoder AND the 5 randomly-initialized
   task heads -- the heads can't move fast enough out of their random
   initial bias before the whole system locks into a degenerate optimum.
   Fix: differential LR (heads train ~25x faster than the encoder).
2. No gradient clipping -- early large updates can overshoot into a bad
   basin. Fix: clip_grad_norm_.
3. No validation-based early stopping -- v1 just reported whatever the
   final epoch (8/8) happened to land on, which on BHM turned out to be a
   collapsed, degenerate state more often than not. Fix: carve the 640
   train rows further into train_core/early-stop-val (via
   common/val_split.py, already built for the retrieval-CoT experiments),
   track overall_macro_f1 each epoch, keep the best checkpoint. Final
   numbers are still reported on the untouched 160-example dev split, same
   as every other HASOC baseline, for a fair comparison.

v1's results (Tamil overall=0.4653, Telugu overall=0.4255) are exactly the
kind of numbers this bug produces -- some task heads (e.g. sarcasm) landing
fine while others (target) get stuck -- so this v2 run is the real test of
what fine-tuning can achieve here, not v1.
"""
import glob
import os
import sys

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
from common.val_split import make_val_split

MODEL_NAME = "xlm-roberta-base"


def pick_device():
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
EPOCHS = 12
BATCH_SIZE = 16
LR = 2e-5
HEAD_LR = 5e-4
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


def run_eval(model, loader, rows):
    model.eval()
    predictions = {task: [] for task in TASKS}
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
            for task in TASKS:
                preds = outputs[task].argmax(dim=-1).cpu().numpy()
                predictions[task].extend(LABEL_SPACE[task][p] for p in preds)
    model.train()
    scores = score_predictions(rows, predictions)
    return predictions, scores


def main():
    print(f"device: {DEVICE}")
    train_core, early_val, _ = make_val_split(LANG)
    _, dev_rows = load_split(LANG)  # untouched dev, used ONLY for the final report
    ocr_text = load_ocr_text()
    print(f"train_core={len(train_core)} early_val={len(early_val)} dev(final)={len(dev_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = MemeTextDataset(train_core, ocr_text, tokenizer)
    val_ds = MemeTextDataset(early_val, ocr_text, tokenizer)
    dev_ds = MemeTextDataset(dev_rows, ocr_text, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MultiTaskIndicBert().to(DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": LR},
        {"params": model.heads.parameters(), "lr": HEAD_LR},
    ])
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, step / warmup_steps)
    )

    class_weights = {}
    for task in TASKS:
        counts = np.array([sum(1 for r in train_core if r[task] == lab) for lab in LABEL_SPACE[task]])
        weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
        class_weights[task] = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print("class weights:", {t: w.cpu().numpy().tolist() for t, w in class_weights.items()})

    best_val_f1 = -1.0
    best_state = None
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        _, val_scores = run_eval(model, val_loader, early_val)
        val_f1 = val_scores["overall_macro_f1"]
        print(f"epoch {epoch+1}/{EPOCHS} loss={total_loss/len(train_loader):.4f} val_overall_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    print(f"\nbest val_overall_macro_f1={best_val_f1:.4f}, restoring that checkpoint for final dev eval")
    model.load_state_dict(best_state)
    model.to(DEVICE)

    predictions, scores = run_eval(model, dev_loader, dev_rows)
    record_result(LANG, "ocr_indicbert_multitask_v2", scores,
                   notes=f"{MODEL_NAME}, {EPOCHS} epochs, differential LR + grad clip + early stop, kaggle",
                   supervision="fine_tuned")
    print("DONE:", scores)


if __name__ == "__main__":
    main()
