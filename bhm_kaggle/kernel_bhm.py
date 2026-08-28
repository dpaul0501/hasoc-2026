"""Kaggle kernel: fine-tune xlm-roberta-base for binary hate classification
on BHM Bengali Hateful Memes captions. Same recipe proven working for HASOC
(kernel_tamil.py/kernel_telugu.py), including the CPU-fallback fix for the
P100/sm_60 CUDA-kernel-image bug found there. Real gradient descent, not a
probe -- BHM's 5758-example, well-balanced (37% hate) train set is a much
more favorable setting for fine-tuning than HASOC's 640-example, severely
imbalanced one, so this is the fair test of whether fine-tuning can actually
beat the simple TF-IDF baseline (macro_f1=0.6469, hate_f1=0.5694) when data
size/balance isn't the limiting factor.
"""
import csv
import glob
import os
import sys

DATASET_SLUG = "bhm-bengali-lean"
print("=== /kaggle/input contents ===", os.listdir("/kaggle/input"))
candidates = glob.glob("/kaggle/input/**/train.csv", recursive=True)
if not candidates:
    raise RuntimeError(f"no mounted dataset contains train.csv -- saw: {os.listdir('/kaggle/input')}")
DATA_DIR = os.path.dirname(candidates[0])
print("using DATA_DIR:", DATA_DIR)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 128
EPOCHS = 10
BATCH_SIZE = 32
LR = 2e-5
LABELS = ["non-hate", "hate"]


def pick_device():
    # HASOC kernels hit "CUDA error: no kernel image is available for
    # execution on the device" on a Tesla P100 (sm_60) with this same
    # preinstalled torch build -- test before committing to cuda.
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda") + torch.zeros(1, device="cuda")
        return "cuda"
    except RuntimeError as e:
        print(f"cuda present but unusable ({e}), falling back to cpu")
        return "cpu"


DEVICE = pick_device()


def load_csv(name):
    with open(os.path.join(DATA_DIR, f"{name}.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class CaptionDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(row["text"], truncation=True, padding="max_length",
                              max_length=MAX_LEN, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["label"] = torch.tensor(LABELS.index(row["label"]))
        return item


class HateClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.head = nn.Linear(self.encoder.config.hidden_size, len(LABELS))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # BERT/ELECTRA-style tokenizers (BanglaBERT, MuRIL, IndicBERTv2) emit
        # token_type_ids; RoBERTa-style ones (xlm-roberta-base) don't -- the
        # original HASOC kernels handled this correctly (optional kwarg),
        # this rewrite dropped it, which is exactly why BanglaBERT crashed
        # in evaluate()'s **batch_gpu call while xlm-roberta-base didn't.
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        pooled = out.last_hidden_state[:, 0]
        return self.head(pooled)


def evaluate(model, loader):
    model.eval()
    preds, y_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(DEVICE) for k, v in batch.items() if k != "label"}
            logits = model(**batch_gpu)
            preds.extend(LABELS[p] for p in logits.argmax(dim=-1).cpu().numpy())
            y_true.extend(LABELS[l] for l in batch["label"].numpy())
    model.train()
    return preds, y_true, f1_score(y_true, preds, average="macro")


def main():
    print(f"device: {DEVICE}")
    train_rows = load_csv("train")
    valid_rows = load_csv("valid")
    test_rows = load_csv("test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = CaptionDataset(train_rows, tokenizer)
    valid_ds = CaptionDataset(valid_rows, tokenizer)
    test_ds = CaptionDataset(test_rows, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = HateClassifier().to(DEVICE)
    # v3's fix (warmup schedule object + no class weights) STILL collapsed
    # to identical degenerate output (recall=1.0, precision=0.374, bit-for-
    # bit identical loss every epoch) -- turned out `scheduler` was created
    # but scheduler.step() was never actually called anywhere, so the
    # intended warmup never took effect. Root-caused locally (not on paid
    # compute) by reproducing the collapse on a 100-example subset: the
    # randomly-initialized head was using the SAME tiny LR as the pretrained
    # encoder, so it couldn't move fast enough out of its random starting
    # bias before the whole system locked into a stable degenerate optimum.
    # Real fix, confirmed to actually escape collapse in local testing:
    # differential LR (head trains much faster than encoder) + actually
    # calling scheduler.step() + gradient clipping to stop early large
    # updates from overshooting into a bad basin.
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": LR},
        {"params": model.head.parameters(), "lr": 5e-4},
    ])
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, step / warmup_steps)
    )

    counts = np.array([sum(1 for r in train_rows if r["label"] == lab) for lab in LABELS])
    print("class counts:", dict(zip(LABELS, counts)))
    # v1 used inverse-frequency class weights (0.79/1.36) -- on this dataset's
    # mild 63/37 split (not HASOC's severe 98/2), that makes "always predict
    # hate" and "always predict non-hate" have almost IDENTICAL expected
    # weighted loss (0.499 vs 0.500, computed by hand from the class
    # proportions), so the optimizer had no real gradient pulling it out of
    # either degenerate constant-output solution -- confirmed empirically:
    # v1 collapsed to predicting "hate" for all 711 test examples (100%
    # recall, 37% precision) with loss stuck near ln(2) the whole run. Balance
    # this mild isn't worth the risk -- dropping the reweighting, adding a
    # warmup schedule so early large gradients don't lock in a bad
    # equilibrium, and more epochs to make sure it has room to actually learn.

    # v2's fix (drop class weights + warmup) still collapsed to predicting
    # "hate" for literally every test example, identical numbers to v1
    # (recall=1.0, precision=0.374) despite a different loss trajectory --
    # a local debug run confirmed gradients DO flow and logits DO
    # differentiate early in training, so this isn't a dead-gradient bug;
    # it's collapse emerging somewhere across the full 10-epoch/5758-example
    # run. Real fix: track macro-F1 on BHM's own "valid" split (640
    # examples, unused until now) after every epoch, keep the best-scoring
    # checkpoint, and report THAT on test instead of just whatever the
    # final epoch happened to land on.
    best_val_f1 = -1.0
    best_state = None
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**{k: v for k, v in batch.items() if k != "label"})
            loss = nn.functional.cross_entropy(logits, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        _, _, val_f1 = evaluate(model, valid_loader)
        print(f"epoch {epoch+1}/{EPOCHS} loss={total_loss/len(train_loader):.4f} val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    print(f"\nbest val_macro_f1={best_val_f1:.4f}, restoring that checkpoint for final test eval")
    model.load_state_dict(best_state)
    model.to(DEVICE)

    preds, y_true, macro_f1 = evaluate(model, test_loader)
    p, r, f1, support = precision_recall_fscore_support(y_true, preds, labels=["hate"], zero_division=0)
    print(f"\nFINAL: macro_f1={macro_f1:.4f}")
    print(f"hate class: precision={p[0]:.4f} recall={r[0]:.4f} f1={f1[0]:.4f} support={support[0]}")

    with open("/kaggle/working/bhm_result.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "macro_f1", "hate_precision", "hate_recall", "hate_f1"])
        w.writerow([MODEL_NAME, round(macro_f1, 4), round(p[0], 4), round(r[0], 4), round(f1[0], 4)])


if __name__ == "__main__":
    main()
