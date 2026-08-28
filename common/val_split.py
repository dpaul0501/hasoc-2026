"""Carves the 640-example TRAIN split further into train_core / prompt_val,
so prompt/retrieval design iteration happens on prompt_val, not the 160-
example dev set we report final numbers on. Without this, "which prompt
design works best" was being decided by looking directly at dev-set scores
-- a real methodological leak (implicitly fitting the prompt to the exact
data used for the reported metric).

dev (160) stays completely untouched until a design is finalized.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import load_split

VAL_FRACTION = 0.22  # ~140/640
VAL_SEED = 99


def make_val_split(lang):
    train_rows, dev_rows = load_split(lang)
    ids = sorted(r["id"] for r in train_rows)
    rng = random.Random(VAL_SEED)
    rng.shuffle(ids)
    n_val = int(len(ids) * VAL_FRACTION)
    val_ids = set(ids[:n_val])
    by_id = {r["id"]: r for r in train_rows}
    train_core = [by_id[i] for i in ids if i not in val_ids]
    prompt_val = [by_id[i] for i in ids if i in val_ids]
    return train_core, prompt_val, dev_rows


if __name__ == "__main__":
    from common.data import LANG_CONFIG
    for lang in LANG_CONFIG:
        train_core, prompt_val, dev_rows = make_val_split(lang)
        print(f"{lang}: train_core={len(train_core)} prompt_val={len(prompt_val)} dev={len(dev_rows)} (untouched)")
        for task in ["vulgar", "abuse"]:
            n_pos = sum(1 for r in prompt_val if r[task] in ("vulgar", "abusive"))
            print(f"  prompt_val {task} positives: {n_pos}")
