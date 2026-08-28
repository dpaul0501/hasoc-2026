"""Data characterization for the FIRE paper's data-study section:
  - per-task class imbalance, per language
  - label co-occurrence (e.g. does abuse=yes correlate with vulgar=yes?)
  - per-language label-space mismatch on `target` (Telugu has religion/
    national origin categories, Tamil doesn't -- a joint multi-task head
    can't share that head's output layer across languages as-is)

Run: python3 common/analyze_data.py
"""
import itertools
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import LANG_CONFIG, TASKS, load_train_rows


def class_imbalance(rows, lang):
    print(f"\n--- {lang}: class distribution (n={len(rows)}) ---")
    for task in TASKS:
        counts = Counter(r[task] for r in rows)
        total = sum(counts.values())
        dist = ", ".join(f"{k}={v} ({v/total:.1%})" for k, v in counts.most_common())
        minority_frac = min(counts.values()) / total
        flag = "  <-- severe imbalance" if minority_frac < 0.05 else ""
        print(f"  {task}: {dist}{flag}")


def label_space_mismatch():
    print("\n--- target label-space mismatch across languages ---")
    tamil_rows = load_train_rows("tamil")
    telugu_rows = load_train_rows("telugu")
    tamil_targets = set(r["target"] for r in tamil_rows)
    telugu_targets = set(r["target"] for r in telugu_rows)
    print(f"  tamil only:  {tamil_targets - telugu_targets}")
    print(f"  telugu only: {telugu_targets - tamil_targets}")
    print(f"  shared:      {tamil_targets & telugu_targets}")
    print("  -> a joint multi-task head across languages needs either a union")
    print("     label space (with structural zeros for unseen classes) or")
    print("     per-language output heads on a shared encoder.")


def label_cooccurrence(rows, lang):
    print(f"\n--- {lang}: binary-task co-occurrence (does abuse track vulgar/sarcasm?) ---")
    binary_tasks = [t for t in TASKS if t in ("sarcasm", "vulgar", "abuse")]
    for t1, t2 in itertools.combinations(binary_tasks, 2):
        pos1 = {"vulgar", "abusive", "yes"}
        a = sum(1 for r in rows if r[t1] in pos1 and r[t2] in pos1)
        b = sum(1 for r in rows if r[t1] in pos1 and r[t2] not in pos1)
        c = sum(1 for r in rows if r[t1] not in pos1 and r[t2] in pos1)
        d = sum(1 for r in rows if r[t1] not in pos1 and r[t2] not in pos1)
        n = a + b + c + d
        # phi coefficient (Pearson correlation for 2x2 binary tables)
        denom = ((a + b) * (c + d) * (a + c) * (b + d)) ** 0.5
        phi = (a * d - b * c) / denom if denom else 0.0
        print(f"  {t1} vs {t2}: phi={phi:+.3f}  (co-positive={a}, {t1}-only={b}, {t2}-only={c}, co-negative={d})")


if __name__ == "__main__":
    for lang in LANG_CONFIG:
        rows = load_train_rows(lang)
        class_imbalance(rows, lang)
        label_cooccurrence(rows, lang)
    label_space_mismatch()
