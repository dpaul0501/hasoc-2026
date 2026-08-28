"""Shared data loading for the HASOC Tamil/Telugu meme baselines.

Both the local fork and the Colab fork import this module (or copy it verbatim
into the Colab notebook's first cell) so every baseline scores against the
exact same held-out dev split.
"""
import csv
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASKS = ["sentiment", "sarcasm", "vulgar", "abuse", "target"]

LANG_CONFIG = {
    "tamil": {
        "dir": os.path.join(ROOT, "Tamil_HASOC"),
        "train_csv": "train_data_Tamil.csv",
        "test_csv": "test_data_Tamil_no.csv",
        "images_dir": "images_all",
    },
    "telugu": {
        "dir": os.path.join(ROOT, "Telugu_HASOC"),
        "train_csv": "train_data_Telugu.csv",
        "test_csv": "test_data_Telugu.csv",
        "images_dir": "images_all",
    },
}

SPLIT_SEED = 42
DEV_FRACTION = 0.2


def load_train_rows(lang):
    cfg = LANG_CONFIG[lang]
    path = os.path.join(cfg["dir"], cfg["train_csv"])
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        row["id"] = row.pop("ids")
        row["image_path"] = os.path.join(cfg["dir"], cfg["images_dir"], row["id"])
    return rows


def make_or_load_split(lang):
    """Deterministic 80/20 split of the 800 train rows, cached to splits/.

    Re-run is idempotent: if the split file already exists it's reused as-is,
    so both forks (and repeated local runs) always score the same dev rows.
    """
    split_path = os.path.join(ROOT, "splits", f"{lang}_split.csv")
    rows = load_train_rows(lang)
    by_id = {r["id"]: r for r in rows}

    if os.path.exists(split_path):
        with open(split_path, newline="", encoding="utf-8") as f:
            assignment = {r["id"]: r["split"] for r in csv.DictReader(f)}
        missing = set(by_id) - set(assignment)
        if missing:
            raise RuntimeError(
                f"{split_path} is stale ({len(missing)} ids missing); delete it to regenerate"
            )
    else:
        ids = sorted(by_id)  # sort first so shuffle is reproducible regardless of CSV row order
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(ids)
        n_dev = int(len(ids) * DEV_FRACTION)
        dev_ids = set(ids[:n_dev])
        assignment = {i: ("dev" if i in dev_ids else "train") for i in ids}
        with open(split_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "split"])
            for i in ids:
                w.writerow([i, assignment[i]])

    train_rows = [by_id[i] for i in by_id if assignment[i] == "train"]
    dev_rows = [by_id[i] for i in by_id if assignment[i] == "dev"]
    return train_rows, dev_rows


def load_split(lang):
    """Public entrypoint: returns (train_rows, dev_rows), each a list of dicts
    with keys: id, image_path, sentiment, sarcasm, vulgar, abuse, target."""
    return make_or_load_split(lang)


if __name__ == "__main__":
    for lang in LANG_CONFIG:
        train_rows, dev_rows = load_split(lang)
        print(f"{lang}: {len(train_rows)} train / {len(dev_rows)} dev")
