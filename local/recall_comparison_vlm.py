"""Same recall/precision breakdown as recall_comparison.py, but for a
zero-shot VLM -- completes the "from a protective standpoint, what's
actually winning" comparison across every method tried, not just the fast
local ones.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import precision_recall_fscore_support

from common.data import LANG_CONFIG, load_split
from common.metrics import TASKS
from local.baseline_vlm_ollama import LABEL_SPACE, MODEL, build_prompt, parse_response, query_ollama

POSITIVE = {"vulgar": "vulgar", "abuse": "abusive"}


def run_lang(lang):
    _, dev_rows = load_split(lang)
    prompt = build_prompt(lang)

    predictions = {task: [] for task in TASKS}
    for row in dev_rows:
        try:
            raw = query_ollama(row["image_path"], prompt)
            parsed, _ = parse_response(raw, lang)
        except Exception as e:
            print(f"  WARN: {row['id']} failed ({e})")
            parsed = {task: LABEL_SPACE[lang][task][0] for task in TASKS}
        for task in TASKS:
            predictions[task].append(parsed[task])

    for task in POSITIVE:
        pos = POSITIVE[task]
        y_true = [r[task] for r in dev_rows]
        y_pred = predictions[task]
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[pos], zero_division=0)
        n_pos = sum(1 for y in y_true if y == pos)
        print(f"{lang:8}{MODEL:28}{task:8}precision={p[0]:.3f}  recall={r[0]:.3f}  "
              f"f1={f1[0]:.3f}  (caught {int(r[0]*n_pos)}/{n_pos} true positives)")


if __name__ == "__main__":
    print(f"MODEL: {MODEL}")
    for lang in LANG_CONFIG:
        run_lang(lang)
