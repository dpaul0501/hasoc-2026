"""Compute TextReliance / VisualReliance per (lang, model, task) from the
obfuscation ablation results in results/comparison.csv.

  TextReliance   = (F1_full - F1_text_obfuscated)   / F1_full
  VisualReliance = (F1_full - F1_visual_obfuscated) / F1_full

Positive = performance dropped when that modality was removed (relies on it).
Near zero = removing it didn't hurt (doesn't rely on it).
Negative = performance improved when that modality was removed (it was
confusing the model / the obfuscation coincidentally correlates with label).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import ROOT
from common.metrics import TASKS

COMPARISON_CSV = os.path.join(ROOT, "results", "comparison.csv")
OUT_CSV = os.path.join(ROOT, "results", "modality_reliance.csv")

MODELS = ["clip_image_linear_probe", "dinov2_image_linear_probe", "zeroshot_vlm_qwen2_5vl"]


def load_rows():
    with open(COMPARISON_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_row(rows, lang, baseline_prefix, suffix=None):
    for r in rows:
        name = r["baseline"]
        if r["lang"] != lang:
            continue
        if suffix:
            if name == f"{baseline_prefix}__{suffix}":
                return r
        else:
            # full/unobfuscated row: exact baseline name match, possibly with
            # a _latest tag for the VLM (Ollama tag naming)
            if name == baseline_prefix or name == f"{baseline_prefix}_latest":
                return r
    return None


def main():
    rows = load_rows()
    out_rows = []
    for lang in ["tamil", "telugu"]:
        for model in MODELS:
            full = find_row(rows, lang, model)
            text_obf = find_row(rows, lang, model, "text_obfuscated")
            vis_obf = find_row(rows, lang, model, "visual_obfuscated")
            if not (full and text_obf and vis_obf):
                print(f"SKIP {lang}/{model}: missing one of full/text_obfuscated/visual_obfuscated")
                continue
            for task in TASKS:
                f1_full = float(full[f"{task}_f1"])
                f1_text = float(text_obf[f"{task}_f1"])
                f1_vis = float(vis_obf[f"{task}_f1"])
                text_reliance = (f1_full - f1_text) / f1_full if f1_full else 0.0
                visual_reliance = (f1_full - f1_vis) / f1_full if f1_full else 0.0
                out_rows.append({
                    "lang": lang, "model": model, "task": task,
                    "f1_full": round(f1_full, 4), "f1_text_obfuscated": round(f1_text, 4),
                    "f1_visual_obfuscated": round(f1_vis, 4),
                    "text_reliance": round(text_reliance, 4),
                    "visual_reliance": round(visual_reliance, 4),
                })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"{'lang':8}{'model':28}{'task':12}{'TextRel':>10}{'VisRel':>10}")
    for r in out_rows:
        print(f"{r['lang']:8}{r['model']:28}{r['task']:12}{r['text_reliance']:>10}{r['visual_reliance']:>10}")
    print(f"\nsaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
