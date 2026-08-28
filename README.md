# HASOC 2026 — Tamil/Telugu Hateful Meme Classification

Baseline study and fine-tuning experiments for the FIRE 2026 HASOC Tamil/Telugu
hateful meme classification task (sentiment, sarcasm, vulgarity, abuse,
target), extended to Bengali (BHM) for cross-lingual coverage.

**Write-up:** [`paper/hasoc2026_working_notes.md`](paper/hasoc2026_working_notes.md)

## Findings, in brief

- A simple OCR/caption text + TF-IDF + logistic regression baseline
  outperforms every zero-shot VLM/LLM configuration tested (7 VLM variants,
  1 text LLM, few-shot, chain-of-thought, retrieval augmentation) across all
  three languages.
- Aggregate macro-F1 hides this: every Tamil baseline, including the best
  fine-tuned model, has 0% recall on the vulgar and abuse classes.
- Optimizer configuration (differential learning rates for task heads vs.
  encoder, gradient clipping, validation-based early stopping) is the
  largest lever for closing the gap between fine-tuning and the linear
  baseline at this data scale — see the ablation in the paper.

## Structure

| Path | Contents |
|---|---|
| `paper/` | Working notes write-up |
| `common/` | Shared data loading, metrics, val split, obfuscation utilities |
| `local/` | Baselines run locally: TF-IDF, CLIP/DINOv2 probes, zero-shot VLM/LLM (Ollama), few-shot/CoT/retrieval, modality-reliance ablation, calibration/conformal prediction |
| `colab/` | Colab-run experiments (diffusion features, late fusion, multi-task fine-tuning draft) |
| `kaggle_kernels/` | HASOC Tamil/Telugu multi-task fine-tuning kernels (naive and optimizer-stabilized versions) |
| `bhm_bengali/` | Bengali (BHM) baselines: TF-IDF, LLM zero-shot/calibrated/CoT, CNERG MuRIL, IMSyPP |
| `bhm_kaggle/` | Bengali fine-tuning kernels (XLM-R, BanglaBERT, MuRIL, IndicBERTv2) |
| `results/` | Aggregated result CSVs (`comparison.csv`, `recall_comparison.csv`, `modality_reliance.csv`, `calibration_conformal.csv`) and prediction checkpoints |

## Data

Raw datasets are not included in this repository:

- **HASOC Tamil/Telugu** — obtain from the FIRE 2026 HASOC shared task
  organizers under their data-sharing agreement.
- **BHM (Bengali Hateful Memes)** — publicly available; see the original
  dataset release.

`common/data.py` expects `Tamil_HASOC/` and `Telugu_HASOC/` at the repo root
once obtained; `bhm_bengali/` and `bhm_kaggle/` expect `train.json` /
`test.json` / `valid.json` (or the equivalent `.csv` files) in place.

## Reproducing results

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # not yet pinned; see imports per script
python3 local/baseline_tfidf.py
python3 local/baseline_clip_probe.py
python3 local/baseline_vlm_ollama.py   # requires a running Ollama instance
```

Kaggle kernels under `kaggle_kernels/` and `bhm_kaggle/` are meant to be
pushed via the Kaggle CLI (`kaggle kernels push`) against a dataset
containing `common/`, `splits/`, and the raw data.
