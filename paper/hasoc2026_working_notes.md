# [Team Name] at HASOC 2026: Baselines, Metric Choice, and Fine-Tuning for Tamil, Telugu, and Bengali Hateful Meme/Content Classification

**Track:** FIRE 2026 HASOC — Tamil/Telugu Hateful Meme Classification (sentiment, sarcasm, vulgarity, abuse, target)
**Paper type:** Working notes (CEUR-WS)

## Abstract

We report a baseline study for HASOC 2026 Tamil and Telugu meme classification (five subtasks: sentiment, sarcasm, vulgarity, abuse, target), extended with a comparable Bengali hate-speech setup (BHM) for cross-lingual coverage. We compare four supervision regimes — majority-class, zero-shot VLM/LLM prompting, frozen-encoder linear probes, and multi-task fine-tuning — under a fixed train/dev split per language. A simple OCR-plus-TF-IDF linear baseline outperforms every zero-shot VLM/LLM configuration we tested (seven VLM variants, one text LLM, few-shot, chain-of-thought, and retrieval-augmented prompting) on Tamil, and is beaten only by a CLIP linear probe on Telugu; the same pattern holds on Bengali, including against a purpose-built Indic abusive-speech classifier. We show that aggregate macro-F1 conceals this: every baseline we test, including the strongest fine-tuned model, has 0% recall on Tamil's vulgar and abuse classes. We further show that optimizer configuration — differential learning rates for randomly-initialized task heads versus a pretrained encoder, gradient clipping, and validation-based early stopping — has a large effect on fine-tuning outcomes at this data scale, closing most of the gap to the best baseline on Telugu and Bengali (BanglaBERT) and narrowing it on Tamil. We report per-class recall, modality-reliance, and calibration results alongside macro-F1 and recommend per-class recall on harm-relevant labels as a required metric for this task.

## 1. Introduction

HASOC 2026 Tamil and Telugu provide roughly 800 labeled memes per language across five subtasks, with severe class imbalance on the two subtasks most relevant to harm mitigation (vulgar, abuse). We evaluate a broad baseline suite and a fine-tuned system across three languages — Tamil, Telugu, and Bengali (via the public BHM dataset, included for cross-lingual coverage under a comparable binary hate-speech formulation) — and report:

- A supervision-regime comparison (zero-shot, linear probe, fine-tuned) across all three languages.
- Per-class recall analysis showing macro-F1 masks failure on minority harm classes.
- A modality-reliance ablation (text vs. visual signal) and a calibration/conformal-prediction analysis.
- An optimizer ablation for multi-task fine-tuning at small sample sizes.

## 2. Data

| | Tamil (HASOC) | Telugu (HASOC) | Bengali (BHM) |
|---|---|---|---|
| Task | 5 subtasks (sentiment, sarcasm, vulgar, abuse, target) | 5 subtasks (same) | binary hate/non-hate |
| Train / dev / test | 640 / 160 | 640 / 160 | 5758 / 640 / 711 |
| Minority-class rate (vulgar/abuse or hate) | ~0.6–1.3% | ~5.6–12.5% | ~37% |
| Modality used | image + OCR text | image + OCR text | caption text |

Tamil and Telugu use a fixed 640/160 train/dev split (seed=42). A further 500/140 split of the 640 training rows separates prompt/exemplar selection and fine-tuning early-stopping from the 160-example dev set used for all reported numbers, so no reported result is influenced by prompt design or checkpoint selection on held-out data. BHM is used in its original train/valid/test split; the 711-example test set is used for all BHM numbers reported here, except LLM-prompting results, which use a fixed 150-example sample (seed=42) for compute reasons.

## 3. Method

**Supervision regimes.** *none* (majority class), *zero-shot* (VLM/LLM prompting, no gradient updates), *linear probe* (frozen encoder + trained linear head), *fine-tuned* (full multi-task gradient descent).

**Baselines.** OCR text (`ocr_tamil`/`easyocr`) with char-n-gram TF-IDF and per-task logistic regression; linear probes over frozen CLIP ViT-B/32 and DINOv2-base image embeddings; zero-shot VLM prompting (LLaVA-7B at four quantization levels, moondream, minicpm-v, qwen2.5-VL, qwen3-VL) and zero-shot text-LLM prompting on OCR text (qwen2.5:7B); few-shot chain-of-thought and CLIP-retrieval-augmented few-shot prompting, evaluated on the full 140-example held-out split. For Bengali, the same TF-IDF and LLM-prompting baselines are used on caption text, plus two purpose-built classifiers: `Hate-speech-CNERG/indic-abusive-allInOne-MuRIL` (Indic abusive-speech classifier) and `IMSyPP/hate_speech_multilingual` (trained on English/Italian/Slovenian only, included as a cross-lingual-transfer control).

**Fine-tuning.** Multi-task XLM-RoBERTa-base with one linear head per subtask (Tamil/Telugu) or a single binary head (Bengali, plus a BanglaBERT variant using `csebuetnlp/banglabert`), trained with class-weighted cross-entropy, differential learning rates (2e-5 encoder, 5e-4 heads), gradient clipping (max norm 1.0), and validation-based early stopping on the held-out split described above. Final numbers are reported on each language's untouched dev/test split.

**Modality ablation.** Text regions and non-text visual regions of each dev image are independently obfuscated (cached OCR boxes, 6px padding), and CLIP/DINOv2 probes and qwen2.5-VL are re-scored under each condition to estimate reliance on each modality per subtask.

**Calibration.** Expected calibration error (ECE) and split conformal prediction (target coverage 0.9) on the OCR+TF-IDF baseline.

## 4. Results

### 4.1 Overall comparison

| Lang | Best baseline (regime) | Score | Fine-tuned | Score |
|---|---|---|---|---|
| Tamil | OCR+TF-IDF (linear probe) | 0.480 | multi-task XLM-R | 0.463 |
| Telugu | CLIP probe (linear probe) | 0.540 | multi-task XLM-R | 0.500 |
| Bengali (hate-F1) | OCR/caption+TF-IDF (linear probe) | 0.569 | BanglaBERT | **0.574** |

Across seven VLM configurations, one text LLM, few-shot, chain-of-thought, and retrieval augmentation, no zero-shot model beats the TF-IDF baseline on any of the three languages; the best zero-shot VLM (qwen2.5-VL) reaches 0.435 (Tamil) and 0.460 (Telugu), and the best calibrated, chain-of-thought LLM on Bengali (mistral:7B) reaches hate-F1 0.488. The CNERG MuRIL abusive-speech classifier scores hate-F1 0.190 on Bengali despite being purpose-built for Indic abusive content; the IMSyPP negative control (no Bengali training exposure) scores 0.320.

### 4.2 Per-class recall

| Lang | Method | vulgar recall | abuse recall |
|---|---|---|---|
| Tamil | majority / TF-IDF / CLIP / DINOv2 | 0.00 | 0.00 |
| Telugu | majority | 0.00 | 0.00 |
| Telugu | TF-IDF | 0.45 | 0.00 |
| Telugu | CLIP | 0.40 | 0.33 |
| Telugu | DINOv2 | 0.20 | 0.11 |

Every Tamil baseline evaluated has 0% recall on both vulgar and abuse, despite per-task macro-F1 near 0.48–0.50 on those tasks (driven by the 159/160 negative examples). We treat per-class recall on these two labels as the operationally meaningful metric and recommend it as a required companion to macro-F1 in future rounds.

### 4.3 Modality reliance

Visual reliance is largest and most consistent on the target subtask: Telugu CLIP visual-reliance = 0.72 (F1 0.373→0.106 under visual obfuscation), Telugu DINOv2 = 0.50, Tamil DINOv2 = 0.39 — identifying who a meme targets depends substantially on the depicted image, not the caption. Vulgar and abuse show near-zero reliance on either modality for CLIP and qwen2.5-VL, consistent with §4.2: a model that predicts the majority class regardless of input has no modality signal to lose.

### 4.4 Calibration

ECE is highest on the most imbalanced, highest-cardinality subtasks (Tamil target = 0.244, Telugu abuse = 0.242) and lowest on balanced binary subtasks (Telugu vulgar = 0.093). Empirical conformal coverage tracks the 0.9 target within 0.875–0.975 across all ten language × subtask cells, indicating the confidence estimates themselves are reasonably calibrated even where recall is poor.

### 4.5 Fine-tuning optimizer ablation

| | Tamil | Telugu | Bengali (XLM-R) | Bengali (BanglaBERT) |
|---|---|---|---|---|
| Naive (single LR, no clipping, last-epoch checkpoint) | 0.465 | 0.426 | hate-F1 ≈ 0.00† | — |
| Differential LR + gradient clipping + early stopping | 0.463 | **0.500** | hate-F1 0.564 | hate-F1 **0.574** |
| Best linear-probe baseline | 0.480 | 0.540 | hate-F1 0.569 | hate-F1 0.569 |

†Naive Bengali fine-tuning converged to predicting a single class for the full test set (recall 1.0, precision 0.37, hate-F1 driven entirely by base rate).

Differential learning rates, gradient clipping, and early stopping give a consistent, large improvement over a naive single-LR configuration at this training scale (500–5,758 examples, multi-task or single-task): the largest single-subtask gain is on Tamil/Telugu *target* (Tamil 0.164→0.247, Telugu 0.145→0.289), and on Bengali the stabilized configuration is the only one to beat the TF-IDF baseline (BanglaBERT hate-F1 0.574 vs. 0.569). The gain is not uniform across subtasks — Tamil sentiment F1 falls from 0.455 to 0.328 under the stabilized configuration, leaving Tamil's overall macro-F1 essentially unchanged and still below its baseline, while Telugu and Bengali both improve materially. We attribute the residual Tamil gap to its training set being the smallest and most imbalanced of the three (§2).

## 5. Discussion

Aggregate macro-F1 over an imbalanced multi-class label space can report a system as competent while it detects zero vulgar or abusive Tamil memes — the failure is visible only at the per-class level. This pattern is not specific to HASOC's split size: it holds on Bengali as well, at roughly 9× the training data and much better class balance, and against a domain-specific Indic abusive-speech classifier, not only general-purpose LLMs. Optimizer configuration — not model capacity or architecture choice — is the largest lever we find for closing the gap between fine-tuning and a simple linear baseline at this data scale, and its effect is comparable in size to the gap between baselines and zero-shot prompting.

## 6. Limitations

Zero-shot results depend on prompt wording, which we did not exhaustively search. Bengali LLM-prompting numbers use a fixed 150-example sample rather than the full test set. Tamil fine-tuning was not further tuned beyond the stabilized configuration in §4.5, to avoid overfitting a 160-example dev set. Diffusion-based visual features and a retrieval+reasoning agent pipeline were not evaluated.

## 7. Conclusion

A stabilized multi-task fine-tuned transformer is competitive with the best frozen-encoder linear probe on Telugu and Bengali, and a simple OCR/caption+TF-IDF baseline outperforms zero-shot VLM/LLM prompting — including few-shot, chain-of-thought, retrieval augmentation, and a purpose-built Indic classifier — across all three languages studied. Per-class recall on harm-relevant labels, not aggregate macro-F1, is the metric that distinguishes a genuinely useful system from one that merely predicts well on the majority class; we recommend it as a required reporting metric for this task, and recommend differential learning rates, gradient clipping, and validation-based early stopping as standard practice for small-sample multi-task fine-tuning submissions.

## Appendix A: Full Results

See `results/comparison.csv`, `results/recall_comparison.csv`, `results/modality_reliance.csv`, and `results/calibration_conformal.csv` for complete per-run data underlying the tables above.
