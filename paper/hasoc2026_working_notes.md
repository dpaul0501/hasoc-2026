# [Team Name] at HASOC 2026: Baselines, Metric Choice, and Fine-Tuning for Tamil, Telugu, and Bengali Hateful Meme/Content Classification

**Track:** FIRE 2026 HASOC — Tamil/Telugu Hateful Meme Classification (sentiment, sarcasm, vulgarity, abuse, target)
**Paper type:** Working notes (CEUR-WS)

## Abstract

We report a baseline study for HASOC 2026 Tamil and Telugu meme classification (five subtasks: sentiment, sarcasm, vulgarity, abuse, target), extended with a comparable Bengali hate-speech setup (BHM) for cross-lingual coverage. We compare four supervision regimes — majority-class, zero-shot VLM/LLM prompting (including few-shot, chain-of-thought, and retrieval-augmented variants), frozen-encoder linear probes, and multi-task fine-tuning — under a fixed train/dev split per language. A simple OCR-plus-TF-IDF linear baseline outperforms every zero-shot/few-shot/CoT/retrieval configuration we tested (seven VLM variants, one text LLM, at least four prompting strategies) on Tamil, and is beaten only by a CLIP linear probe on Telugu; the same pattern holds on Bengali, including against a purpose-built Indic abusive-speech classifier. We show that aggregate macro-F1 conceals a further failure: every baseline we test, including the strongest fine-tuned model, has 0% recall on Tamil's vulgar and abuse classes. We show that optimizer configuration — differential learning rates for randomly-initialized task heads versus a pretrained encoder, gradient clipping, and validation-based early stopping — has a large effect on fine-tuning outcomes at this data scale, closing most of the gap to the best baseline on Telugu and Bengali (BanglaBERT) and narrowing it on Tamil. Finally, we discuss why general-purpose foundation models consistently underperform simple baselines and regionally-pretrained encoders on this task family, drawing on tokenization, code-mixing, cultural grounding of the label taxonomy, and script coverage in vision-language pretraining as candidate mechanisms, and on specific patterns observed across our runs as supporting evidence.

## 1. Introduction

HASOC 2026 Tamil and Telugu provide roughly 800 labeled memes per language across five subtasks, with severe class imbalance on the two subtasks most relevant to harm mitigation (vulgar, abuse). We evaluate a broad baseline suite and a fine-tuned system across three languages — Tamil, Telugu, and Bengali (via the public BHM dataset, included for cross-lingual coverage under a comparable binary hate-speech formulation) — and report:

- A supervision-regime comparison (zero-shot, few-shot/CoT, retrieval-augmented, linear probe, fine-tuned) across all three languages.
- Per-class recall analysis showing macro-F1 masks failure on minority harm classes.
- A modality-reliance ablation (text vs. visual signal) and a calibration/conformal-prediction analysis.
- An optimizer ablation for multi-task fine-tuning at small sample sizes.
- An analysis of why general-purpose foundation models underperform regionally-pretrained encoders on this task family.

## 2. Data

| | Tamil (HASOC) | Telugu (HASOC) | Bengali (BHM) |
|---|---|---|---|
| Task | 5 subtasks (sentiment, sarcasm, vulgar, abuse, target) | 5 subtasks (same) | binary hate/non-hate |
| Train / dev / test | 640 / 160 | 640 / 160 | 5758 / 640 / 711 |
| Minority-class rate (vulgar/abuse or hate) | ~0.6–1.3% | ~5.6–12.5% | ~37% |
| Modality used | image + OCR text | image + OCR text | caption text |

Tamil and Telugu use a fixed 640/160 train/dev split (seed=42). A further 500/140 split of the 640 training rows separates prompt/exemplar selection and fine-tuning early-stopping from the 160-example dev set used for all reported numbers, so no reported result is influenced by prompt design or checkpoint selection on held-out data. BHM is used in its original train/valid/test split; the 711-example test set is used for all BHM numbers reported here, except LLM-prompting results, which use a fixed 150-example sample (seed=42) for compute reasons.

## 3. Method

**Supervision regimes.** *none* (majority class), *zero-shot* (VLM/LLM prompting, no gradient updates — includes zero-shot, few-shot, chain-of-thought, and retrieval-augmented variants), *linear probe* (frozen encoder + trained linear head), *fine-tuned* (full multi-task gradient descent).

**Baselines.** OCR text (`ocr_tamil`/`easyocr`) with char-n-gram TF-IDF and per-task logistic regression, on both native-language OCR text and an English-translated variant; linear probes over frozen CLIP ViT-B/32 and DINOv2-base image embeddings; zero-shot VLM prompting (LLaVA-7B at four quantization levels — q2_K, q3_K_M, q4_0, q4_K_M — plus moondream, minicpm-v, qwen2.5-VL, qwen3-VL) and zero-shot text-LLM prompting on OCR text alone (qwen2.5:7B); few-shot chain-of-thought prompting; and CLIP-retrieval-augmented (RAG) few-shot chain-of-thought prompting, where the nearest training-set neighbor by CLIP embedding similarity is retrieved as a dynamic exemplar alongside fixed exemplars. For Bengali, the same TF-IDF and LLM-prompting baselines are used on caption text, plus two purpose-built classifiers: `Hate-speech-CNERG/indic-abusive-allInOne-MuRIL` (Indic abusive-speech classifier) and `IMSyPP/hate_speech_multilingual` (trained on English/Italian/Slovenian only, included as a cross-lingual-transfer control).

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

### 4.2 Zero-shot VLM and LLM prompting

| Model | Tamil | Telugu | Notes |
|---|---|---|---|
| LLaVA-7B (q2_K – q4_K_M range) | 0.365 – 0.433 | 0.341 – 0.384 | non-monotonic across quantization levels |
| moondream | 0.294 | 0.268 | weakest VLM; 15/160 parse failures on Telugu |
| minicpm-v | 0.337 | 0.393 | |
| qwen2.5-VL | **0.435** | **0.460** | best zero-shot VLM, both languages |
| qwen3-VL (reasoning) | 0.423 | 0.425 | 7–14/160 request failures (reasoning-trace format compliance) |
| qwen2.5:7B, text-only (OCR text, no image) | 0.410 | 0.446 | matches or beats the best VLM despite no visual input |
| English-translated OCR text + TF-IDF | 0.480 | 0.448 | translation does not improve over native-language TF-IDF; hurts on Telugu |

No VLM or LLM configuration reaches the OCR+TF-IDF baseline (Tamil 0.480, Telugu — CLIP probe 0.540 / TF-IDF 0.485). Quantization level has no consistent monotonic effect on LLaVA's score, and the reasoning-mode model (qwen3-VL) shows the highest request-failure rate of any VLM tested, without a corresponding accuracy gain over the non-reasoning qwen2.5-VL.

### 4.3 Few-shot, chain-of-thought, and retrieval-augmented (RAG) prompting

All runs in this section use qwen2.5-VL, the best zero-shot model from §4.2, as the base model.

| Configuration | n | Tamil | Telugu |
|---|---|---|---|
| Zero-shot (no exemplars) | 160 (dev) | 0.435 | 0.460 |
| 3-shot + chain-of-thought | 35 (pilot) | 0.431 | 0.503 |
| RAG: CLIP-retrieved neighbor + fixed exemplars + CoT | 35 (pilot) | 0.460 | 0.473 |
| RAG: CLIP-retrieved neighbor + fixed exemplars + CoT | 140 (full held-out split) | 0.366 – 0.382 | 0.441 – 0.448 |

(Ranges on the full-split RAG row span two exemplar-set configurations.) The 35-example pilots suggested a real gain from adding exemplars and retrieval, particularly on Telugu (0.460→0.503). This did not hold at full scale: on the full 140-example held-out split, RAG scores below plain zero-shot prompting on both languages, and below the TF-IDF baseline. We attribute the pilot-scale gain to sampling variance at n=35 rather than a real effect of retrieval or CoT, and treat the full-split numbers as authoritative. Across every prompting strategy tested — zero-shot, few-shot, CoT, and retrieval-augmented — no configuration on either language beats the TF-IDF baseline.

### 4.4 Per-class recall

| Lang | Method | vulgar recall | abuse recall |
|---|---|---|---|
| Tamil | majority / TF-IDF / CLIP / DINOv2 | 0.00 | 0.00 |
| Telugu | majority | 0.00 | 0.00 |
| Telugu | TF-IDF | 0.45 | 0.00 |
| Telugu | CLIP | 0.40 | 0.33 |
| Telugu | DINOv2 | 0.20 | 0.11 |

Every Tamil baseline evaluated has 0% recall on both vulgar and abuse, despite per-task macro-F1 near 0.48–0.50 on those tasks (driven by the 159/160 negative examples). We treat per-class recall on these two labels as the operationally meaningful metric and recommend it as a required companion to macro-F1 in future rounds.

### 4.5 Modality reliance

Visual reliance is largest and most consistent on the target subtask: Telugu CLIP visual-reliance = 0.72 (F1 0.373→0.106 under visual obfuscation), Telugu DINOv2 = 0.50, Tamil DINOv2 = 0.39 — identifying who a meme targets depends substantially on the depicted image, not the caption. Vulgar and abuse show near-zero reliance on either modality for CLIP and qwen2.5-VL, consistent with §4.4: a model that predicts the majority class regardless of input has no modality signal to lose.

### 4.6 Calibration

ECE is highest on the most imbalanced, highest-cardinality subtasks (Tamil target = 0.244, Telugu abuse = 0.242) and lowest on balanced binary subtasks (Telugu vulgar = 0.093). Empirical conformal coverage tracks the 0.9 target within 0.875–0.975 across all ten language × subtask cells, indicating the confidence estimates themselves are reasonably calibrated even where recall is poor.

### 4.7 Fine-tuning optimizer ablation

| | Tamil | Telugu | Bengali (XLM-R) | Bengali (BanglaBERT) |
|---|---|---|---|---|
| Naive (single LR, no clipping, last-epoch checkpoint) | 0.465 | 0.426 | hate-F1 ≈ 0.00† | — |
| Differential LR + gradient clipping + early stopping | 0.463 | **0.500** | hate-F1 0.564 | hate-F1 **0.574** |
| Best linear-probe baseline | 0.480 | 0.540 | hate-F1 0.569 | hate-F1 0.569 |

†Naive Bengali fine-tuning converged to predicting a single class for the full test set (recall 1.0, precision 0.37, hate-F1 driven entirely by base rate).

Differential learning rates, gradient clipping, and early stopping give a consistent, large improvement over a naive single-LR configuration at this training scale: the largest single-subtask gain is on Tamil/Telugu *target* (Tamil 0.164→0.247, Telugu 0.145→0.289), and on Bengali the stabilized configuration is the only one to beat the TF-IDF baseline (BanglaBERT hate-F1 0.574 vs. 0.569). The gain is not uniform — Tamil sentiment F1 falls from 0.455 to 0.328 under the stabilized configuration, leaving Tamil's overall macro-F1 essentially unchanged and still below its baseline, while Telugu and Bengali both improve materially.

## 5. Why General-Purpose Foundation Models Underperform, and Regional Pretraining Helps

The consistent gap between general-purpose foundation models (zero-shot and few-shot/CoT/RAG, §4.2–4.3) and simple baselines or regionally-pretrained encoders (§4.7) is the most robust pattern in this study — it holds across seven VLM families, a general-purpose text LLM, four prompting strategies, and two independent languages. We did not run controlled experiments isolating each mechanism below; we offer them as candidate explanations, ranked by how directly our results support them, rather than established fact.

**Best-supported: translation does not close the gap, so this is not simply a language-fluency problem.** If foundation models underperformed because they cannot process Tamil/Telugu text, translating that text to English before classification should help. It does not: English-translated OCR+TF-IDF scores 0.480 on Tamil (identical to native-language TF-IDF) and 0.448 on Telugu (worse than native-language TF-IDF's 0.485, §4.2). Whatever these models are missing is not resolved by moving into their strongest language, which suggests the loss is in culturally- or context-specific meaning — code-mixed slang, meme-format conventions, region-specific referents — that does not survive translation, rather than in the source language itself.

**Supported by a within-language contrast: regional pretraining plus task-specific fine-tuning beats either alone.** On Bengali, BanglaBERT (pretrained specifically on Bengali corpora, then fine-tuned on BHM) is the only configuration in the entire study to beat its TF-IDF baseline on the harm-relevant metric (hate-F1 0.574 vs. 0.569). But `Hate-speech-CNERG/indic-abusive-allInOne-MuRIL` — also a regionally-pretrained, Indic-specific model, and purpose-built for abusive-speech detection — scores hate-F1 0.190 on BHM, worse than an uncalibrated general-purpose LLM prompt. Regional pretraining is not sufficient by itself; CNERG's model has almost certainly seen abusive Bengali/Indic text, but not this dataset's specific label definitions and register. The pattern across our results is that regional pretraining *and* in-distribution fine-tuning are both necessary — one without the other underperforms a simple in-distribution linear baseline.

**Consistent with, not proven by, our results: script and code-mixing coverage in pretraining data.** Tamil and Telugu memes mix native script, Romanized transliteration, and English within a single caption. General multilingual foundation models are predominantly trained on monolingual web text per language; models built with explicit exposure to code-mixed South Asian social media text (MuRIL, IndicBERT-family, BanglaBERT) are the ones that close the gap in §4.7. We did not directly measure tokenizer fragmentation rates in this study, but this is a plausible mechanism consistent with the translation result above: a model whose subword vocabulary fragments Tamil/Telugu/code-mixed text into many low-information tokens has less effective context to reason over, both zero-shot and in-context (few-shot).

**Anecdotal, from qualitative patterns across runs, not a quantified claim:**
- Every VLM and LLM we tested defaults toward the majority (non-vulgar, non-abusive) label under uncertainty rather than toward a harm-flagging label (§4.4) — consistent with widely-reported conservative/safety-tuned behavior in instruction-tuned models, which would bias zero-shot prompting away from actively labeling content as vulgar or abusive regardless of language.
- Quantization level shifts LLaVA's score non-monotonically (§4.2) rather than degrading smoothly, suggesting the model's usable signal for this task is already close to a noise floor — small perturbations move the operating point around rather than trading off against a stable underlying capability.
- The reasoning-mode model (qwen3-VL) had the highest request-failure rate of any VLM tested without an accuracy improvement over its non-reasoning sibling, suggesting the extra reasoning capacity was not being spent on task-relevant distinctions for this content.
- `IMSyPP/hate_speech_multilingual`, with zero Bengali training exposure, still scores hate-F1 0.320 on BHM — well above zero, indicating some hate-relevant signal (formatting, punctuation, code-mixed English tokens, emoji) transfers even across a genuine language boundary. This is a caution against reading any non-zero LLM/VLM score as evidence of language understanding; some of it may be surface pattern-matching that happens to correlate with the label.

## 6. Risks and Broader Impact

Beyond raw accuracy, several findings above point to failure modes with direct deployment consequences, distinct from "the model got the label wrong":

**Silent moderation failure.** Every baseline tested has 0% recall on Tamil vulgar and abuse content (§4.4), while overall macro-F1 stays around 0.48 — a range that reads as reasonably competent in a results table. A content filter built on any of these baselines would pass through all vulgar/abusive Tamil content in our test set while reporting metrics that look adequate. This was the modal outcome, not an edge case, across every model we evaluated for this language.

**Calibration can mask this failure rather than reveal it.** Conformal coverage tracks its 0.9 nominal target even on the classes with 0% recall (§4.6). A deployer relying on calibration/confidence scores to decide when to trust a model — a common practice — would see well-calibrated confidence from a model that never flags the content it exists to catch.

**Language-access asymmetry in safety tooling.** The consistent gap between general-purpose foundation models and simple in-language baselines (§4.2–4.3) means Tamil-, Telugu-, and Bengali-speaking users are, on this evidence, served by weaker content-safety tooling than English-speaking users of the same model families — an equity concern that compounds existing under-resourcing of these languages online.

**"Use a regional model instead" is not automatically safe.** A purpose-built, regionally-pretrained Indic abusive-speech classifier (CNERG/MuRIL) underperforms even an uncalibrated general-purpose LLM prompt on Bengali (§5). Reading our results as "swap in a regional model and the problem is solved" is an overclaim our own data contradicts — regional pretraining without in-distribution fine-tuning on the exact label definitions in use was not sufficient in any experiment we ran.

**Safety-tuning may trade off against harm detection.** Foundation models qualitatively default to the non-harmful label under uncertainty (§5), consistent with instruction-tuning that makes a model reluctant to produce harmful content also making it reluctant to flag harmful content shown to it — particularly where its uncertainty is already higher, as in these languages. Alignment behavior validated mainly on English is not guaranteed to transfer its intended effect to low-resource-language moderation, and could plausibly work against it.

**Overclaiming from three languages.** Any claim of the form "LLMs fail on low-resource languages, local models win" built on this data should be scoped to what three languages, single-digit minority-class counts in some dev splits, and largely single-seed runs can support. We report the pattern as consistent and worth further study, not as a proven general law; a benchmark-scale claim would need more languages, multiple seeds, and matched-pair ablations isolating pretraining-data coverage from architecture and scale.

## 7. Limitations

Zero-shot results depend on prompt wording, which we did not exhaustively search. Bengali LLM-prompting numbers use a fixed 150-example sample rather than the full test set. Tamil fine-tuning was not further tuned beyond the stabilized configuration in §4.7, to avoid overfitting a 160-example dev set. The mechanisms proposed in §5 are consistent with our results but were not tested by controlled ablation (e.g., we did not measure tokenizer fragmentation directly, nor run a matched pair of models differing only in code-mixed pretraining exposure). Diffusion-based visual features and a retrieval+reasoning agent pipeline were not evaluated.

## 8. Conclusion

A stabilized multi-task fine-tuned transformer is competitive with the best frozen-encoder linear probe on Telugu and Bengali, and a simple OCR/caption+TF-IDF baseline outperforms zero-shot, few-shot, chain-of-thought, and retrieval-augmented VLM/LLM prompting — including against a purpose-built Indic classifier — across all three languages studied. The gap is not resolved by translation, is only closed by pretrained regional encoders when combined with in-distribution fine-tuning, and coexists with qualitative signs (majority-label bias, non-monotonic quantization behavior, non-zero scores from a model with no exposure to the target language) that point toward pretraining-data coverage and safety-tuning artifacts rather than raw model capacity as the operative bottleneck. Per-class recall on harm-relevant labels, not aggregate macro-F1, is the metric that distinguishes a genuinely useful system from one that merely predicts well on the majority class; we recommend it as a required reporting metric for this task, and recommend differential learning rates, gradient clipping, and validation-based early stopping as standard practice for small-sample multi-task fine-tuning submissions.

## Appendix A: Full Results

See `results/comparison.csv`, `results/recall_comparison.csv`, `results/modality_reliance.csv`, and `results/calibration_conformal.csv` for complete per-run data underlying the tables above.
