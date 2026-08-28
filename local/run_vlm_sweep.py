"""Driver: run the zero-shot VLM baseline across every pulled Ollama vision
model (and, for llava, every quantization level already on disk) so the
comparison table shows both a model-family ablation and a quantization
ablation on the same dev split.

Run individual models with, e.g.:
    HASOC_VLM_MODEL=qwen2.5vl python3 local/baseline_vlm_ollama.py

This script just loops that over the full sweep list, skipping any model
that isn't actually pulled yet (checked via `ollama list`).
"""
import os
import subprocess
import sys

# llava quantization ablation (all already pulled locally)
LLAVA_QUANT_SWEEP = [
    "llava:7b-v1.6-mistral-q2_K",
    "llava:7b-v1.6-mistral-q3_K_M",
    "llava:7b-v1.6-mistral-q4_0",
    "llava:7b-v1.6-mistral-q4_K_M",
]

# cross-model-family sweep (default tag per model, open-source only)
MODEL_FAMILY_SWEEP = [
    "moondream",
    "minicpm-v",
    "qwen2.5vl",
    "llama3.2-vision",
    "qwen3-vl",
]

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_vlm_ollama.py")


def get_pulled_models():
    out = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True).stdout
    return {line.split()[0] for line in out.strip().splitlines()[1:]}


def main():
    pulled = get_pulled_models()
    sweep = LLAVA_QUANT_SWEEP + MODEL_FAMILY_SWEEP
    for model in sweep:
        # ollama list shows tags like "moondream:latest" for bare pulls
        candidates = {model, f"{model}:latest"}
        if not (candidates & pulled):
            print(f"SKIP {model}: not pulled yet")
            continue
        actual_tag = next(iter(candidates & pulled))
        print(f"=== running {actual_tag} ===")
        env = dict(os.environ, HASOC_VLM_MODEL=actual_tag)
        subprocess.run([sys.executable, SCRIPT], env=env, check=False)


if __name__ == "__main__":
    main()
