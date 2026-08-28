"""Colab GPU cell: frozen Stable Diffusion U-Net features + linear probe.

Third vision-representation baseline alongside CLIP (text-contrastive) and
DINOv2 (self-supervised discriminative). Diffusion features come from a
generative denoising objective -- literature (see search notes) finds them
generally weaker than DINOv2 alone but sometimes complementary on
spatial/compositional properties, which is exactly the kind of cue
(iconography layout, symbol placement) the original plan flagged DINOv2 for.
This baseline tests whether that holds here, not for augmentation/generation.

Method: encode each image once through SDXL's VAE to get the clean latent,
add a small fixed amount of noise (single timestep, not a full denoising
loop -- we're extracting features, not generating), run one U-Net forward
pass, and global-average-pool a mid-block feature map. This is the standard
"single-step diffusion feature extraction" recipe from the representation-
learning literature, not full inference, so it's one forward pass per image.

Needs a GPU (SDXL is ~7GB in fp16) -- run in Colab after the setup cell in
colab/00_setup.py, plus:
    !pip install -q diffusers
"""
import os
import sys

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import CLIPTextModel, CLIPTokenizer

from common.data import LANG_CONFIG, load_split
from common.metrics import TASKS, score_predictions, record_result

SD_MODEL = "stabilityai/stable-diffusion-2-1-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
NOISE_TIMESTEP = 100  # low-noise regime -- feature quality drops off at high t
IMG_SIZE = 512


def load_models():
    vae = AutoencoderKL.from_pretrained(SD_MODEL, subfolder="vae", torch_dtype=DTYPE).to(DEVICE).eval()
    unet = UNet2DConditionModel.from_pretrained(SD_MODEL, subfolder="unet", torch_dtype=DTYPE).to(DEVICE).eval()
    tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(SD_MODEL, subfolder="text_encoder", torch_dtype=DTYPE).to(DEVICE).eval()
    # empty-prompt conditioning -- we want image structure, not text-guided features
    with torch.no_grad():
        empty_ids = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length,
                               return_tensors="pt").input_ids.to(DEVICE)
        empty_embed = text_encoder(empty_ids)[0]
    scheduler = DDPMScheduler.from_pretrained(SD_MODEL, subfolder="scheduler")
    return vae, unet, empty_embed, scheduler


def preprocess(image_path):
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def extract_feature(image_path, vae, unet, empty_embed, scheduler):
    pixel_values = preprocess(image_path).to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        latent = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latent)
        t = torch.tensor([NOISE_TIMESTEP], device=DEVICE)
        # use the model's actual training-time noise schedule (alpha_cumprod-weighted
        # mix of signal and noise), not an ad hoc linear blend -- otherwise the U-Net
        # sees an out-of-distribution input and the "features" are meaningless
        noisy_latent = scheduler.add_noise(latent, noise, t)

        # hook the mid-block output as the feature map
        feats = {}
        def hook(module, inp, out):
            feats["mid"] = out
        handle = unet.mid_block.register_forward_hook(hook)
        unet(noisy_latent, t, encoder_hidden_states=empty_embed)
        handle.remove()

    feat_map = feats["mid"]  # [1, C, H, W]
    pooled = feat_map.mean(dim=[2, 3]).squeeze(0)  # global average pool -> [C]
    return pooled.float().cpu().numpy()


def embed_images(rows, vae, unet, empty_embed, scheduler):
    return np.stack([extract_feature(r["image_path"], vae, unet, empty_embed, scheduler) for r in rows])


def run_lang(lang, vae, unet, empty_embed, scheduler):
    train_rows, dev_rows = load_split(lang)
    X_train = embed_images(train_rows, vae, unet, empty_embed, scheduler)
    X_dev = embed_images(dev_rows, vae, unet, empty_embed, scheduler)

    predictions = {}
    for task in TASKS:
        y_train = [r[task] for r in train_rows]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_train, y_train)
        predictions[task] = list(clf.predict(X_dev))

    scores = score_predictions(dev_rows, predictions)
    record_result(lang, "diffusion_unet_feature_probe", scores,
                   notes=f"{SD_MODEL}, t={NOISE_TIMESTEP}, mid_block_gap")


if __name__ == "__main__":
    vae, unet, empty_embed, scheduler = load_models()
    for lang in LANG_CONFIG:
        print(f"=== {lang} ===")
        run_lang(lang, vae, unet, empty_embed, scheduler)
