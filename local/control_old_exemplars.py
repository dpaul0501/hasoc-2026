"""Control experiment: same prompt_val set, same retrieval mechanism, but
TARGET-HOMOGENEOUS exemplars (all target=individual, replicating the actual
bug being tested) instead of the diversified set -- isolates whether the
diversity fix helps.

NOT the literal original v1 exemplar IDs: 2 of those (image_tamil_0543.jpg,
image_telugu_0524.png) turned out to be in the DEV split, not train --
meaning the original pilots had those exact images serving as BOTH a fixed
few-shot demonstration AND a scored eval example simultaneously (confirmed:
both appeared in pilot_subset(dev_rows, 35) for their language). That's real
data leakage, not a hypothetical -- it inflated the earlier "few-shot fixed
the 0%-recall problem" results (Tamil dev has only 2 abusive examples total;
one of them was leaked). Discarding those IDs and picking valid,
leakage-free train_core exemplars with the same target=individual property
instead, so this control isolates ONLY the diversity variable.

Uses run_id="val_control_oldexemplars" (separate from the diversified run's
plain "val") so it gets its own checkpoint file and result row instead of
silently reusing -- and being contaminated by -- the diversified run's cache.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import local.baseline_vlm_retrieval_cot as m

# all target=individual (the property being tested), all confirmed members
# of train_core (never dev, never prompt_val) -- see docstring above for why
# the literal v1 IDs couldn't be reused
m.FIXED_EXEMPLAR_IDS = {
    "tamil": ["image_tamil_0590.jpg", "image_tamil_0907.jpg", "image_tamil_0356.jpg"],
    "telugu": ["image_telugu_0172.png", "image_telugu_0650.png", "image_telugu_0476.png"],
}

if __name__ == "__main__":
    from transformers import CLIPModel, CLIPProcessor
    from local.retrieval_index import DEVICE

    only = sys.argv[1] if len(sys.argv) > 1 else None
    pilot_size = int(os.environ.get("HASOC_PILOT_SIZE", "140"))
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    for lang in m.LANG_CONFIG:
        if only and lang != only:
            continue
        m.run_lang(lang, clip_model, clip_proc, pilot_size=pilot_size,
                   eval_mode="val", run_id="val_control_oldexemplars")
