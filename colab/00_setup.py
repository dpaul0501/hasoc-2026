"""Paste this as the FIRST cell in a new Colab notebook (Runtime > Change
runtime type > T4/L4 GPU). Then paste indic_bert_multitask.py's contents as
cell 2 and late_fusion_clip_indicbert.py's contents as cell 3 (each file's
`if __name__ == "__main__":` block runs as top-level code in a notebook cell,
that's fine).

Before running: upload the hasoc/ project folder to Google Drive (e.g. drag
the whole folder -- common/, splits/, Tamil_HASOC/, Telugu_HASOC/, results/ --
into "My Drive/hasoc" via drive.google.com, or use Google Drive desktop sync).
Only splits/*.csv, common/*.py, and each language's train_data_*.csv +
images_all/ are actually read by these two baselines -- test_data_*.csv and
the zips aren't needed on Drive.
"""
from google.colab import drive
drive.mount('/content/drive')

import sys
ROOT = '/content/drive/MyDrive/hasoc'  # edit if you uploaded elsewhere
sys.path.insert(0, ROOT)

import subprocess
subprocess.run(['pip', 'install', '-q', 'transformers', 'accelerate',
                 'scikit-learn', 'pillow'], check=True)

# sanity check -- should print "tamil: 640 train / 160 dev" etc, matching
# the local fork's split exactly (same splits/*.csv file, same seed)
from common.data import load_split, LANG_CONFIG
for lang in LANG_CONFIG:
    train_rows, dev_rows = load_split(lang)
    print(f"{lang}: {len(train_rows)} train / {len(dev_rows)} dev")
