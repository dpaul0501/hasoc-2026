import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import load_split, LANG_CONFIG
from common.metrics import majority_predict, score_predictions, record_result

if __name__ == "__main__":
    for lang in LANG_CONFIG:
        train_rows, dev_rows = load_split(lang)
        predictions = majority_predict(train_rows, dev_rows)
        scores = score_predictions(dev_rows, predictions)
        record_result(lang, "majority_class", scores, notes="floor baseline", supervision="none")
