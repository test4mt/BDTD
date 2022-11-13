import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import subprocess

import pandas as pd
import shutil
import numpy as np

METHODS = ["bow", "ftc", "sbert"]
MODELS = ['bing', 'google', 'opus']
MUTATION_TYPE = ["gender", "negative", "plural", "tense"]


def sample(num: int):
    unmarked_path = "rq/golden_dataset"
    marked_path = "rq/golden_dataset_marked"
    if not os.path.exists(marked_path):
        os.mkdir(marked_path)
    
    for model in MODELS:
        for mut in MUTATION_TYPE:
            print(model, mut)
            path = os.path.join(unmarked_path, f"golden-dataset-{model}-{mut}.csv")
            df = pd.read_csv(path)

            positive = df[(df["unfair"] == 1)&(df["sc_id"] != -1)]
            negative = df[(df["unfair"] == 0)&(df["sc_id"] != -1)]

            half_num = num // 2
            if len(positive) < half_num:
                print("Error: ", path, "positive", f"{len(positive)}<{half_num}")
                cat_num = len(positive)
            else:
                cat_num = half_num

            if len(negative) < half_num:
                print("Error: ", path, "negative", f"{len(negative)}<{half_num}")
                continue

            sampled_df = pd.concat([positive.head(cat_num), negative.head(num - cat_num)])
            sampled_df.to_csv(os.path.join(marked_path, f"golden-dataset-{model}-{mut}.csv"), index=False)

if __name__ == "__main__":
    sample(100)