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

def sample_golden_dataset(model: str, mut_type: str, number: int=800):
    print(model, mut_type)

    output_path = f'rq/golden_dataset/golden-dataset-{model}-{mut_type}.csv'

    df = pd.read_csv(f'result/{model}/test-set-{mut_type}_{model}_merged_sbert_49.csv')
    df = df[(df['mut.tgt_sense'] != "[]") & (df['ori.tgt_sense'] != "[]")]
    positive = df[((df['ori.sc_id'] >= 0) & (df['mut.sc_id'] >= 0)) & (df['ori.sc_id'] != df['mut.sc_id'])]
    negative = df[(df['ori.sc_id'] >= 0) & (df['mut.sc_id'] >= 0)] # & (df['ori.sc_id'] == df['mut.sc_id'])

    positive_samples = positive.sample(min(number//2, len(positive)), random_state=0)
    negative_samples = negative.sample(number//2, random_state=0)

    positive_samples = positive_samples.copy()
    negative_samples = negative_samples.copy()

    positive_samples["unfair"] = 1
    negative_samples["unfair"] = 0

    new_df = pd.concat([positive_samples, negative_samples])
    new_df["ori.right"] = 1
    new_df["mut.right"] = 1
    new_df["sc_id"] = -1


    new_df.to_csv(output_path, index=False)
    print('Sample {} sentences from {}, {}'.format(number, model, mut_type))
    print('Saved to {}'.format(output_path))

if __name__ == "__main__":
    for model in MODELS:
        for mut in MUTATION_TYPE:
            sample_golden_dataset(model=model, mut_type=mut)