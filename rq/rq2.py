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

BEST_WINS = {
    "bow": 0,
    "ftc": 0,
    "sbert": 49,
}

def rq2():
    """RQ2: How effective is our method in finding unfair issues
    """
    sense_inventory_path = '../asset/sense_inventory/sense_dict.json'

    out_df = {
        "method": [],
        "model": [],
        "mut": [],
        "positive": [],
        "true_positive": [],
    }
    for model in MODELS:
        for mt in MUTATION_TYPE:
            for method in METHODS:
                alignment_window_size = BEST_WINS[method]
                golden_set_result_path = f'rq/rq2/golden-dataset-{model}-{mt}_{method}_{alignment_window_size}.csv'
                df = pd.read_csv(golden_set_result_path)
                # df['unfair'] = np.where((df['ori.sc_id'] != df['mut.sc_id']) & (df['ori.sc_id'] >= 0) & (df['mut.sc_id'] >= 0), 1, 0)
                # df['inconsistency'] = np.where(df['ori.sc_id'] == df['mut.sc_id'], 0, 1)
 
                true_positive = len(df[(df['ori.sc_id'] >= 0) & (df['mut.sc_id'] >= 0)
                                    & (df['ori.sc_id'] != df['mut.sc_id']) & (df['unfair'] > 0)])
                positive = len(df[(df['ori.sc_id'] >= 0) & (df['mut.sc_id'] >= 0) & (df['ori.sc_id'] != df['mut.sc_id'])])

                out_df["method"].append(method)
                out_df["model"].append(model)
                out_df["mut"].append(mt)
                out_df["positive"].append(positive)
                out_df["true_positive"].append(true_positive)
    rq2_result = pd.DataFrame(out_df)
    rq2_result.to_csv('rq/rq2/rq2.csv', index=False)

    print(rq2_result)

if __name__ == "__main__":
    rq2()
