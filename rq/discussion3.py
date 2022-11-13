'''
不同翻译软件同义词错误的分布
'''

from html import entities
import sys
import os
from collections import Counter
from typing import List

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import pandas as pd
import spacy
import tqdm
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns
import json

with open("asset/sense_inventory/sense_dict.json", "r", encoding="utf-8") as f:
    sense_dict = json.loads(f.read())

METHODS = ["bow", "ftc", "sbert"]
MODELS = ['bing', 'google', 'opus']
MUTATION_TYPE = ["gender", "negative", "plural", "tense"]

def auto_unfair_mark(df):
    df['unfair'] = np.where((df['ori.sc_id'] != df['mut.sc_id']) & (df['ori.sc_id'] >= 0) & (df['mut.sc_id'] >= 0), 1, 0)
    df['inconsistency'] = np.where(df['ori.sc_id'] == df['mut.sc_id'], 0, 1)
    return df

def data_mining(mutation_type: str, model: str, draw_df):
    print(mutation_type, model)
    
    golden_set_path = f'result/{model}/test-set-{mutation_type}_{model}_merged_bow_0.csv'
    golden = pd.read_csv(golden_set_path)

    # unfair_and_fair_golden = golden[(golden["unfair"] == 0) | (golden["unfair"] == 1)]
    # unfair_golden = golden[(golden['ori.sc_id'] != golden['mut.sc_id']) & (golden['ori.sc_id'] >= 0) & (golden['mut.sc_id'] >= 0)]
    golden = auto_unfair_mark(golden)

    counter = Counter()
    for i, (index, row) in enumerate(golden.iterrows()):
        poly_word, poly_index = row["poly_word"].split("-")
        counter[poly_word] += 1
    
    print(counter.most_common(30))

    unfair_counter = {}
    for i, (index, row) in enumerate(golden.iterrows()):
        poly_word, poly_index = row["poly_word"].split("-")
        if poly_word not in unfair_counter:
            unfair_counter[poly_word] = []
        unfair_counter[poly_word].append(row["unfair"])
    
    for k, v in unfair_counter.items():
        if counter[k] == 0 or counter[k] > 10000 or sum(v) == 0 or len(sense_dict[k]) <= 1:
            continue
        draw_df["model"].append(model)
        draw_df["mutation"].append(mutation_type)

        draw_df["word"].append(k)
        draw_df["count"].append(counter[k])
        draw_df["bugs"].append(sum(v) / len(v))
        draw_df["poly_num"].append(len(sense_dict[k]))
    
def discussion():
    if os.path.exists("discussion3.csv"):
        draw_df = pd.read_csv("discussion3.csv")
    else:
        draw_df = {
            "model": [],
            "mutation": [],
            "word": [],
            "count": [],
            "bugs": [],
            "poly_num": [],
        }
        for model in MODELS:
            for mut in MUTATION_TYPE:
                data_mining(mut, model, draw_df)
                # dep(model)
        
        draw_df = pd.DataFrame(draw_df)
        draw_df.to_csv("discussion3.csv")

    new_draw_df = draw_df.groupby("count").agg("mean")
    new_draw_df['count'] = new_draw_df.index

    bins = [0, 2000, 4000, 6000, 8000, 10000]
    new_draw_df['Number'] = (np.select([new_draw_df['count'].between(i, j, inclusive='right') 
                            for i,j in zip(bins, bins[1:])], 
                            ["0-2000", "2000-4000", "4000-6000", "6000-8000", "8000-10000"]))
    new_draw_df['Bugs'] = new_draw_df['bugs']
    sns.boxplot(data=new_draw_df, x="Number", y="Bugs")
    plt.savefig("discussion3_count.pdf")
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close() # Close a figure window
    
    new_draw_df2 = draw_df.copy()
    new_draw_df2["Sense Number"] = new_draw_df2["poly_num"]
    new_draw_df2["Bugs"] = new_draw_df2["bugs"]
    sns.histplot(data=new_draw_df2, x="Sense Number", y="Bugs", bins=(16, 16), discrete=(True, False), log_scale=(False, False),
        cbar=True, thresh=None, cmap="crest")
    plt.savefig("discussion3_poly.pdf")
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close() # Close a figure window

if __name__ == "__main__":
    discussion()
