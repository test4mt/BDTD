'''
一些指标和不公平性的相关性
'''

from html import entities
import sys
import os
from collections import Counter, defaultdict
from typing import List

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import pandas as pd
import spacy
import tqdm
import json
import scipy.stats
import numpy as np

spacy_map = {'en': 'en_core_web_sm'}
src_nlp = spacy.load(spacy_map['en'])

with open("asset/sense_inventory/sense_dict.json", "r", encoding="utf-8") as f:
    sense_dict = json.loads(f.read())

data_frame = {
    "mutation": [],
    "model": [],
    "metric": [],
    "pearson": [],
    "p": [],
}

def auto_unfair_mark(df):
    df['unfair'] = np.where((df['ori.sc_id'] != df['mut.sc_id']) & (df['ori.sc_id'] >= 0) & (df['mut.sc_id'] >= 0), 1, 0)
    df['inconsistency'] = np.where(df['ori.sc_id'] == df['mut.sc_id'], 0, 1)
    return df

def get_graph(doc):
    graph = defaultdict(list)
    for token_i, token in enumerate(doc):
        # print(token.text, token.dep_, token.head.text, token.head.pos_)
        for child in token.children:
            graph[token_i].append(child.i)
        graph[token_i].append(token.head.i)
    
    return graph
    

def get_tree_depth(tokens, polyword_i: int, graph, mut_words):
    min_nhop = 0
    queue = [(polyword_i, 0)]
    visited = set()
    while len(queue) > 0:
        token_i, nhop = queue.pop(0)
        if token_i in visited:
            continue
        visited.add(token_i)
        for child_i in graph[token_i]:
            queue.append((child_i, nhop+1))
            if tokens[child_i].lower() in mut_words:
                min_nhop = nhop + 1
                break
    return min_nhop


def data_mining():
    all_df_list = []
    for mut_type in ["gender", "negative", "plural", "tense"]:
        for model in ["bing", "google", "opus"]:
            print(mut_type, model)
            # golden_set_path = f'result/{model}/test-set-{mutation_type}_{model}_merged_bow_0.csv'
            golden_set_path = f'rq/rq4/golden-dataset-{model}-{mut_type}_bow_49.csv'
            golden = pd.read_csv(golden_set_path)
            all_df_list.append(golden)

    df = pd.concat(all_df_list)
    # df = auto_unfair_mark(df)

    unfair_and_fair_golden = df[(df["unfair"] == 0) | (df["unfair"] == 1)]

    out_df = unfair_and_fair_golden.copy()
    out_df["metric.tok_num"] = 0
    out_df["target.fair"] = 0

    src_line_nlps = src_nlp.pipe(unfair_and_fair_golden["ori.src"].values.tolist(), batch_size=1000, n_process=16)
    spacy_tokens = []
    spacy_pos = []
    spacy_entities = []
    spacy_doc = []
    for src_line_nlp in tqdm.tqdm(src_line_nlps, total=len(unfair_and_fair_golden)):
        original_tokens = [token.text_with_ws for token in src_line_nlp]
        original_pos = [token.pos_ for token in src_line_nlp]
        original_entities = [entity for entity in src_line_nlp.ents]

        spacy_tokens.append(original_tokens)
        spacy_pos.append(original_pos)
        spacy_entities.append(original_entities)
        spacy_doc.append(src_line_nlp)

    for i, (index, row) in enumerate(unfair_and_fair_golden.iterrows()):
        graph = get_graph(spacy_doc[i])
        tokens = row["tok.ori.src"].split(' ')
        out_df.loc[index, "metric.tok_num"] = len(tokens)
        out_df.loc[index, "metric.noun_num"] = len([pos for pos in spacy_pos[i] if pos == "NOUN"])
        out_df.loc[index, "metric.ent_num"] = len(spacy_entities[i])
        out_df.loc[index, "metric.org_num"] = len([ent for ent in spacy_entities[i] if ent.label_ == "ORG"])
        out_df.loc[index, "metric.gpe_num"] = len([ent for ent in spacy_entities[i] if ent.label_ == "GPE"])
        poly_word, polyword_i = row["poly_word"].split("-")
        polyword_i = int(polyword_i)
        out_df.loc[index, "metric.poly_num"] = len(sense_dict[poly_word])
        mut_words = eval(row["mutkind_word"])
        mut_words_out = []
        for w in mut_words:
            if isinstance(w, tuple):
                mut_words_out.append(w[0])
            else:
                mut_words_out.append(w)
        out_df.loc[index, "metric.tree_depth"] = get_tree_depth(tokens, polyword_i, graph, mut_words_out)
        out_df.loc[index, "target.fair"]= row["unfair"]

    print("metric.tok_num", scipy.stats.pearsonr(out_df["metric.tok_num"].values, out_df["target.fair"].values))
    print("metric.noun_num", scipy.stats.pearsonr(out_df["metric.noun_num"].values, out_df["target.fair"].values))
    print("metric.ent_num", scipy.stats.pearsonr(out_df["metric.ent_num"].values, out_df["target.fair"].values))
    print("metric.org_num", scipy.stats.pearsonr(out_df["metric.org_num"].values, out_df["target.fair"].values))
    print("metric.gpe_num", scipy.stats.pearsonr(out_df["metric.gpe_num"].values, out_df["target.fair"].values))
    print("metric.poly_num", scipy.stats.pearsonr(out_df["metric.poly_num"].values, out_df["target.fair"].values))
    print("metric.tree_depth", scipy.stats.pearsonr(out_df["metric.tree_depth"].values, out_df["target.fair"].values))

    def col_name(col):
        data_frame["mutation"].append(mut_type)
        data_frame["model"].append(model)
        data_frame["metric"].append(col)
        corr = scipy.stats.pearsonr(out_df[col].values, out_df["target.fair"].values)
        data_frame["pearson"].append(corr[0])
        data_frame["p"].append(corr[1])
    
    col_name("metric.tok_num")
    col_name("metric.noun_num")
    col_name("metric.ent_num")
    # col_name("metric.org_num")
    # col_name("metric.gpe_num")
    # col_name("metric.poly_num")
    col_name("metric.tree_depth")
    
def discussion():
    data_mining()
    
    df = pd.DataFrame(data_frame)
    df.to_csv("discussion2.csv")

if __name__ == "__main__":
    discussion()
