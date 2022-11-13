import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from typing import List, Tuple

import argparse
import json
import os.path
import re
import time
from collections import Counter
import numpy as np
import pandas as pd
from sacrebleu.tokenizers import tokenizer_zh
from tqdm import tqdm
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity, check_pairwise_arrays, normalize

# SSL Verification error work around
os.environ["CURL_CA_BUNDLE"]=""

# bow models
char_zh_tokenize = None
bow_lookup_table = None

# ftc models
fasttext_zh_300 = None
sense_embedding_table = None

# sbert models
sense_clusters = None
sbert_zh = None

def build_sense_cluster_bow(sense_clusters_path):
    with open(sense_clusters_path, 'r', encoding='utf8') as scp:
        sense_clusters = json.load(scp)
    sense_to_cluster_path = sense_clusters_path[:-5] + '_s2c.json'
    sense_to_cluster_table = dict()
    for term in sense_clusters.keys():
        # print('Looking-up the term \'{:s}\''.format(term))
        entry = dict()
        for sc_id in sense_clusters[term].keys():
            for sense in sense_clusters[term][sc_id]['[SENSES]']:
                nlp_sense = char_zh_tokenize(sense)
                for tok in nlp_sense.split():
                    if not entry.get(tok, None):
                        entry[tok] = [sc_id]
                    else:
                        entry[tok].append(sc_id)
        for k in entry.keys():
            entry[k] = list(set(entry[k]))
            # if len(entry[k]) > 1:
            # print('Sense {:s} is in multiple clusters'.format(k))
        sense_to_cluster_table[term] = entry
    with open(sense_to_cluster_path, 'w', encoding='utf8') as s2c:
        json.dump(sense_to_cluster_table, s2c, indent=3, sort_keys=True, ensure_ascii=False)
    return sense_to_cluster_table


def build_sense_cluster_ftc(sense_clusters_path):
    with open(sense_clusters_path, 'r', encoding='utf8') as scp:
        sense_clusters = json.load(scp)

    for poly_word in sense_clusters.keys():
        for sc_id in sense_clusters[poly_word].keys():
            entry = sense_clusters[poly_word][sc_id]
            mean_embed = np.mean([fasttext_zh_300.wv[sense] for sense in entry['[SENSES]']], axis=0)
            sense_clusters[poly_word][sc_id]['CENTROID'] = mean_embed
    return sense_clusters

def parse_alignment_line(alignment_line: str) -> List[Tuple[int, int]]:
    items = alignment_line.split(" ")
    ret = []
    for item in items:
        lhs, rhs = item.split("-")
        lhs, rhs = int(lhs), int(rhs)
        ret.append((lhs, rhs))
    return ret

# Extend alignment window to improve recall
def extend_window(poly_word_index: int, alignment_line: str, alignment_window_size: int, tgt_tokens: List[str]) -> List[int]:
    # src2tgt_alignments = re.findall(r'{:d}-([0-9]+)'.format(poly_word_index), alignment_line) # 正则表达式bug，例如 0-1会匹配到10-1
    alignment = parse_alignment_line(alignment_line)

    src2tgt_alignments = [rhs for lhs, rhs in alignment if poly_word_index == lhs]

    src2tgt_alignments = sorted(src2tgt_alignments)
    tgt_window = list()
    if alignment_window_size > 0:
        for src2tgt_alignment in src2tgt_alignments:
            min_tgt_idx = max(0, src2tgt_alignment - alignment_window_size)
            max_tgt_idx = min(len(tgt_tokens), src2tgt_alignment + alignment_window_size + 1)
            tgt_window.extend(range(min_tgt_idx, max_tgt_idx))
        tgt_window = list(set(tgt_window))
        tgt_window = sorted(tgt_window)
    else:
        tgt_window = src2tgt_alignments
    return tgt_window


def assign_sense_cluster_id_by_bow(poly_word: str, tgt_tokens: List[str], tgt_window: List[int]):
    tgt_senses = [(tgt_tokens[i], i) for i in tgt_window]
    tmp_sc_ids = [(bow_lookup_table[poly_word][t], i) for (t, i) in tgt_senses
                  if bow_lookup_table[poly_word].get(t, None)]
    if len(tmp_sc_ids) == 0:
        # The polysemous word's alignment result is not included in sense inventory
        return -2, tgt_senses

    # Find the most supported sc_id
    possible_sc_ids = list()
    for tmp_sc_id, position in tmp_sc_ids:
        for sc_id in tmp_sc_id:
            possible_sc_ids.append(sc_id)
    sc_id_count = Counter(possible_sc_ids)
    if len(sc_id_count) == 1:
        return int(sc_id_count.most_common(1)[0][0]), tgt_senses
    elif len(sc_id_count) > 1:
        top_two_most_common = sc_id_count.most_common(2)
        if top_two_most_common[0][1] > top_two_most_common[1][1]:
            return int(top_two_most_common[0][0]), tgt_senses
    # The polysemous word's alignment result is include in more than two sense cluster, and they have a tie
    return -3, tgt_senses


def assign_sense_cluster_id_by_ftc(poly_word: str, tgt_tokens: List[str], tgt_window: List[int]):
    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    tgt_senses = ''.join([tgt_tokens[i] for i in tgt_window])
    cleaned_tgt_senses = tgt_senses.replace("。", "").replace("，", "")
    if len(cleaned_tgt_senses) == 0:
        return -1, []
    else:
        tgt_senses = cleaned_tgt_senses

    if tgt_senses not in fasttext_zh_300.wv:
        tgt_embedding = (reduce(lambda x, y: x + y, [fasttext_zh_300.wv[c] for c in tgt_senses])) / len(tgt_senses)
    else:
        tgt_embedding = fasttext_zh_300.wv[tgt_senses]

    result_sim = -1.0
    result_id = -1
    for sc_id in sense_embedding_table[poly_word].keys():
        cur_similarity = cos_sim(sense_embedding_table[poly_word][sc_id]['CENTROID'], tgt_embedding)
        # print(sc_id, cur_similarity)
        if cur_similarity > result_sim:
            result_id = sc_id
            result_sim = cur_similarity
    return int(result_id), tgt_senses

def assign_sense_cluster_id_by_sbert(poly_word: str, tgt_tokens: List[str], tgt_window: List[int]):
    def cal_semantic_sim(tgt, _sense_sentence_list):
        _sense_sentence_list.append(tgt)
        # print(sense_sentence_list)
        embeddings = sbert_zh.encode(_sense_sentence_list)
        return cosine_similarity([embeddings[-1]], embeddings[0:-1])

    tgt_senses = ''.join([tgt_tokens[i] for i in tgt_window])
    sense_sentence_list = list()
    sc_ids = list(sense_clusters[poly_word].keys())
    for sc_id in sc_ids:
        sense_sentence = ''.join(sense_clusters[poly_word][sc_id]["[SENSES]"])
        sense_sentence_list.append(sense_sentence)

    similarities = cal_semantic_sim(tgt_senses, sense_sentence_list)[0].tolist()
    # print(similarities)
    result_id = sc_ids[similarities.index(max(similarities))]
    return int(result_id), tgt_senses

def assign_sense_cluster_id(poly_word: str, poly_word_index: int, alignment_line, tgt_tokens, alignment_window_size, method):
    tgt_window = extend_window(poly_word_index, alignment_line, alignment_window_size, tgt_tokens)
    # polysemous word is not aligned
    if len(tgt_window) == 0:
        return -1, []
    if method == 'bow':
        result = assign_sense_cluster_id_by_bow(poly_word, tgt_tokens, tgt_window)
    elif method == 'ftc':
        result = assign_sense_cluster_id_by_ftc(poly_word, tgt_tokens, tgt_window)
    elif method == 'sbert':
        result = assign_sense_cluster_id_by_sbert(poly_word, tgt_tokens, tgt_window)
    else:
        raise Exception("Method is not defined")
    return result

def parse_mut_align(input: str) -> List[Tuple[int, int]]:
    align = []
    for item in input.split(","):
        lhs, rhs = item.split("-")
        align.append((int(lhs), int(rhs)))
    return align

def assign(input_path: str, alignment_window_size: int, method: str):
    output_path = input_path.rsplit(".", maxsplit=1)[0] + '_' + '_'.join([method, str(alignment_window_size)]) + '.csv'
    df = pd.read_csv(input_path)

    start = time.time()
    print('Alignment window size: ', alignment_window_size)
    ori_sc_ids = list()
    mut_sc_ids = list()
    ori_tgt_senses = list()
    mut_tgt_senses = list()

    print('Assigning cluster id by {}...'.format(method))
    for (idx, row) in tqdm(df.iterrows(), total=len(df)):
        if not isinstance(row['tok.ori.tgt'], str) and np.isnan(row['tok.ori.tgt']) or \
                not isinstance(row['tok.mut.tgt'], str) and np.isnan(row['tok.mut.tgt']):
            ori_sc_id, ori_tgt_sense = -1, []
            mut_sc_id, mut_tgt_sense = -1, []
        else:
            poly_word, poly_word_index = row['poly_word'].split('-')
            poly_word_index = int(poly_word_index)

            ori_sc_id, ori_tgt_sense = assign_sense_cluster_id(poly_word, poly_word_index, row['ori.alignment'],
                                                            row['tok.ori.tgt'].split(), alignment_window_size,
                                                            method)
            align = parse_mut_align(row["mut_align"])
            mut_poly_word_index = poly_word_index
            for lhs_index, rhs_index in align:
                if lhs_index == poly_word_index:
                    mut_poly_word_index = rhs_index
                    break

            mut_sc_id, mut_tgt_sense = assign_sense_cluster_id(poly_word, mut_poly_word_index, row['mut.alignment'],
                                                            row['tok.mut.tgt'].split(), alignment_window_size,
                                                            method)
        ori_sc_ids.append(ori_sc_id)
        ori_tgt_senses.append(ori_tgt_sense)
        mut_sc_ids.append(mut_sc_id)
        mut_tgt_senses.append(mut_tgt_sense)

    df['ori.sc_id'] = ori_sc_ids
    df['ori.tgt_sense'] = ori_tgt_senses
    df['mut.sc_id'] = mut_sc_ids
    df['mut.tgt_sense'] = mut_tgt_senses
    df.to_csv(output_path, index=False)
    positive = df[(df['ori.sc_id'] >= 0) & (df['mut.sc_id'] >= 0) & (df['ori.sc_id'] != df['mut.sc_id'])]
    print('Report bugs: ', len(positive))
    print('Total time cost (second): ', time.time() - start)

def load_models(method: str, sense_inventory_path: str):
    if method == 'bow':
        global char_zh_tokenize, bow_lookup_table
        print("Building sense clusters bag-of-words...")
        char_zh_tokenize = tokenizer_zh.TokenizerZh()
        bow_lookup_table = build_sense_cluster_bow(sense_inventory_path)
    elif method == 'ftc':
        global fasttext_zh_300, sense_embedding_table
        from gensim.models.fasttext import load_facebook_model, _load_fasttext_format
        print('Loading FastText embedding model...')
        fasttext_zh_300 = load_facebook_model('~/embedding/cc.zh.300.bin')
        sense_embedding_table = build_sense_cluster_ftc(sense_inventory_path)
    elif method == 'sbert':
        global sense_clusters, sbert_zh
        from sentence_transformers import SentenceTransformer
        with open(sense_inventory_path, 'r', encoding='utf8') as scp:
            sense_clusters = json.load(scp)
        print('Loading SentenceTransformer...')
        sbert_zh = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    else:
        raise Exception('Method is not defined')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, required=True,
                        help='path to csv file containing mutated, translated and aligned sentence')
    parser.add_argument('--sense_inventory_path', '-si', type=str, default=None,
                        help='path to sense clusters')
    parser.add_argument('--alignment_window_size', '-as', type=int, default=0,
                        help='alignment window size for extension')
    parser.add_argument('--alignment_window_range', '-ar', type=str, default='',
                        help='alignment window range for extension')
    parser.add_argument('--method', '-me', type=str, required=True)

    args = parser.parse_args()

    load_models(args.method, args.sense_inventory_path)

    if args.alignment_window_range != "":
        range_start, range_end = args.alignment_window_range.split("-")
        range_start, range_end = int(range_start), int(range_end)
    else:
        range_start = args.alignment_window_size
        range_end = args.alignment_window_size + 1
    
    for win in range(range_start, range_end):
        assign(args.input_path, win, args.method)
