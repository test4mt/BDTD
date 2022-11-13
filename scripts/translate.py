import sys
import os
from typing import List, Literal

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import argparse
import json
import time
import datetime
from tenacity import retry, stop_after_attempt, wait_fixed

import pandas as pd
from tqdm import tqdm
from fairwsd4mt.translation.baidu import translate_text_with_baidu, translate_text_with_baidu_batch
from fairwsd4mt.translation.bing import translate_text_with_bing, translate_text_with_bing_batch
from fairwsd4mt.translation.google import translate_text_with_google, translate_text_with_google_batch
from fairwsd4mt.translation.opus import translate_text_with_opus, translate_text_with_opus_batch
from fairwsd4mt.utils import list_chunked

from fairwsd4mt.translation.bing import TRANSLATION_COUNTER as BING_TRANSLATION_COUNTER, REAL_TRANSLATION_COUNTER as BING_REAL_TRANSLATION_COUNTER
from fairwsd4mt.translation.google import TRANSLATION_COUNTER as GOOGLE_TRANSLATION_COUNTER, REAL_TRANSLATION_COUNTER as GOOGLE_REAL_TRANSLATION_COUNTER

from fairwsd4mt.translation import ModelType, ONLINE_MODELS, PARALLEL_MODELS

# SSL Verification error work around
# os.environ["CURL_CA_BUNDLE"]=""

@retry(stop=stop_after_attempt(14))
def auto_translate(src: str, model: ModelType) -> str:
    if src.strip() == "":
        return ""
    if model == "google":
        tgt = translate_text_with_google(src)
    elif model == "bing":
        tgt = translate_text_with_bing(src)
    elif model == "opus":
        tgt = translate_text_with_opus(src)
    elif model == "baidu":
        tgt = translate_text_with_baidu(src)
    else:
        raise ValueError("Unknown model: {}".format(model))
    return tgt

@retry(stop=stop_after_attempt(14))
def auto_translate_batch(src: List[str], model: ModelType) -> List[str]:
    if len(src) == 0:
        return []
    if model == "google":
        tgt = translate_text_with_google_batch(src)
    elif model == "bing":
        tgt = translate_text_with_bing_batch(src)
    elif model == "opus":
        tgt = translate_text_with_opus_batch(src)
    elif model == "baidu":
        tgt = translate_text_with_baidu_batch(src)
    else:
        raise ValueError("Unknown model: {}".format(model))
    return tgt

def translate(sentence_info_path: str, model: str, mutation_type: str):
    # Save tmp result to a json file
    model_result_path = "./result/{}/".format(model)
    if not os.path.exists(model_result_path):
        os.makedirs(model_result_path)
    tmp_result_path = os.path.join(model_result_path, sentence_info_path.split(".")[0] + '_' + model + '.json')
    output_path = os.path.join(model_result_path, sentence_info_path.split(".")[0] + f"_{model}.csv")
    if os.path.exists(tmp_result_path):
        with open(tmp_result_path, 'r', encoding='utf8') as trp:
            tmp_result = json.load(trp)
    else:
        df = pd.read_csv(sentence_info_path)
        tmp_result = dict()
        for (src, tok_src) in zip(df['ori.src'].values, df['tok.ori.src'].values):
            tmp_result[src] = dict()
            tmp_result[src]['TOK'] = tok_src
        for (src, tok_src) in zip(df['mut.src'].values, df['tok.mut.src'].values):
            tmp_result[src] = dict()
            tmp_result[src]['TOK'] = tok_src
        with open(tmp_result_path, 'w', encoding='utf8') as trp:
            json.dump(tmp_result, trp, ensure_ascii=False, indent=3, sort_keys=True)

    CHUNK_SIZE = 32
    SAVE_INTERVAL = 100 if model == "opus" else 100
    lst = sorted(list(tmp_result.keys()))
    chunked_list = list(list_chunked(lst, CHUNK_SIZE))
    for (chunk_index, chunk_src) in tqdm(enumerate(chunked_list), total=len(chunked_list)):
        need_translated_chunk_src = []
        for index, src in enumerate(chunk_src):
            if tmp_result[src].get("TGT", None) is not None:
                continue
            else:
                need_translated_chunk_src.append(src)
        
        if model in PARALLEL_MODELS:
            tgts = auto_translate_batch(need_translated_chunk_src, model)

        for index, src in enumerate(need_translated_chunk_src):
            if tmp_result[src].get("TGT", None) is not None:
                continue
            else:
                start = time.time()
                if model in PARALLEL_MODELS:
                    tgt = tgts[index]
                else:
                    tgt = auto_translate(src, model)
                tmp_result[src]["TGT"] = tgt
                tmp_result[src]["TIME"] = time.time() - start

                if model in ["baidu"]:
                    time.sleep(1)

        if model in ["google"]:
            time.sleep(0.1)
        if len(need_translated_chunk_src) > 0 and chunk_index % SAVE_INTERVAL == 0:
            with open(tmp_result_path, 'w', encoding='utf8') as rp:
                json.dump(tmp_result, rp, ensure_ascii=False, indent=3, sort_keys=True)

    with open(tmp_result_path, 'w', encoding='utf8') as rp:
        json.dump(tmp_result, rp, ensure_ascii=False, indent=3, sort_keys=True)
    
    # Save tmp result to csv
    src_lines = list()
    tokenized_src_lines = list()
    tgt_lines = list()
    total_time = 0.0
    for src_line in tmp_result.keys():
        src_lines.append(src_line)
        tokenized_src_lines.append(tmp_result[src_line]['TOK'])
        tgt_lines.append(tmp_result[src_line]['TGT'])
        total_time += tmp_result[src_line]['TIME']

    print("Translation by {} cost {} (second)".format(model, total_time))
    result = pd.DataFrame({'src': src_lines, 'tok.src': tokenized_src_lines, 'tgt': tgt_lines})
    result.to_csv(output_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_info_path', '-s', type=str, required=True, help='path to csv file containing mutated sentence and related info')
    parser.add_argument('--mutation_type', '-t', type=str, required=True, help='mutation type')
    parser.add_argument('--model', '-m', type=str, required=True) # opus, google, bing

    args = parser.parse_args()
    translate(args.sentence_info_path, args.model, args.mutation_type)