import argparse
import os.path
import subprocess
import time
import re
from typing import Callable, NamedTuple, Optional
from sacrebleu.tokenizers import tokenizer_13a
from tqdm import tqdm
import pandas as pd
from sacrebleu.tokenizers import tokenizer_zh

import os
# SSL Verification error work around
os.environ["CURL_CA_BUNDLE"]=""

# Generate alignment pairs from OpenSubtitles2018 and concatenate it to improve alignment quality
def helper():
    en_tokenize = tokenizer_13a.Tokenizer13a()

    os18_srcs = [re.sub('\\s+', ' ', line).strip() + '\n' for line in
                 open('./asset/corpus/os18.src.txt', 'r', encoding='utf8').readlines()]
    os18_tgts = [re.sub('\\s+', ' ', line).strip() + '\n' for line in
                 open('./asset/corpus/os18.trg.txt', 'r', encoding='utf8').readlines()]
    tok_os18_pairs = list()

    for (src, tgt) in tqdm(zip(os18_srcs, os18_tgts), total=len(os18_srcs)):
        tok_os18_src = en_tokenize(src)
        tok_os18_tgt = char_zh_tokenize(tgt)
        if len(tok_os18_src) == 0 or len(tok_os18_tgt) == 0:
            print(tok_os18_src)
            print(tok_os18_tgt)
            continue
        tok_os18_pair = tok_os18_src.strip('\n') + ' ||| ' + tok_os18_tgt.strip('\n') + '\n'
        tok_os18_pairs.append(tok_os18_pair)

    with open('./asset/os18.pair', 'w', encoding='utf8') as f:
        f.writelines(tok_os18_pairs)

def pd_iter_func(df: pd.DataFrame, criteria: Callable[[NamedTuple], bool]) -> Optional[NamedTuple]:
    for i, row in df.iterrows():
        if criteria(row):
            return row

# Incorporate assignment result and translation result
def merge_result(sentence_info_path, trans_align_path, merged_result_path):
    df = pd.read_csv(sentence_info_path)
    trans_align_df = pd.read_csv(trans_align_path)

    # build fast lookup mapping
    mapping = {}
    for i, row in tqdm(trans_align_df.iterrows(), total=len(trans_align_df)):
        src = row['src']
        if src not in mapping:
            mapping[src] = []
        mapping[src].append(i)

    ori_tgt_lines = list()
    mut_tgt_lines = list()
    tokenized_ori_tgt_lines = list()
    tokenized_mut_tgt_lines = list()
    ori_alignments = list()
    mut_alignments = list()
    for (idx, row) in tqdm(df.iterrows(), total=len(df)):
        ori_row = trans_align_df.iloc[mapping[row['ori.src']]] # trans_align_df[trans_align_df['src'] == row['ori.src']]
        mut_row = trans_align_df.iloc[mapping[row['mut.src']]] # trans_align_df[trans_align_df['src'] == row['mut.src']]
        ori_tgt_line = ori_row['tgt'].values[0]
        mut_tgt_line = mut_row['tgt'].values[0]
        tokenized_ori_tgt_line = ori_row['tok.tgt'].values[0]
        tokenized_mut_tgt_line = mut_row['tok.tgt'].values[0]
        ori_alignment = ori_row['alignment'].values[0]
        mut_alignment = mut_row['alignment'].values[0]
        ori_tgt_lines.append(ori_tgt_line)
        mut_tgt_lines.append(mut_tgt_line)
        ori_alignments.append(ori_alignment)
        mut_alignments.append(mut_alignment)
        tokenized_ori_tgt_lines.append(tokenized_ori_tgt_line)
        tokenized_mut_tgt_lines.append(tokenized_mut_tgt_line)
    df['ori.tgt'] = ori_tgt_lines
    df['mut.tgt'] = mut_tgt_lines
    df['tok.ori.tgt'] = tokenized_ori_tgt_lines
    df['tok.mut.tgt'] = tokenized_mut_tgt_lines
    df['ori.alignment'] = ori_alignments
    df['mut.alignment'] = mut_alignments
    df.to_csv(merged_result_path, index=False)


# Build alignment pairs of source text and target text divided by ||| and fast align
def align(sentence_info_path: str, input_path: str, src: str, tgt: str, alignment_tool_path: str):
    start = time.time()
    input_name_prefix = input_path.rsplit(".", maxsplit=1)[0]

    df = pd.read_csv(input_path)
    alignment_pair_path = input_name_prefix + '.pair'
    forward_alignment_result_path = input_name_prefix + '.forward.align'
    backward_alignment_result_path = input_name_prefix + '.backward.align'
    alignment_result_path = input_name_prefix + '.align'
    output_path = input_name_prefix + '_aligned.csv'
    merged_result_path = input_name_prefix + '_merged.csv'

    alignment_pair_lines = list()
    tokenized_src_lines = df[src].values
    tgt_lines = df[tgt].values.astype(str).tolist()
    # Deduplicate space for alignment
    tgt_lines = [re.sub('\\s+', ' ', line).strip() + '\n' for line in tgt_lines]
    tokenized_tgt_lines = [char_zh_tokenize(tgt_line) for tgt_line in tgt_lines]
    df['tok.tgt'] = tokenized_tgt_lines
    for tok_src_line, tok_tgt_line in zip(tokenized_src_lines, tokenized_tgt_lines):
        alignment_pair_line = tok_src_line.strip('\n') + ' ||| ' + tok_tgt_line.strip('\n') + '\n'
        alignment_pair_lines.append(alignment_pair_line)
    input_len = len(alignment_pair_lines)

    if not os.path.exists('./asset/os18.pair'):
        helper()
    with open('./asset/os18.pair', 'r', encoding='utf-8') as f:
        os18_helper_pairs = f.readlines()
    alignment_pair_lines.extend(os18_helper_pairs)
    with open(alignment_pair_path, 'w', encoding='utf-8') as f:
        f.writelines(alignment_pair_lines)

    # Fast-align and incorporate the result to sentence_info_path
    result = subprocess.run([os.path.join(alignment_tool_path, 'fast_align'), "-i", alignment_pair_path, "-d", "-o", "-v"], stdout=subprocess.PIPE)
    with open(forward_alignment_result_path, "w", encoding="utf-8") as f:
        f.write(result.stdout.decode('utf-8'))

    result = subprocess.run([os.path.join(alignment_tool_path, 'fast_align'), "-i", alignment_pair_path, "-d", "-o", "-v", "-r"], stdout=subprocess.PIPE)
    with open(backward_alignment_result_path, "w", encoding="utf-8") as f:
        f.write(result.stdout.decode('utf-8'))

    result = subprocess.run(
        [os.path.join(alignment_tool_path, 'atools'),
            "-i", forward_alignment_result_path,
            "-j", backward_alignment_result_path,
            "-c", "grow-diag-final-and"], stdout=subprocess.PIPE)
    with open(alignment_result_path, "w", encoding="utf-8") as f:
        f.write(result.stdout.decode('utf-8'))

    with open(alignment_result_path) as f:
        alignment_lines = f.readlines()
    df['alignment'] = alignment_lines[0:input_len]
    print('Saving result...')
    df.to_csv(output_path, index=False)
    print('merging result...')
    merge_result(sentence_info_path, output_path, merged_result_path)
    print('Total time cost (second): ', time.time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_info_path', '-s', type=str, required=True,
                        help='path to csv file containing mutated sentence and related info')
    parser.add_argument('--input_path', '-i', type=str, required=True,
                        help='path to csv file containing source side and target side')
    parser.add_argument('--src_side', '-src', type=str, required=True,
                        help='column keyword for tokenized source side in csv file')
    parser.add_argument('--tgt_side', '-tgt', type=str, required=True,
                        help='column keyword for target side in csv file')
    parser.add_argument('--alignment_tool_path', '-a', type=str, required=True)
    args = parser.parse_args()
    char_zh_tokenize = tokenizer_zh.TokenizerZh()
    align(args.sentence_info_path, args.input_path, args.src_side, args.tgt_side, args.alignment_tool_path)
