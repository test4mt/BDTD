import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import json
import re
import argparse
import pandas as pd
import spacy
from sacrebleu.tokenizers import tokenizer_zh
from tqdm import tqdm


def filter_polysemous_words(tokenized_sentence_nlp):
    contained_polysemous_words = list()
    name_entities = ' '.join([ent.text for ent in tokenized_sentence_nlp.ents])
    for idx, token in enumerate(tokenized_sentence_nlp):
        # Exclude name entities
        if token.text in name_entities:
            continue
        if token.lemma_.lower() in polysemous_words and token.pos_ == 'NOUN':
            contained_polysemous_words.append((token.lemma_.lower(), idx))
    return contained_polysemous_words

def clean_bitext(src_path, ref_path, min_len, max_len, max_len_ratio, output_dir):
    """ Preprocesses the given bitext and filter to obtain sentences containing polysemous words"""

    corpus_name = re.findall(r'.+/(.+)', ref_path)[0].split('.')[0]
    clean_csv_path = output_dir + '/' + corpus_name + '.csv'
    clean_src_lines = list()
    tokenized_src_lines = list()
    clean_ref_lines = list()
    tokenized_ref_lines = list()
    poly_words_lists = list()
    lines_kept = 0

    print('Cleaning corpora ...')
    # Remove duplicate space for alignment
    original_src_lines = [re.sub('\\s+', ' ', line).strip()+'\n' for line in
                          open(src_path, 'r', encoding='utf-8').readlines()]
    original_ref_lines = [re.sub('\\s+', ' ', line).strip()+'\n' for line in
                          open(ref_path, 'r', encoding='utf-8').readlines()]

    tokenized_src_line_nlps = src_nlp.pipe(original_src_lines, batch_size=1000, n_process=16)
    pbar = tqdm(enumerate(tokenized_src_line_nlps), total=len(original_src_lines))

    for (line_id, tokenized_src_line_nlp) in pbar:
        tokenized_src_line = ' '.join([tok.text for tok in tokenized_src_line_nlp])
        # Tokenize Chinese by characters
        tokenized_ref_line = char_zh_tokenize(original_ref_lines[line_id])
        tsl_len = len(tokenized_src_line.split())
        ttl_len = len(tokenized_ref_line.split())
        contained_polysemous_words = filter_polysemous_words(tokenized_src_line_nlp)
        if min_len <= tsl_len <= max_len and min_len <= ttl_len <= max_len and \
                max(tsl_len, ttl_len) / min(tsl_len, ttl_len) <= max_len_ratio and \
                len(contained_polysemous_words) > 0:
            for cpw in contained_polysemous_words:
                clean_src_lines.append(original_src_lines[line_id])
                clean_ref_lines.append(original_ref_lines[line_id])
                tokenized_src_lines.append(tokenized_src_line)
                tokenized_ref_lines.append(tokenized_ref_line)
                poly_words_lists.append(cpw[0]+'-'+str(cpw[1]))
            lines_kept += 1
        pbar.set_description("Kept {:d}".format(lines_kept))

    corpus_list = [corpus_name] * len(clean_src_lines)
    print('-' * 20)
    print('Saving result...')
    df = pd.DataFrame({'ori.src': clean_src_lines, 'ref': clean_ref_lines, 'tok.ori.src': tokenized_src_lines,
                       'tok.ref': tokenized_ref_lines, 'poly_word': poly_words_lists, 'corpus': corpus_list})
    df.to_csv(clean_csv_path, index=False)
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', '-s', type=str,
                        help='path to the source side of the parallel corpus to be cleaned', required=True)
    parser.add_argument('--ref_path', '-t', type=str,
                        help='path to the reference side of the parallel corpus to be cleaned', required=True)
    parser.add_argument('--max_len', type=int, default=50,
                        help='threshold for the maximum allowed sentence length to exclude too specific cases')
    parser.add_argument('--min_len', type=int, default=10,
                        help='threshold for the minimum allowed sentence length to avoid ambiguity')
    parser.add_argument('--max_len_ratio', type=int, default=2.0,
                        help='threshold for maximum allowed sentence length ratio')
    parser.add_argument('--output_directory', '-o', type=str, required=True,
                        help='path to directory containing the output file')
    parser.add_argument('--sense_inventory', '-si', type=str, required=True,
                        help='path to sense clusters')
    args = parser.parse_args()

    spacy_map = {'en': 'en_core_web_sm'}
    src_nlp = spacy.load(spacy_map['en'], disable=['parser', 'textcat'])
    with open(args.sense_inventory, 'r', encoding='utf8') as sip:
        sense_inventory = json.load(sip)
    polysemous_words = list(sense_inventory.keys())
    char_zh_tokenize = tokenizer_zh.TokenizerZh()
    clean_bitext(args.src_path, args.ref_path, args.min_len, args.max_len, args.max_len_ratio, args.output_directory)
