import os
import re
import sys
import json
import spacy
import argparse
from babelnetpy.babelnet import BabelNet
import opencc

# Reference: Detecting Word Sense Disambiguation Biases in Machine Translation
# for Model-Agnostic Adversarial Attacks, Emelin et al. 2020

BABEL_KEY = "YOUR_BABEL_KEY" # insert your own key

def scrape_senses(polysemous_word_path, pos, src_lang, tgt_lang, output_path):
    """ Scrapes BabelNet sense synsets, de-duplicating them, and filtering out low-quality sense entries. """
    src_term_list = [t.strip() for t in open(polysemous_word_path, 'r', encoding='utf-8').readlines()]
    # Instantiate a BabelNet object and sense table
    bn = BabelNet(BABEL_KEY)
    print('Scraping BabelNet ...')
    print('Currently processing the \'{:s}\' input list ... '.format(pos))
    if os.path.isfile(output_path):
        with open(output_path, 'r', encoding='utf8') as in_fo:
            bn_clusters = json.load(in_fo)
        print('Partial sense-map loaded; continuing collecting sense clusters from last point of interruption.')
    else:
        bn_clusters = dict()
        print('Initializing a new sense map.')

    for term_id, term in enumerate(src_term_list):
        print('Looking up senses for the source language term \'{:s}\''.format(term))
        try:
            # Skip previously looked-up terms
            if term in bn_clusters.keys():
                continue
            ids = bn.getSynset_Ids(term, src_lang)
            # Skip empty entries
            if len(ids) == 0:
                print('Non synsets found for \'{:s}\''.format(term))
                continue
            # Extend sense map and avoid clashes
            if not bn_clusters.get(term, None):
                bn_clusters[term] = dict()
            else:
                print('Avoided adding a duplicate entry for \'{:s}\''.format(term))
                continue
            # Iterate over synsets
            sense_clusters = list()
            for id_entry in ids:
                synset_pos = id_entry['pos']
                if synset_pos != pos:
                    continue
                synsets = bn.getSynsets(id_entry.id, [src_lang, tgt_lang], change_lang=True)
                # POS, synonyms, senses, src_glosses, tgt_glosses
                curr_sense_cluster = [synset_pos, list(), list(), list(), list()]
                # Scan synset entries
                for synset in synsets:
                    # Exclude name entities
                    if synset['synsetType'] != 'CONCEPT':
                        continue
                    # Scan retrieved senses
                    for sense in synset['senses']:
                        # Extend current cluster; 'WIKITR' / 'WNTR' translations tend to be source copies;
                        # 'WIKIRED' senses are semantically related but often not direct translations
                        if sense['properties']['source'] in ['WIKITR', 'WNTR', 'WIKIRED']:
                            continue
                        if sense['properties']['language'] == src_lang:
                            # Exclude the term itself
                            print(sense)
                            maybe_syn = sense['properties']['simpleLemma']
                            if maybe_syn.strip().lower() != term.strip().lower():
                                curr_sense_cluster[1].append(sense['properties']['simpleLemma'])
                        if sense['properties']['language'] == tgt_lang:
                            print(sense)
                            if tgt_lang == 'ZH':
                                curr_sense_cluster[2].append(cc.convert(sense['properties']['simpleLemma']))
                            else:
                                curr_sense_cluster[2].append(sense['properties']['simpleLemma'])
                    # Retrieve glosses
                    for gloss_obj in synset['glosses']:
                        if gloss_obj['language'] == src_lang:
                            curr_sense_cluster[3].append((gloss_obj['gloss'], gloss_obj['source']))
                        if gloss_obj['language'] == tgt_lang:
                            if tgt_lang == 'ZH':
                                curr_sense_cluster[4].append((cc.convert(gloss_obj['gloss']), gloss_obj['source']))
                            else:
                                curr_sense_cluster[4].append((gloss_obj['gloss'], gloss_obj['source']))
                    # De-duplicate glosses
                    curr_sense_cluster[3] = list(set(curr_sense_cluster[3]))
                    curr_sense_cluster[4] = list(set(curr_sense_cluster[4]))
                # Extend cluster list only for those senses actually has sense
                if len(curr_sense_cluster[2]) > 0:
                    sense_clusters.append(curr_sense_cluster)
        except Exception as e:
            print('Encountered exception: {:s}'.format(e))
            break

        # De-duplicate synonyms and senses
        if len(sense_clusters) > 0:
            for sc_id, sc in enumerate(sense_clusters):
                unique_synonyms = dict()
                for syn in sc[1]:
                    if not unique_synonyms.get(syn.lower(), None):
                        unique_synonyms[syn.lower()] = re.sub('\\n', '_', syn)
                unique_synonyms = list(unique_synonyms.values())

                unique_senses = dict()
                for sense in sc[2]:
                    if not unique_senses.get(sense.lower(), None):
                        unique_senses[sense.lower()] = re.sub('\\n', '_', sense)
                unique_senses = list(unique_senses.values())
                if len(unique_senses) > 0:
                    bn_clusters[term][sc_id] = [sc[0], unique_synonyms, unique_senses, sc[3], sc[4]]

        # Dump partial table to JSON
        with open(output_path, 'w', encoding='utf8') as out_fo:
            json.dump(bn_clusters, out_fo, indent=3, sort_keys=True, ensure_ascii=False)

        # Report occasionally -- not very efficient
        if term_id > 0 and term_id % 10 == 0:
            total_number_of_clusters = 0
            total_number_of_senses = 0
            for t in bn_clusters.keys():
                total_number_of_clusters += len(bn_clusters[t])
                for s in bn_clusters[t].keys():
                    total_number_of_senses += len(bn_clusters[t][s][2])
            print('Looked-up {:d} polysemous source terms; last one was \'{:s}\''.format(term_id, term))
            print('Average number of sense clusters per term = {:.3f}'
                  .format(total_number_of_clusters / term_id))
            print('Average number of senses per cluster = {:.3f}'
                  .format(total_number_of_senses / total_number_of_clusters))

    # Final report
    print('Done!')
    total_number_of_clusters = 0
    total_number_of_senses = 0
    for t in bn_clusters.keys():
        total_number_of_clusters += len(bn_clusters[t])
        for s in bn_clusters[t].keys():
            total_number_of_senses += len(bn_clusters[t][s][2])
    print('Average number of sense clusters per term = {:.3f}'.
          format(total_number_of_clusters / len(bn_clusters.keys())))
    print('Average number of senses per cluster = {:.3f}'.
          format(total_number_of_senses / total_number_of_clusters))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='polysemous word list needed to be crawled')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='file path where the scraped sense table will be saved')
    parser.add_argument('--lang_pair', '-l', type=str, default='en-zh',
                        help='language pair of the bitext; expected format is src-tgt')
    parser.add_argument('--pos', '-p', type=str, required=True,
                        help='part of speech of concerned senses')
    args = parser.parse_args()

    # Instantiate processing pipeline
    src_lang_id, tgt_lang_id = args.lang_pair.strip().split('-')
    spacy_map = {'en': 'en_core_web_sm', 'zh': 'zh_core_web_sm'}
    try:
        src_nlp = spacy.load(spacy_map[src_lang_id], disable=['parser', 'ner', 'textcat'])
        tgt_nlp = spacy.load(spacy_map[tgt_lang_id], disable=['parser', 'ner', 'textcat'])
    except KeyError:
        print('SpaCy does not support the language {:s} or {:s}. Exiting.'.format(src_lang_id, tgt_lang_id))
        sys.exit(0)
    cc = opencc.OpenCC('t2s')

    scrape_senses(args.input, args.pos, src_lang_id.upper(), tgt_lang_id.upper(), args.output)
