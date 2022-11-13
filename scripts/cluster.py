import sys
import json
import spacy
import argparse
import opencc


def _clean_clusters(bn_senses_path, clean_path):
    with open(bn_senses_path, 'r', encoding='utf8') as in_fo:
        original_clusters = json.load(in_fo)
    print('Cleaning BabelNet clusters ...')

    # Convert traditional chinese to simplified chinese
    # Remove source glosses not from WN/WIKI
    for term in original_clusters.keys():
        for cs_id in original_clusters[term].keys():
            original_clusters[term][cs_id][2] = [cc.convert(s) for s in original_clusters[term][cs_id][2]]
            original_clusters[term][cs_id][3] = [sgl[0] for sgl in original_clusters[term][cs_id][3] if
                                                 sgl[1] in ['WN', 'WIKI']]
            original_clusters[term][cs_id][4] = [cc.convert(tgl[0]) for tgl in original_clusters[term][cs_id][4]]

    # Try to avoid duplicate and too specific senses, and non-Chinese sense or non-English lemma
    # Remove target sense clusters which are deemed too specific
    clean_clusters = dict()
    clusters_kept = 0
    clusters_dropped = 0
    for term in original_clusters.keys():
        for cs_id in original_clusters[term].keys():
            senses = set()
            lemmas = set()
            for sense in original_clusters[term][cs_id][2]:
                for s in sense.split('_'):
                    is_chinese = True
                    # filter non-Chinese sense
                    for w in s:
                        if not '\u4e00' <= w <= '\u9fff':
                            is_chinese = False
                    # sense whose length over 4 is deemed too specific
                    if len(s) <= 5 and is_chinese:
                        senses.add(s)
            for lemma in original_clusters[term][cs_id][1]:
                is_english = True
                for le in lemma.split('_'):
                    if not le.isalpha():
                        is_english = False
                if is_english and lemma.lower() != term:
                    lemmas.add(lemma.lower())
            original_clusters[term][cs_id][1] = list(lemmas)
            original_clusters[term][cs_id][2] = list(senses)
            if len(senses) == 0:
                print(term + " : " + cs_id + " dropped")
                clusters_dropped += 1
                continue
            else:
                if not clean_clusters.get(term, None):
                    clean_clusters[term] = dict()
                clean_clusters[term][cs_id] = original_clusters[term][cs_id]
                clusters_kept += 1

        # Remove term if it only has one sense cluster
        if clean_clusters.get(term, None):
            num_clusters = len(clean_clusters[term])
            if num_clusters < 2:
                clusters_kept -= num_clusters
                clusters_dropped += num_clusters
                clean_clusters.pop(term)

    # Report casualties
    for term in original_clusters.keys():
        if term not in clean_clusters.keys():
            print('Filtered out BabelNet entry for term \'{:s}\''.format(term))
    print('Filtered out {:d} sense clusters in total, kept {:d}'.format(clusters_dropped, clusters_kept))

    with open(clean_path, 'w', encoding='utf8') as out_fo:
        json.dump(clean_clusters, out_fo, indent=3, sort_keys=True, ensure_ascii=False)
    return clean_clusters


def _get_tokens_and_lemmas(line, nlp):
    """ Helper function for obtaining word tokens and lemmas for the computation of the overlap between two strings. """
    line_nlp = nlp(line)
    line_tokens = list()
    line_lemmas = list()
    for tok in line_nlp:
        lexeme = nlp.vocab[tok.text]
        # remove stopwords and punctuation
        if not lexeme.is_stop and not lexeme.is_punct:
            line_tokens.append(tok)
            if tok.lemma_ == '-PRON-' or tok.lemma_.isdigit():
                line_lemmas.append(tok.lower_)
            else:
                line_lemmas.append(tok.lemma_.lower().strip())
    return line_tokens, line_lemmas


def check_overlap(list1, list2, nlp, absolute_threshold, relate_threshold):
    tokens = [set(), set()]
    lemmas = [set(), set()]
    _lists = [list1, list2]
    for idx in range(0, 2):
        for line in _lists[idx]:
            tmp_tokens, tmp_lemmas = _get_tokens_and_lemmas(line, nlp)
            tokens[idx] |= set(tmp_tokens)
            lemmas[idx] |= set(tmp_lemmas)
    tokens_overlap_size = len(tokens[0] & tokens[1])
    tokens_overlap_ratio = tokens_overlap_size / max(min(len(tokens[0]), len(tokens[1])), 1)
    lemmas_overlap_size = len(lemmas[0] & lemmas[1])
    lemmas_overlap_ratio = lemmas_overlap_size / max(min(len(lemmas[0]), len(lemmas[1])), 1)
    if tokens_overlap_size >= absolute_threshold or tokens_overlap_ratio > relate_threshold:
        print('Tokens Overlap size:', tokens_overlap_size)
        print('Tokens Overlap ratio:', tokens_overlap_ratio)
        print(tokens[0])
        print(tokens[1])
        print(tokens[0] & tokens[1])
        return True
    if lemmas_overlap_size >= absolute_threshold or lemmas_overlap_ratio > relate_threshold:
        print('Lemmas Overlap size:', lemmas_overlap_size)
        print('Lemmas Overlap ratio:', lemmas_overlap_ratio)
        print(lemmas[0])
        print(lemmas[1])
        print(lemmas[0] & lemmas[1])
        return True
    return False


def check_cluster_overlap(info1, info2):
    # Check if cluster has synonym overlap
    if check_overlap(info1[1], info2[1], src_nlp, 3, 0.5):
        print('SYNONYM OVERLAP MERGE!')
        return True
    # Check if cluster has sense overlap
    if check_overlap(info1[2], info2[2], tgt_nlp, 3, 0.5):
        print('SENSE OVERLAP MERGE!')
        return True
    # Check if cluster source glosses overlap
    if check_overlap(info1[3], info2[3], src_nlp, 5, 0.5):
        print('SOURCE GLOSS OVERLAP MERGE!')
        return True
    # Check if cluster target glosses overlap
    if check_overlap(info1[4], info2[4], tgt_nlp, 5, 0.5):
        print('TARGET GLOSS OVERLAP MERGE!')
        return True
    return False


def cluster_senses(entry):
    print('Clustering senses ...')
    iter_count = 0
    while True:
        iter_count += 1
        previous_entry_size = len(entry)
        print('Iteration {:d}'.format(iter_count))
        merged = list()
        cluster_ids = list(entry.keys())
        # Compute pairwise synset similarity through overlap check
        for idx, sc1_id in enumerate(cluster_ids):
            if sc1_id in merged:
                continue
            for sc2_id in cluster_ids[idx+1:]:
                # Continue if cluster as already been merged into another
                if (not entry.get(sc2_id, None)) or (not entry.get(sc1_id, None)):
                    continue
                # Only merge clusters with the same POS
                if entry[sc1_id][0] != entry[sc2_id][0]:
                    continue
                if check_cluster_overlap(entry[sc1_id], entry[sc2_id]):
                    parent_id = sc1_id
                    child_id = sc2_id
                    # Merge cluster representations
                    original_cluster = entry[parent_id]
                    merged.append(child_id)
                    for index in range(1, 5):
                        entry[parent_id][index] += entry[child_id][index]
                        entry[parent_id][index] = list(set(entry[parent_id][index]))
                    print('=' * 20)
                    print('Merged clusters {:s} and {:s}'.format(parent_id, child_id))
                    print('PARENT cluster senses: {}'.format(original_cluster[2]))
                    print('CHILD cluster senses: {}'.format(entry[child_id][2]))
                    print('PRODUCT cluster senses: {}'.format(entry[parent_id][2]))
                    print('=' * 20)
                    # Update round-specific merge tracker
                    entry.pop(child_id)
        if len(entry) == previous_entry_size:
            break


def refine_clusters(bn_senses_path, output_path, clean_path, simplified_path):
    """ Merges similar BabelNet clusters into super-clusters, either including or ignoring multi-word senses. """

    clean_clusters = _clean_clusters(bn_senses_path, clean_path)
    sense_map = dict()
    for term_id, term in enumerate(clean_clusters.keys()):
        print('Refining target sense clusters for source term \'{:s}\''.format(term))
        clean_entry = clean_clusters[term]
        if not sense_map.get(term, None):
            sense_map[term] = dict()
        cluster_senses(clean_entry)
        print('Updating sense map ...')
        for parent_id in clean_entry.keys():
            sense_map[term][parent_id] = dict()
            sense_map[term][parent_id]['[POS]'] = clean_entry[parent_id][0]
            sense_map[term][parent_id]['[SYNONYMS]'] = clean_entry[parent_id][1]
            sense_map[term][parent_id]['[SENSES]'] = clean_entry[parent_id][2]
            sense_map[term][parent_id]['[SOURCE GLOSSES]'] = clean_entry[parent_id][3]
            sense_map[term][parent_id]['[TARGET GLOSSES]'] = clean_entry[parent_id][4]

        # Report occasionally -- not very efficient
        if term_id > 0 and term_id % 10 == 0:
            total_number_of_clusters = 0
            total_number_of_senses = 0
            for t in sense_map.keys():
                total_number_of_clusters += len(sense_map[t])
                for s in sense_map[t].keys():
                    total_number_of_senses += len(sense_map[t][s]['[SENSES]'])
            print('Looked-up {:d} polysemous source terms; last one was \'{:s}\''.format(term_id, term))
            print('Average number of sense clusters per term = {:.3f}'
                  .format(total_number_of_clusters / term_id))
            print('Average number of senses per cluster = {:.3f}'
                  .format(total_number_of_senses / total_number_of_clusters))

    # Final report
    print('Done!')
    total_number_of_clusters = 0
    total_number_of_senses = 0
    for t in sense_map.keys():
        total_number_of_clusters += len(sense_map[t])
        for s in sense_map[t].keys():
            total_number_of_senses += len(sense_map[t][s]['[SENSES]'])
    print('Average number of sense clusters per term = {:.3f}'.
          format(total_number_of_clusters / len(sense_map.keys())))
    print('Average number of senses per cluster = {:.3f}'.
          format(total_number_of_senses / total_number_of_clusters))

    # Dump complete map to JSON
    with open(output_path, 'w', encoding='utf8') as out_fo:
        json.dump(sense_map, out_fo, indent=3, sort_keys=True, ensure_ascii=False)

    # Dump partial map for manual review to JSON
    manual_clusters = dict()
    for t in sense_map.keys():
        manual_clusters[t] = dict()
        for sc in sense_map[t].keys():
            manual_clusters[t][sc] = dict()
            manual_clusters[t][sc]['[POS]'] = sense_map[t][sc]['[POS]']
            manual_clusters[t][sc]['[SENSES]'] = sense_map[t][sc]['[SENSES]']
    with open(simplified_path, 'w', encoding='utf-8') as out_fo:
        json.dump(manual_clusters, out_fo, indent=3, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='sense inventories scraped which need to be clustered')
    parser.add_argument('--cluster_output', '-o', type=str, required=True,
                        help='file where coarse-grained sense inventories are saved')
    parser.add_argument('--clean_output', '-co', type=str, required=True,
                        help='file where intermediate clean sense inventories are saved')
    parser.add_argument('--simplified_output', '-so', type=str, required=True,
                        help='file where simplified coarse-grained sense inventories for manual review are saved')
    parser.add_argument('--lang_pair', '-l', type=str, default='en-zh',
                        help='language pair of the bitext; expected format is src-tgt')
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

    refine_clusters(args.input, args.cluster_output, args.clean_output, args.simplified_output)
