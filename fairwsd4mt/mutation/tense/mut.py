

from typing import List
from fairwsd4mt.mutation.sentence import MutationSentence
from fairwsd4mt.mutation.syntax import find_child_index, recover_word
from fairwsd4mt.mutation.tense import TenseType
from fairwsd4mt.mutation.tense.detection import TenseDetector

import spacy
from spacy.tokens import Token

import nltk

from fairwsd4mt.mutation.tense.irregular import AUX_PAST_PRESENT, VERB_PAST, AUX_PRESENT_PAST, PAST_VERB


EN_NLP = spacy.load('en_core_web_sm')
LEMMA = nltk.wordnet.WordNetLemmatizer()

def past_to_present(word: str) -> str:
    cleaned_word = word.strip().lower()
    if cleaned_word in AUX_PAST_PRESENT:
        return AUX_PAST_PRESENT[cleaned_word]
    if cleaned_word in PAST_VERB:
        return PAST_VERB[cleaned_word]
    
    ret = LEMMA.lemmatize(word.lower().strip(), "v")
    return ret

def present_to_past(word: str) -> str:
    cleaned_word = word.lower().strip()
    if cleaned_word == "are":
        return "were"
    elif cleaned_word == "is":
        return "was"
    elif cleaned_word == "am":
        return "was"
    token = LEMMA.lemmatize(cleaned_word, "v")
    if token in VERB_PAST:
        return VERB_PAST[token][0]
    if token.endswith("e"):
        return token + "d"
    else:
        return token + "ed"

def aux_past_to_present(word: str, subj: str) -> str:
    if subj.lower().strip() == "i":
        return "am"
    if word in AUX_PAST_PRESENT:
        return AUX_PAST_PRESENT[word]
    return word

def aux_present_to_past(word: str, subj: str) -> str:
    if subj.lower().strip() == "i":
        return "was"
    if word in AUX_PRESENT_PAST:
        return AUX_PRESENT_PAST[word]
    return word

def mutate_simple_past_to_simple_present(sentence: MutationSentence, verb_index: int, original_pos: List[str], original_head, original_deps):
    verb_token = sentence[verb_index]
    cleaned_verb_token = sentence[verb_index].strip().lower()

    auxpass_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="auxpass")
    is_pass = auxpass_index != -1

    if not is_pass:
        aux_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="aux")
        nsubj_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="nsubj")
        
        if aux_index != -1 and nsubj_index != -1:
            aux_token = sentence[aux_index].strip().lower()
            cleand_nsubj = sentence[nsubj_index].strip().lower()
            # 一般疑问句
            sentence[aux_index] = recover_word(aux_past_to_present(aux_token, cleand_nsubj), sentence[aux_index])
        else:
            if original_pos[verb_index] == "AUX" and nsubj_index != -1:
                cleand_nsubj = sentence[nsubj_index].strip().lower()
                # 特殊疑问句
                sentence[verb_index] = recover_word(aux_past_to_present(cleaned_verb_token, cleand_nsubj), verb_token)
            else:
                sentence[verb_index] = recover_word(past_to_present(cleaned_verb_token), verb_token)
    else:
        auxpass_token = sentence[auxpass_index].strip().lower()
        sentence[auxpass_index] = recover_word(past_to_present(auxpass_token), verb_token)

    return True

def mutate_simple_present_to_simple_past(sentence: MutationSentence, verb_index: int, original_pos: List[str], original_head, original_deps):
    verb_token = sentence[verb_index]
    cleaned_verb_token = sentence[verb_index].strip().lower()

    auxpass_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="auxpass")
    is_pass = auxpass_index != -1

    if not is_pass:
        aux_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="aux")
        nsubj_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="nsubj")
        
        if aux_index != -1 and nsubj_index != -1:
            aux_token = sentence[aux_index].strip().lower()
            cleand_nsubj = sentence[nsubj_index].strip().lower()
            if aux_token in ["should", "shall", "must"]:
                return False
            # 一般疑问句
            sentence[aux_index] = recover_word(aux_present_to_past(aux_token, cleand_nsubj), sentence[aux_index])
        else:
            if original_pos[verb_index] == "AUX" and nsubj_index != -1:
                cleand_nsubj = sentence[nsubj_index].strip().lower()
                # 特殊疑问句
                sentence[verb_index] = recover_word(aux_present_to_past(cleaned_verb_token, cleand_nsubj), verb_token)
            else:
                if cleaned_verb_token == "'s":
                    sentence[verb_index] = recover_word(' was', verb_token)
                elif cleaned_verb_token == "'re":
                    sentence[verb_index] = recover_word(' were', verb_token)
                else:
                    sentence[verb_index] = recover_word(present_to_past(cleaned_verb_token), verb_token)
    else:
        auxpass_token = sentence[auxpass_index].strip().lower()
        sentence[auxpass_index] = recover_word(present_to_past(auxpass_token), verb_token)

    return True