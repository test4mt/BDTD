
from typing import List, Tuple

import spacy
from spacy.tokens import Token
import nltk

from fairwsd4mt.mutation.syntax import find_child_index, find_children_indices

from .irregular import VERB_PAST, VERB_PAST_PARTICIPLE, PAST_VERB, PAST_PARTICIPLE_VERB

from . import TenseType


def is_simple_past(word: str) -> bool:
    '''
    a simple past
    一般过去形式
    '''
    word = word.lower().strip()
    # Search irregular form first
    if word in PAST_VERB and PAST_VERB[word] != word:
        return True
    return word.endswith("ed")

def is_past_participle(word: str) -> bool:
    '''
    a past participle
    过去分词
    '''
    word = word.lower().strip()
    # Search irregular form first
    if word in PAST_VERB and PAST_VERB[word] != word:
        return True
    return word.endswith("ed")

class TenseDetector(object):
    def __init__(self) -> None:
        self.en_nlp = spacy.load('en_core_web_sm')
        self.lemma = nltk.wordnet.WordNetLemmatizer()
    

    def is_present_participle(self, word: str) -> bool:
        '''
        a present participle or a gerund
        现在分词或者动名词
        '''
        word = word.lower().strip()
        lemma_word = self.lemma.lemmatize(word, "v")
        if lemma_word.endswith("ing"):
            if lemma_word == word:
                return False
            else:
                return True
        else:
            return word.endswith("ing")

    def detect_tense(self, sentence: str) -> List[Tuple[int, TenseType]]:
        doc = self.en_nlp(sentence)
        original_tokens = [token.text_with_ws for token in doc]
        original_tags = [token.pos_ for token in doc]
        original_deps = [token.dep_ for token in doc]
        original_head = [token.head for token in doc]

        # A verb for a tense
        verb_indices = []
        for i, token in enumerate(original_tokens):
            if original_deps[i] in ["ROOT", "conj"]:
                verb_indices.append(i)

        # detect tense for each verb
        ret = []
        for index in verb_indices:
            t = self.detect_tense_verb(original_tokens, original_tags, original_deps, original_head, index)
            ret.append((index, original_tokens[index], t))
        
        return ret
        
    def detect_tense_verb(
            self,
            original_tokens: List[str],
            original_tags: List[str],
            original_deps: List[str],
            original_head: List[Token],
            verb_index: int) -> TenseType:
        verb = original_tokens[verb_index].strip()

        auxpass_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="auxpass")
        is_pass = auxpass_index != -1
        
        if is_pass:
            return self.passive_tense(original_tokens, original_tags, original_deps, original_head, verb_index)
        else:
            return self.active_tense(original_tokens, original_tags, original_deps, original_head, verb_index)

    def passive_tense(self,
            original_tokens: List[str],
            original_tags: List[str],
            original_deps: List[str],
            original_head: List[Token],
            verb_index: int) -> TenseType:
        verb = original_tokens[verb_index].strip()
        aux_index = find_child_index(original_deps, original_head, verb_index, "auxpass")
        if aux_index == -1:
            if is_simple_past(verb) or is_past_participle(verb):
                return TenseType.SimplePast
            else:
                return TenseType.SimplePresent
        else:
            aux_token = original_tokens[aux_index]
            cleaned_aux_token = aux_token.strip().lower()
            if cleaned_aux_token in ["am", "is", "are", "'m", "'s"]:
                return TenseType.SimplePresent
            elif cleaned_aux_token in ["was", "were"]:
                return TenseType.SimplePast

    def active_tense(self,
            original_tokens: List[str],
            original_tags: List[str],
            original_deps: List[str],
            original_head: List[Token],
            verb_index: int) -> TenseType:
        verb = original_tokens[verb_index].strip()

        aux_indices = find_children_indices(original_deps, original_head, verb_index, "aux")
        if len(aux_indices) == 0:
            if is_simple_past(verb) or is_past_participle(verb):
                return TenseType.SimplePast
            else:
                return TenseType.SimplePresent
        elif len(aux_indices) == 1:
            aux_index = aux_indices[0]
            aux_token = original_tokens[aux_index]
            cleaned_aux_token = aux_token.strip().lower()
            if cleaned_aux_token in ["am", "is", "are", "'m", "'s"]:
                return TenseType.PresentProgressive
            elif cleaned_aux_token in ["was", "were"]:
                return TenseType.PastProgressive
            elif cleaned_aux_token in ["will", "wo", "'ll"]:
                return self.dispatch_active_future(original_tokens, original_tags, original_deps, original_head, aux, verb_index)
            elif cleaned_aux_token in ["would"]:
                pass
            elif cleaned_aux_token in ["have", "has"]:
                return TenseType.SimplePresentPerfect
            elif cleaned_aux_token in ["had"]:
                return TenseType.SimplePastPerfect
            elif cleaned_aux_token in ["did"]:
                return TenseType.SimplePast
            elif cleaned_aux_token in ["do"]:
                return TenseType.SimplePresent
            else:
                if is_simple_past(original_tokens[verb_index]) or is_past_participle(original_tokens[verb_index]):
                    return TenseType.SimplePast
                else:
                    return TenseType.SimplePresent
        elif len(aux_indices) == 2:
            first_aux_index, second_aux_index = aux_indices[0], aux_indices[1]
            first_aux_token, second_aux_token = original_tokens[first_aux_index].lower().strip(), original_tokens[second_aux_index].lower().strip()
            if first_aux_token in ["will", "wo", "'ll"] and second_aux_token == "be":
                return TenseType.FutureProgressive
        
        return None
    def dispatch_active_future(self,
            original_tokens: List[str],
            original_tags: List[str],
            original_deps: List[str],
            original_head: List[Token],
            aux: List[int],
            verb_index: int) -> TenseType:
        
        if len(aux) == 1:
            return TenseType.WillFuture
        elif len(aux) == 2:
            first_aux = original_tokens[aux[0]].lower().strip()
            second_aux = original_tokens[aux[1]].lower().strip()
            if second_aux == "be":
                return TenseType.FutureProgressive
            elif second_aux == "have":
                return TenseType.SimpleFuturePerfect
            else:
                raise Exception("Unknown future aux with two aux")
        elif len(aux) == 3:
            first_aux = original_tokens[aux[0]].lower().strip()
            second_aux = original_tokens[aux[1]].lower().strip()
            third_aux = original_tokens[aux[2]].lower().strip()
            if second_aux in ["have", "has"] and third_aux == "been":
                return TenseType.FuturePerfectProgressive
            else:
                raise Exception("Unknown future aux with three aux")
        else:
            raise Exception("Unknown future aux with more than three aux")
