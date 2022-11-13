

from typing import List
from fairwsd4mt.mutation.plural import is_plural, is_word_plural
from fairwsd4mt.mutation.sentence import MutationSentence
from fairwsd4mt.mutation.syntax import find_child_index, recover_word
from fairwsd4mt.mutation.tense.detection import is_simple_past
from fairwsd4mt.mutation.tense.mut import past_to_present


def mutate_negative_to_positive(sentence: MutationSentence, verb_index: int, original_pos: List[str], original_head, original_deps):
    verb_token = sentence[verb_index]
    cleaned_verb_token = sentence[verb_index].strip().lower()

    neg_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="neg")
    is_neg = neg_index != -1

    if is_neg:
        sentence.delete_token(neg_index)
        aux_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type=["aux", "auxpass"])
        if aux_index != -1:
            aux_token = sentence[aux_index].strip().lower()
            if aux_token in ["do", "does", "did"]:
                sentence.delete_token(aux_index)
            if aux_token == "wo":
                sentence[aux_index] = recover_word("will", sentence[aux_index])
    else:
        return False
    return True

def mutate_positive_to_negative(sentence: MutationSentence, verb_index: int, original_pos: List[str], original_head, original_deps):
    verb_token = sentence[verb_index]
    cleaned_verb_token = sentence[verb_index].strip().lower()

    neg_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="neg")
    is_neg = neg_index != -1

    if not is_neg:
        cleaned_tokens = [tok.lower().strip() for tok in sentence]
        if "?" in cleaned_tokens or "what" in cleaned_tokens or \
                "why" in cleaned_tokens or "how" in cleaned_tokens:
            return False
        
        aux_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type=["aux", "auxpass"])
        if aux_index != -1:
            sentence.insert_tokens(aux_index + 1, ['not '])
            return True
        if original_pos[verb_index] == "AUX":
            sentence.insert_tokens(verb_index + 1, ['not '])
            return True

        nsubj_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="nsubj")
        if nsubj_index != -1:
            nsubj = sentence[nsubj_index].strip().lower()
            verb = sentence[verb_index].strip().lower()
            if is_simple_past(verb):
                sentence.insert_tokens(verb_index, ['did ', 'not '])
            else:
                if is_word_plural(nsubj) or nsubj in ["you", "we", "they"]:
                    sentence.insert_tokens(verb_index, ['do ', 'not '])
                else:
                    sentence.insert_tokens(verb_index, ['does ', 'not '])
            # 动词时态还原
            sentence[verb_index] = recover_word(past_to_present(cleaned_verb_token), verb_token)
            return True

        expl_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="expl")
        if expl_index == -1:
            # 祈使句
            if verb_index == 0:
                sentence.insert_tokens(verb_index, ['Do ', 'not '])
                sentence[verb_index] = sentence[verb_index].lower()
            else:
                sentence.insert_tokens(verb_index, ['do ', 'not '])
            return True
        else:
            return False
    else:
        return False
    return True