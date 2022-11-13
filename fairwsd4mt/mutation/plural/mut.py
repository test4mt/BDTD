


from sys import prefix
from fairwsd4mt.mutation.sentence import MutationSentence
from ..syntax import find_child_index, recover_word
from . import is_verb_plural, is_verb_singular, pluralize, singularize, is_word_plural, is_word_singular, pluralize_verb, singularize_verb
from .det import starts_with_vowel_sound

def mutate_singular_to_plural(sentence: MutationSentence, noun_index: int, original_pos, original_head, original_deps):
    token = sentence[noun_index]
    det_index = find_child_index(original_deps, original_head, noun_index, "det")
    nount_token = sentence[det_index].strip().lower()
    if det_index != -1 and nount_token in ["a", "an"]: # "this", "that"可以修饰不可数
        sentence[noun_index] = pluralize(token)
        sentence.delete_token(det_index)

        if original_deps[noun_index] == "nsubj":
            parent_index = original_head[noun_index].i
            parent_token = sentence[parent_index]
            cleaned_parent_token = parent_token.lower().strip()
            # search have had has
            verb_aux_token_index = find_child_index(original_deps, original_head, parent_index, "aux")
            if verb_aux_token_index != -1:
                verb_aux_token = sentence[verb_aux_token_index].lower().strip()
                if verb_aux_token == "has":
                    sentence[verb_aux_token_index] = recover_word("have", sentence[verb_aux_token_index])
                else:
                    sentence[verb_aux_token_index] = recover_word(pluralize_verb(verb_aux_token), sentence[verb_aux_token_index])
            else:
                if is_verb_singular(cleaned_parent_token):
                    sentence[parent_index] = recover_word(pluralize_verb(cleaned_parent_token), parent_token)
        elif original_deps[noun_index] == "nsubjpass":
            parent_index = original_head[noun_index].i
            auxpass_index = find_child_index(original_deps, original_head, parent_index, "auxpass")
            auxpass_token = sentence[auxpass_index]
            cleaned_auxpass_token = auxpass_token.lower().strip()
            if is_verb_singular(cleaned_auxpass_token):
                sentence[auxpass_index] = recover_word(pluralize_verb(cleaned_auxpass_token), auxpass_token)
    else:
        return False
    return True

def mutate_plural_to_singular(sentence: MutationSentence, noun_index: int, original_pos, original_head, original_deps):
    token = sentence[noun_index]
    det_index = find_child_index(original_deps, original_head, noun_index, "det")
    amod_index = find_child_index(original_deps, original_head, noun_index, "amod")

    prefix_token = None
    prefix_index = None
    if det_index != -1:
        det_token = sentence[det_index].strip().lower()
        if det_token in ["some", "these", "those"]:
            prefix_token = det_token
            prefix_index = det_index
        else:
            return False
    elif amod_index != -1:
        amod_token = sentence[amod_index].strip().lower()
        if amod_token in ["many"]:
            prefix_token = amod_token
            prefix_index = amod_index
        else:
            return False
    else:
        return False

    sentence[noun_index] = singularize(token)
    sentence.delete_token(prefix_index)

    if prefix_token in ["many", "some"]:
        if starts_with_vowel_sound(token.lower().strip()):
            sentence.insert_tokens(prefix_index, ["an "])
        else:
            sentence.insert_tokens(prefix_index, ["a "])
    elif prefix_token in ["these", "those"]:
        sentence.insert_tokens(prefix_index, ["the "])
    
    if original_deps[noun_index] == "nsubj":
        parent_index = original_head[noun_index].i
        parent_token = sentence[parent_index]
        cleaned_parent_token = parent_token.lower().strip()
        # search have had has
        verb_aux_token_index = find_child_index(original_deps, original_head, parent_index, "aux")
        if verb_aux_token_index != -1:
            verb_aux_token = sentence[verb_aux_token_index].lower().strip()
            if verb_aux_token == "have":
                sentence[verb_aux_token_index] = recover_word("has", sentence[verb_aux_token_index])
            else:
                sentence[verb_aux_token_index] = recover_word(singularize_verb(verb_aux_token), sentence[verb_aux_token_index])
        else:
            if is_verb_plural(cleaned_parent_token):
                sentence[parent_index] = recover_word(singularize_verb(cleaned_parent_token), parent_token)
    elif original_deps[noun_index] == "nsubjpass":
        parent_index = original_head[noun_index].i
        auxpass_index = find_child_index(original_deps, original_head, parent_index, "auxpass")
        auxpass_token = sentence[auxpass_index]
        cleaned_auxpass_token = auxpass_token.lower().strip()
        if is_verb_plural(cleaned_auxpass_token):
            sentence[auxpass_index] = recover_word(singularize_verb(cleaned_auxpass_token), auxpass_token)
    
    return True
