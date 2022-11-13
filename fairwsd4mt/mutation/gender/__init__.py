
import re
import spacy
from typing import List
from .vocab import swap_gender

def has_person_name(entities: List[spacy.tokens.Span]) -> bool:
    '''
    Kick out sentences which contain person name entity
    For examples: Louis Galicia said he could not think of anyone.
    '''
    return any([entity.label_ == 'PERSON' for entity in entities])


def replace_gender_token(token: str, pos: str) -> str:
    t = token.strip().lower()
    replacement = swap_gender(t)

    # Correctly replace 'her' with 'his' or 'him' by pos
    if t == "her":
        if pos == "DET":
            replacement = "his"
        elif pos == "PRON":
            replacement = "him"
    mutant_token = re.sub(t, replacement, token, flags=re.IGNORECASE)
    # Keep case
    if token.istitle():
        mutant_token = mutant_token.title()
    if token.isupper():
        mutant_token = mutant_token.upper()
    return mutant_token
