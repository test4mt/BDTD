
import pandas as pd
# Kick out 'guy', 'master' to avoid weird mutation
GENDER_SPECIFIC_WORDS = pd.read_csv("./asset/gender_specific_words.csv")
MALE_WORDS = list(GENDER_SPECIFIC_WORDS["male"].values)
FEMALE_WORDS = list(GENDER_SPECIFIC_WORDS["female"].values)

MTF = {MALE_WORDS[i]:FEMALE_WORDS[i] for i in range(len(GENDER_SPECIFIC_WORDS))}
FTM = {FEMALE_WORDS[i]:MALE_WORDS[i] for i in range(len(GENDER_SPECIFIC_WORDS))}

MALE_WORDS_SET = set(MALE_WORDS)
FEMALE_WORDS_SET = set(FEMALE_WORDS)

def is_male_word(word: str) -> bool:
    return word in MALE_WORDS_SET

def is_female_word(word: str) -> bool:
    return word in FEMALE_WORDS_SET

def swap_gender(word: str) -> str:
    if is_male_word(word):
        return MTF[word]
    elif is_female_word(word):
        return FTM[word]
    else:
        return word
