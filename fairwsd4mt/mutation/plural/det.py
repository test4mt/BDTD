import nltk  # $ pip install nltk
from nltk.corpus import cmudict  # >>> nltk.download('cmudict')

def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
    for syllables in pronunciations.get(word, []):
        return syllables[0][-1].isdigit()  # use only the first one