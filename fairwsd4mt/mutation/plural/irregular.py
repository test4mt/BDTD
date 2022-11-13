
'''
https://github.com/sindresorhus/irregular-plurals
'''
import json

IRREGULAR_FILE = "./asset/irregular-plurals.json"

with open(IRREGULAR_FILE, 'r', encoding='utf-8') as f:
    d = json.loads(f.read())

SINGULAR_PLURAL_DICT = d
PLURAL_SINGULAR_DICT = {v: k for k, v in SINGULAR_PLURAL_DICT.items()}


VERB_AUX_PLURAL_SINGULAR = {
    "are": "is",
    "were": "was",
    "do": "does",
    "have": "has",
}

VERB_AUX_SINGULAR_PLURAL = {v: k for k, v in VERB_AUX_PLURAL_SINGULAR.items()}
