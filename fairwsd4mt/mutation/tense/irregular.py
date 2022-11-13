
'''
https://github.com/WithEnglishWeCan/generated-english-irregular-verbs/
'''
import json

IRREGULAR_VERB_PATH = "./asset/irregular.verbs.build.json"

with open(IRREGULAR_VERB_PATH, 'r', encoding='utf-8') as f:
    table = json.loads(f.read())

VERB_PAST = {key: item[0]["2"] for key, item in table.items()}
VERB_PAST_PARTICIPLE = {key: item[0]["3"] for key, item in table.items()}
PAST_VERB = {}
for key, value in VERB_PAST.items():
    for past in value:
        PAST_VERB[past] = key
PAST_PARTICIPLE_VERB = {}
for key, value in VERB_PAST_PARTICIPLE.items():
    for past in value:
        PAST_PARTICIPLE_VERB[past] = key

AUX_PRESENT_PAST = {
    "is": "was",
    "are": "were",
    "am": "was",
    "'m": "was",
    "can": "could",
    "do": "did",
    "does": "did",
}

AUX_PAST_PRESENT = {
    "was": "is",
    "were": "are",
    "could": "can",
    "did": "do",
}